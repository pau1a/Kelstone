import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3-4B")
TARGET_LAYER = 33
TARGET_TOKEN = 75

"""
Findings 3-5 are all correlational: attention heads that attend to
CALCULATE/ANSWER tokens, MLP neurons whose activation correlates with
the logit gap, gradients that flag certain input tokens as influential.
None of that proves any of it actually CAUSES the model's decision —
correlation, not causation. This script tests causation directly via
activation patching: surgically edit layer 33's MLP output at the
decision token and see whether the final P(CALCULATE)/P(ANSWER) split
actually changes.

Three interventions, in increasing specificity:
1. Zero the entire layer-33 MLP output at the decision token (does this
   layer's MLP matter at all?)
2. Zero only neuron 941's contribution (finding 4's single strongest
   ANSWER-suppressing neuron) — does removing just this one neuron's
   effect measurably shift the split?
3. Scale neuron 941's activation to 2x — does amplifying it push
   further in the same direction, confirming the sign of its effect?

Implemented via a forward hook on the MLP module that rewrites its
output in place before it's added back to the residual stream.
"""

SYSTEM_PROMPT = """You are a careful assistant with access to one tool: a calculator.

To use it, respond with exactly this format on its own line:
CALCULATE: <a valid Python arithmetic expression>

Do not compute the result yourself. Wait for the tool result to be given
back to you before continuing. Once you have the result and are ready to
give the final answer, respond with exactly this format:
ANSWER: <your final answer>
"""

device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(device)
model.eval()

CALCULATE_ID = tokenizer.encode("CALCULATE", add_special_tokens=False)[0]
ANSWER_ID = tokenizer.encode("ANSWER", add_special_tokens=False)[0]
NEURON = 941  # finding 4: strongest ANSWER-suppressing neuron at layer 33

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is 12 + 7?"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

for i in range(TARGET_TOKEN):
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    chosen_id = int(torch.argmax(output.logits[0, -1, :]).item())
    input_ids = torch.cat([input_ids, torch.tensor([[chosen_id]], device=device)], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1
    )


def run_with_patch(patch_fn=None):
    """
    patch_fn, if given, receives (module, inputs, output) from a forward
    hook on layer 33's mlp.down_proj and returns a replacement output.
    Returns final P(CALCULATE), P(ANSWER) at the decision token.
    """
    handle = None
    if patch_fn is not None:
        handle = model.model.layers[TARGET_LAYER].mlp.down_proj.register_forward_hook(patch_fn)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        return (
            probs[CALCULATE_ID].item(),
            probs[ANSWER_ID].item(),
            logits[CALCULATE_ID].item(),
            logits[ANSWER_ID].item(),
        )
    finally:
        if handle is not None:
            handle.remove()


def zero_entire_mlp_output(module, inputs, output):
    patched = output.clone()
    patched[0, -1, :] = 0.0
    return patched


def make_zero_neuron_hook(neuron_idx):
    down_proj_weight = model.model.layers[TARGET_LAYER].mlp.down_proj.weight  # [2560, 9728]

    def hook(module, inputs, output):
        # inputs[0] is the 9728-dim intermediate activation. Recompute
        # this token's MLP output with just this one neuron zeroed out,
        # by subtracting its isolated contribution from the real output.
        intermediate = inputs[0][0, -1, :]  # [9728]
        neuron_contribution = intermediate[neuron_idx] * down_proj_weight[:, neuron_idx]
        patched = output.clone()
        patched[0, -1, :] = patched[0, -1, :] - neuron_contribution.to(patched.dtype)
        return patched

    return hook


def make_scale_neuron_hook(neuron_idx, scale):
    down_proj_weight = model.model.layers[TARGET_LAYER].mlp.down_proj.weight

    def hook(module, inputs, output):
        intermediate = inputs[0][0, -1, :]
        neuron_contribution = intermediate[neuron_idx] * down_proj_weight[:, neuron_idx]
        extra = neuron_contribution * (scale - 1.0)
        patched = output.clone()
        patched[0, -1, :] = patched[0, -1, :] + extra.to(patched.dtype)
        return patched

    return hook


def report(label, result):
    calc_p, ans_p, calc_logit, ans_logit = result
    print(
        f"{label:<28} P(CALCULATE)={calc_p:.6f}  P(ANSWER)={ans_p:.6f}  "
        f"logit(CALC)={calc_logit:+.3f}  logit(ANS)={ans_logit:+.3f}  "
        f"gap={calc_logit - ans_logit:+.3f}"
    )
    return calc_logit - ans_logit


baseline = run_with_patch(None)
base_gap = report("baseline:", baseline)

zeroed_mlp = run_with_patch(zero_entire_mlp_output)
gap = report(f"layer {TARGET_LAYER} MLP zeroed:", zeroed_mlp)
print(f"  -> gap change: {gap - base_gap:+.3f}")

zeroed_n = run_with_patch(make_zero_neuron_hook(NEURON))
gap = report(f"neuron {NEURON} zeroed:", zeroed_n)
print(f"  -> gap change: {gap - base_gap:+.3f}")

scaled_n = run_with_patch(make_scale_neuron_hook(NEURON, 2.0))
gap = report(f"neuron {NEURON} scaled 2x:", scaled_n)
print(f"  -> gap change: {gap - base_gap:+.3f}")

print(
    "\nP(CALCULATE)/P(ANSWER) were already saturated at baseline (softmax"
    " ceiling), so probability deltas round to 0 even if the underlying"
    " logit gap actually moved — the logit gap numbers above are the"
    " real signal to read, not the probabilities."
)
