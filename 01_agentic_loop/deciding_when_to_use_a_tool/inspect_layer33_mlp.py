import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3-4B")
TARGET_LAYER = 33
TARGET_TOKEN = 75

"""
inspect_layer33_attention.py found which attention heads retrieve the
literal CALCULATE/ANSWER instruction tokens at layer 33 — the "where do
I look" half of the mechanism. This script looks at the "what do I
compute from it" half: layer 33's MLP block (SwiGLU, 9728 intermediate
neurons) is where attention's retrieved information actually gets
turned into a directional push toward one output token or the other.

Qwen3's MLP: down_proj(act_fn(gate_proj(x)) * up_proj(x)). The
intermediate activation (9728-dim, right before down_proj) is captured
via a forward hook. Each of its 9728 neurons has a fixed "output
direction" — its corresponding column in down_proj's weight matrix.
Projecting that direction through the model's own final norm + lm_head
gives each neuron's direct contribution to every vocab logit,
independent of what the activation value actually was. Multiplying by
the neuron's actual activation at the decision token gives each
neuron's real, signed contribution to logit(CALCULATE) - logit(ANSWER)
at this specific token — a direct per-neuron attribution, not just "big
activation = important."
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

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is 12 + 7?"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Hook layer 33's MLP intermediate activation: act_fn(gate_proj(x)) * up_proj(x),
# the 9728-dim vector immediately before down_proj projects it back down.
captured = {}


def capture_intermediate(module, inputs, output):
    # output here is down_proj's *input* only if we hook down_proj itself.
    captured["intermediate"] = inputs[0].detach()


hook_handle = model.model.layers[TARGET_LAYER].mlp.down_proj.register_forward_hook(
    capture_intermediate
)

# Regenerate greedily up to the decision token, identical to the
# earlier scripts, so this is the same hidden state / context.
for i in range(TARGET_TOKEN + 1):
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    last_logits = output.logits[0, -1, :]
    chosen_id = int(torch.argmax(last_logits).item())

    if i == TARGET_TOKEN:
        break

    input_ids = torch.cat([input_ids, torch.tensor([[chosen_id]], device=device)], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1
    )

hook_handle.remove()

# intermediate: [1, seq, 9728] -> take the decision token's activation vector
intermediate = captured["intermediate"][0, -1, :].float()  # [9728]

down_proj_weight = model.model.layers[TARGET_LAYER].mlp.down_proj.weight.float()  # [2560, 9728]

# Each neuron n's isolated output direction: down_proj_weight[:, n], a
# 2560-dim vector in residual-stream space. Passing that direction
# alone through the model's final norm + lm_head gives neuron n's fixed
# contribution to every vocab logit *per unit of activation*.
with torch.no_grad():
    unit_directions = down_proj_weight.T  # [9728, 2560]
    normed_directions = model.model.norm(unit_directions.to(device).to(torch.bfloat16))
    per_unit_logits = model.lm_head(normed_directions).float()  # [9728, vocab]

per_unit_calc = per_unit_logits[:, CALCULATE_ID]
per_unit_answer = per_unit_logits[:, ANSWER_ID]

# Real, signed contribution at this token = per-unit contribution *
# actual activation value.
contribution_calc = intermediate * per_unit_calc
contribution_answer = intermediate * per_unit_answer
contribution_diff = contribution_calc - contribution_answer

top_calc = torch.argsort(contribution_diff, descending=True)[:10]
top_answer = torch.argsort(contribution_diff, descending=False)[:10]

print(f"layer {TARGET_LAYER} MLP, token {TARGET_TOKEN}, intermediate dim {intermediate.shape[0]}")
print(f"\ntop-10 neurons pushing toward CALCULATE (logit_calc - logit_answer contribution):")
for n in top_calc.tolist():
    print(
        f"  neuron {n:5d}: activation={intermediate[n].item():+.3f}  "
        f"contrib(CALC)={contribution_calc[n].item():+.4f}  "
        f"contrib(ANS)={contribution_answer[n].item():+.4f}  "
        f"diff={contribution_diff[n].item():+.4f}"
    )

print(f"\ntop-10 neurons pushing toward ANSWER:")
for n in top_answer.tolist():
    print(
        f"  neuron {n:5d}: activation={intermediate[n].item():+.3f}  "
        f"contrib(CALC)={contribution_calc[n].item():+.4f}  "
        f"contrib(ANS)={contribution_answer[n].item():+.4f}  "
        f"diff={contribution_diff[n].item():+.4f}"
    )

net_diff = contribution_diff.sum().item()
print(f"\nsum of all 9728 neurons' diff contributions: {net_diff:+.4f}")
print("(the actual logit(CALCULATE) - logit(ANSWER) gap also includes the residual")
print(" stream's incoming value before this layer's MLP, so this won't match exactly —")
print(" it isolates just this layer's MLP's net directional push.)")

# Plot: sorted contribution_diff across all 9728 neurons — shows
# whether the push is concentrated in a few neurons or broadly spread.
sorted_diff, _ = torch.sort(contribution_diff, descending=True)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(range(len(sorted_diff)), sorted_diff.cpu().numpy())
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("neuron rank (sorted by contribution)")
ax.set_ylabel("logit(CALCULATE) - logit(ANSWER) contribution")
ax.set_title(f"layer {TARGET_LAYER} MLP — per-neuron directional push, sorted")
fig.tight_layout()
out_path = "tinkers/01_1_bare_react_loop/outputs/layer33_mlp_neuron_contributions.png"
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"\nsaved: {out_path}")
