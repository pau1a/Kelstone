import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3-4B")
TARGET_LAYER = 33
TARGET_TOKEN = 75
TOP_N_TO_VERIFY = 15

"""
Finding 6 found a contradiction: neuron 941, flagged by finding 4's
correlational ranking (activation x fixed output direction) as the
single largest individual contributor, turned out to have almost no
real causal effect when actually ablated (Δgap=+0.44, noise-level).
That leaves an open question this script answers directly: which
neuron(s), if any, ARE causally dominant?

Testing all 9728 neurons via one full forward pass each would work but
is slow. Instead: a single backward pass gives, for every neuron
simultaneously, a first-order estimate of "how much would the logit
gap change if I turned this neuron's contribution off" — this is
exactly what gradient x activation approximates (the same math as
finding 5, but applied to the MLP intermediate activation instead of
input embeddings). That ranks all 9728 neurons by ESTIMATED causal
effect in one pass. The top candidates from that estimate are then
verified with real ablation (an actual forward pass with that neuron's
contribution actually zeroed) — the same ground-truth method finding 6
used on neuron 941 — to confirm the estimate is trustworthy before
trusting the ranking.
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

for i in range(TARGET_TOKEN):
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    chosen_id = int(torch.argmax(output.logits[0, -1, :]).item())
    input_ids = torch.cat([input_ids, torch.tensor([[chosen_id]], device=device)], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1
    )

# --- Step 1: one backward pass, gradient x activation for every neuron ---
captured = {}


def capture_and_require_grad(module, inputs):
    x = inputs[0]
    x.retain_grad()
    captured["intermediate"] = x
    return None


hook_handle = model.model.layers[TARGET_LAYER].mlp.down_proj.register_forward_pre_hook(
    capture_and_require_grad
)

out = model(input_ids=input_ids, attention_mask=attention_mask)
last_logits = out.logits[0, -1, :].float()
decision_score = last_logits[CALCULATE_ID] - last_logits[ANSWER_ID]

model.zero_grad(set_to_none=True)
decision_score.backward()

intermediate = captured["intermediate"][0, -1, :].detach().float()
intermediate_grad = captured["intermediate"].grad[0, -1, :].detach().float()
hook_handle.remove()

# First-order estimate of "ablate this neuron" effect: removing neuron
# n's activation changes decision_score by approximately
# -activation[n] * gradient[n] (standard first-order Taylor expansion).
estimated_ablation_effect = -(intermediate * intermediate_grad)

top_estimated = torch.argsort(estimated_ablation_effect.abs(), descending=True)[:TOP_N_TO_VERIFY]

print(f"layer {TARGET_LAYER}, token {TARGET_TOKEN} — gradient-based causal effect estimate")
print(f"baseline decision score (logit gap) = {decision_score.item():.4f}\n")
print(f"top-{TOP_N_TO_VERIFY} neurons by |estimated ablation effect|:")
for n in top_estimated.tolist():
    print(f"  neuron {n:5d}: estimated Δgap = {estimated_ablation_effect[n].item():+.4f}")

# --- Step 2: verify the top candidates with real single-neuron ablation ---
down_proj_weight = model.model.layers[TARGET_LAYER].mlp.down_proj.weight  # [2560, 9728]


def make_zero_neuron_hook(neuron_idx):
    def hook(module, inputs, output):
        contribution = inputs[0][0, -1, neuron_idx] * down_proj_weight[:, neuron_idx]
        patched = output.clone()
        patched[0, -1, :] = patched[0, -1, :] - contribution.to(patched.dtype)
        return patched

    return hook


def real_ablation_gap(neuron_idx):
    handle = model.model.layers[TARGET_LAYER].mlp.down_proj.register_forward_hook(
        make_zero_neuron_hook(neuron_idx)
    )
    try:
        with torch.no_grad():
            o = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = o.logits[0, -1, :].float()
        return (logits[CALCULATE_ID] - logits[ANSWER_ID]).item()
    finally:
        handle.remove()


baseline_gap = decision_score.item()
print(f"\nverifying top-{TOP_N_TO_VERIFY} candidates with REAL ablation (ground truth):")
real_effects = {}
for n in top_estimated.tolist():
    real_gap = real_ablation_gap(n)
    real_delta = real_gap - baseline_gap
    real_effects[n] = real_delta
    est = estimated_ablation_effect[n].item()
    print(
        f"  neuron {n:5d}: estimated Δgap={est:+.4f}   real Δgap={real_delta:+.4f}   "
        f"{'match' if (est > 0) == (real_delta > 0) else 'SIGN MISMATCH'}"
    )

# Also re-verify neuron 941 from finding 6 for direct comparison.
n941_real = real_ablation_gap(941)
n941_estimated = estimated_ablation_effect[941].item()
print(
    f"\n  (for comparison) neuron 941: estimated Δgap={n941_estimated:+.4f}   "
    f"real Δgap={n941_real - baseline_gap:+.4f}"
)

most_causal = max(real_effects, key=lambda n: abs(real_effects[n]))
print(
    f"\nmost causally impactful neuron found: {most_causal} "
    f"(real Δgap={real_effects[most_causal]:+.4f})"
)

# Plot: estimated vs real, for the verified set. Points cluster tightly
# in places (e.g. 941/9, 60/317, 282/66), so naive fixed-offset labels
# overlap illegibly — instead, greedily nudge each label in the
# direction least crowded by its nearest already-placed neighbors.
verified_ns = list(real_effects.keys())
est_vals = [estimated_ablation_effect[n].item() for n in verified_ns]
real_vals = [real_effects[n] for n in verified_ns]

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(est_vals, real_vals, color="tab:purple", zorder=3)

candidate_offsets = [
    (10, 6), (10, -12), (-24, 6), (-24, -12),
    (10, 18), (-24, 18), (10, -24), (-24, -24),
]
placed_xy_px = []  # label positions already placed, in axes-pixel space
for n, ex, ry in zip(verified_ns, est_vals, real_vals):
    data_to_px = ax.transData.transform((ex, ry))
    best_offset, best_score = candidate_offsets[0], -1
    for dx, dy in candidate_offsets:
        label_px = (data_to_px[0] + dx, data_to_px[1] + dy)
        min_dist = min(
            (((label_px[0] - px) ** 2 + (label_px[1] - py) ** 2) ** 0.5 for px, py in placed_xy_px),
            default=1e9,
        )
        if min_dist > best_score:
            best_score = min_dist
            best_offset = (dx, dy)
    label_px = (data_to_px[0] + best_offset[0], data_to_px[1] + best_offset[1])
    placed_xy_px.append(label_px)
    ax.annotate(
        str(n),
        (ex, ry),
        fontsize=8,
        xytext=best_offset,
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.5, shrinkA=3, shrinkB=3),
    )

lims = [min(est_vals + real_vals) * 1.2, max(est_vals + real_vals) * 1.2]
ax.plot(lims, lims, color="gray", linestyle="--", linewidth=0.8, label="perfect agreement")
ax.axhline(0, color="lightgray", linewidth=0.6)
ax.axvline(0, color="lightgray", linewidth=0.6)
ax.set_xlabel("estimated Δgap (gradient x activation)")
ax.set_ylabel("real Δgap (actual ablation)")
ax.set_title(f"layer {TARGET_LAYER}: estimated vs. real per-neuron causal effect")
ax.legend()
fig.tight_layout()
out_path = "tinkers/01_1_bare_react_loop/outputs/all_neurons_estimated_vs_real.png"
fig.savefig(out_path, dpi=130)
plt.close(fig)
print(f"\nsaved: {out_path}")
