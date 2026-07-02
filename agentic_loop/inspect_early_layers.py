import re

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3-4B")

"""
v6's logit-lens chart showed P(CALCULATE)/P(ANSWER) at exactly 0.0 for
layers 0-32 on a linear 0.0-1.0 axis, then a sharp jump at layer 33.
That could mean the probability is genuinely ~0 that early, or just too
small to render on a linear scale next to values near 1.0 — a real
difference worth distinguishing. This script reproduces step 1's exact
decision point (token 75, prompt "What is 12 + 7?"), prints the raw
per-layer values, and re-plots on a log y-axis so anything above true
zero becomes visible.
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

# Regenerate greedily up to token 75 (step 1's decision point from v6),
# same as the manual loop does, so this is the identical hidden state.
TARGET_TOKEN = 75
for i in range(TARGET_TOKEN + 1):
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=(i == TARGET_TOKEN),
        )
    last_logits = output.logits[0, -1, :]
    chosen_id = int(torch.argmax(last_logits).item())

    if i == TARGET_TOKEN:
        break

    input_ids = torch.cat([input_ids, torch.tensor([[chosen_id]], device=device)], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1
    )

calc_probs, answer_probs = [], []
with torch.no_grad():
    for layer_hidden in output.hidden_states:
        last_token_hidden = layer_hidden[0, -1, :]
        normed = model.model.norm(last_token_hidden)
        layer_logits = model.lm_head(normed)
        layer_probs = torch.softmax(layer_logits, dim=-1)
        calc_probs.append(layer_probs[CALCULATE_ID].item())
        answer_probs.append(layer_probs[ANSWER_ID].item())

print(f"{'layer':>6}  {'P(CALCULATE)':>14}  {'P(ANSWER)':>14}")
for layer, (cp, ap) in enumerate(zip(calc_probs, answer_probs)):
    print(f"{layer:>6}  {cp:>14.2e}  {ap:>14.2e}")

EPS = 1e-12
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(
    range(len(calc_probs)),
    [max(p, EPS) for p in calc_probs],
    marker="o",
    color="tab:orange",
    label="P(CALCULATE)",
)
ax.plot(
    range(len(answer_probs)),
    [max(p, EPS) for p in answer_probs],
    marker="o",
    color="tab:blue",
    label="P(ANSWER)",
)
ax.set_yscale("log")
ax.set_xlabel("layer (0 = embeddings, last = final hidden state)")
ax.set_ylabel("probability (log scale, via logit lens)")
ax.set_title(f"step 1 — CALCULATE vs ANSWER through the layers @ token {TARGET_TOKEN} (log scale)")
ax.legend()
fig.tight_layout()
out_path = "tinkers/01_1_bare_react_loop/outputs/step1_token75_logitlens_logscale.png"
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"\nsaved: {out_path}")
