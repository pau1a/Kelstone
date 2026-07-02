import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3-4B")
TARGET_TOKEN = 75

"""
The attention-head (finding 3) and MLP-neuron (finding 4) analyses both
looked at layer 33 specifically, chosen because the logit-lens showed
that's where the CALCULATE/ANSWER contest resolves. This script asks a
different question, spanning the whole network at once: which INPUT
tokens, at any position in the prompt/reasoning, most influenced the
final decision? Backprop from logit(CALCULATE) - logit(ANSWER) all the
way back to the input embeddings gives a gradient at every token
position. Gradient x input (a standard, cheap saliency method) turns
that into a per-token attribution score — this traces the decision back
to its trigger in the actual text, rather than to a layer/neuron inside
the network.

Requires inputs_embeds instead of input_ids so gradients can flow into
the embedding layer (input_ids are just integer indices — not
differentiable).
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

# Regenerate greedily up to the decision token (no grad needed here —
# we're just reconstructing the same context as the earlier scripts).
for i in range(TARGET_TOKEN):
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    chosen_id = int(torch.argmax(output.logits[0, -1, :]).item())
    input_ids = torch.cat([input_ids, torch.tensor([[chosen_id]], device=device)], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1
    )

# Now the real forward pass, with gradients, via inputs_embeds so the
# embedding output (not the discrete input_ids) is the leaf tensor we
# can backprop to.
embed_layer = model.model.embed_tokens
inputs_embeds = embed_layer(input_ids).detach().clone().requires_grad_(True)

output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
last_logits = output.logits[0, -1, :]
decision_score = last_logits[CALCULATE_ID] - last_logits[ANSWER_ID]

model.zero_grad(set_to_none=True)
decision_score.backward()

# Gradient x input, summed over the embedding dimension, at every token
# position — standard saliency attribution.
grad = inputs_embeds.grad[0]  # [seq, hidden]
embeds = inputs_embeds[0].detach()  # [seq, hidden]
saliency = (grad * embeds).sum(dim=-1).float()  # [seq]

seq_len = saliency.shape[0]
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
readable_tokens = [tokenizer.convert_tokens_to_string([t]).strip() or t for t in tokens]

print(f"decision score logit(CALCULATE) - logit(ANSWER) = {decision_score.item():.4f}")
print(f"sequence length {seq_len}\n")

top_pos = torch.argsort(saliency.abs(), descending=True)[:15]
print("top-15 input tokens by |gradient x input| saliency:")
for p in top_pos.tolist():
    tok = readable_tokens[p] if p < len(readable_tokens) else "?"
    print(f"  token {p:3d} ({tok!r}): saliency={saliency[p].item():+.4f}")

# Plot: saliency across the whole sequence.
fig, ax = plt.subplots(figsize=(12, 4.5))
colors = ["tab:orange" if s > 0 else "tab:blue" for s in saliency.tolist()]
ax.bar(range(seq_len), saliency.detach().cpu().numpy(), color=colors, width=1.0)
ax.set_xlabel("token position")
ax.set_ylabel("saliency (gradient x input, toward CALCULATE if positive)")
ax.set_title(f"gradient attribution for logit(CALCULATE) - logit(ANSWER) @ token {TARGET_TOKEN}")
fig.tight_layout()
out_path = "tinkers/01_1_bare_react_loop/outputs/gradient_attribution_full.png"
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"\nsaved: {out_path}")

# Zoomed plot with token labels around the highest-saliency region.
peak = int(saliency.abs().argmax().item())
window = range(max(0, peak - 20), min(seq_len, peak + 20))
window_labels = [readable_tokens[i] if i < len(readable_tokens) else "?" for i in window]
window_vals = saliency[list(window)].detach().cpu().numpy()
window_colors = ["tab:orange" if v > 0 else "tab:blue" for v in window_vals]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(window)), window_vals, color=window_colors)
ax.set_xticks(range(len(window)))
ax.set_xticklabels(window_labels, rotation=90, fontsize=7)
ax.set_ylabel("saliency (gradient x input)")
ax.set_title(f"gradient attribution, zoomed around peak (token {peak})")
fig.tight_layout()
out_path_zoom = "tinkers/01_1_bare_react_loop/outputs/gradient_attribution_zoom.png"
fig.savefig(out_path_zoom, dpi=120)
plt.close(fig)
print(f"saved: {out_path_zoom}")
