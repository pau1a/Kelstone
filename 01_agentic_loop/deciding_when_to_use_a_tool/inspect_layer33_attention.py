import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3-4B")
TARGET_LAYER = 33
TARGET_TOKEN = 75

"""
inspect_early_layers.py found the CALCULATE/ANSWER decision resolves in
a brief competitive window at layer 33 (both probabilities jump
together) before layer 34+ suppresses ANSWER and commits to CALCULATE.
That's a logit-lens view — it shows *what* the hidden state encodes at
each layer, not *how* layer 33 arrives at it. This script goes one
level deeper: pull layer 33's raw self-attention weights (32 heads) at
the exact same decision point (token 75, same prompt/reasoning as
before) and see which heads are attending most strongly to which prior
tokens when the model is about to commit.

Needs attn_implementation="eager" — the default SDPA/flash attention
paths compute attention without ever materializing the weight matrix,
so there's nothing for output_attentions=True to return under them.
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
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, attn_implementation="eager"
).to(device)
model.eval()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is 12 + 7?"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Regenerate greedily up to the decision token, identical to
# inspect_early_layers.py, so this is the same hidden state / context.
for i in range(TARGET_TOKEN + 1):
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=(i == TARGET_TOKEN),
        )
    last_logits = output.logits[0, -1, :]
    chosen_id = int(torch.argmax(last_logits).item())

    if i == TARGET_TOKEN:
        break

    input_ids = torch.cat([input_ids, torch.tensor([[chosen_id]], device=device)], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1
    )

# attentions[TARGET_LAYER]: [batch=1, heads=32, seq, seq]. We only care
# about the last row — how the current (about-to-generate) token
# attends back over everything seen so far.
layer_attn = output.attentions[TARGET_LAYER][0]  # [32, seq, seq]
last_token_attn = layer_attn[:, -1, :]  # [32, seq] — attention from the current token
seq_len = last_token_attn.shape[-1]

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
readable_tokens = [tokenizer.convert_tokens_to_string([t]).strip() or t for t in tokens]

# Most heads dump the bulk of their weight onto token 0 (<|im_start|>)
# regardless of content — a well-documented "attention sink" artifact,
# not signal. Excluding it is necessary to see which heads actually
# differentiate on content.
content_attn = last_token_attn[:, 1:]  # drop the sink token
max_weight_per_head, argmax_per_head = content_attn.max(dim=-1)
argmax_per_head = argmax_per_head + 1  # shift back to original indices
top_heads = torch.argsort(max_weight_per_head, descending=True)[:8]

sink_weight = last_token_attn[:, 0]
print(f"layer {TARGET_LAYER}, token {TARGET_TOKEN}, sequence length {seq_len}")
print(f"mean attention-sink weight (token 0) across heads: {sink_weight.mean().item():.4f}")
print(f"\ntop-8 most peaked heads on CONTENT (sink token excluded):")
for h in top_heads.tolist():
    tgt_idx = argmax_per_head[h].item()
    weight = max_weight_per_head[h].item()
    tgt_tok = readable_tokens[tgt_idx] if tgt_idx < len(readable_tokens) else "?"
    print(f"  head {h:2d}: attends to token {tgt_idx:3d} ({tgt_tok!r}) with weight {weight:.4f}")

def save_heatmap(token_range: range, filename: str, title_suffix: str):
    """
    Heatmap: all 32 heads x a window of context tokens. Color scale is
    capped at the content-only max (sink excluded) so the sink doesn't
    wash out everything else as a uniform bright column.
    """
    idx = list(token_range)
    sub = last_token_attn[:, idx].float().cpu().numpy()
    labels = [readable_tokens[i] if i < len(readable_tokens) else "?" for i in idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(sub, aspect="auto", cmap="viridis", vmax=content_attn.max().item())
    ax.set_xlabel(f"context token (indices {idx[0]}-{idx[-1]} of {seq_len})")
    ax.set_ylabel("attention head")
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_title(f"layer {TARGET_LAYER} attention from decision token {TARGET_TOKEN} — {title_suffix}")
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    out_path = f"tinkers/01_1_bare_react_loop/outputs/{filename}"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved: {out_path}")


# Two windows: the tail (recent reasoning/context immediately before
# the decision) and the system-prompt region around CALCULATE/ANSWER
# (tokens ~25-90), which the top-heads ranking pointed to directly but
# which a tail-only window would never show.
save_heatmap(range(max(0, seq_len - 40), seq_len), "layer33_attention_tail.png", "last 40 tokens")
save_heatmap(range(20, 95), "layer33_attention_instructions.png", "system-prompt CALCULATE/ANSWER region")
