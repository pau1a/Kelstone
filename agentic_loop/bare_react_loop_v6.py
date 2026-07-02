import os
import re

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3-4B")
MAX_STEPS = 5
MAX_NEW_TOKENS = 400
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

"""
Hand-rolled ReAct-style loop: no agent framework, no tool-calling API.

v5 fixed the parsing bug and confirmed the loop works, but every
"internals" signal so far has been two scalar numbers (P(CALCULATE),
P(ANSWER)) read off the final logits. That's a fraction of what PyTorch
access actually offers. v6 digs deeper at each decision point and saves
two charts:

1. A top-N bar chart of the full next-token probability distribution
   (not just two cherry-picked tokens) — shows the whole shape of the
   model's uncertainty, not a pre-selected pair.
2. A logit-lens view: the hidden state at every one of the 36 layers,
   independently passed through the model's own final norm + lm_head,
   plotted as P(CALCULATE) vs P(ANSWER) per layer — shows how (or
   whether) the tool-vs-answer decision builds up progressively through
   the network, rather than only existing at the final layer.
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

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_calculator(expression: str) -> str:
    """
    The 'action' half of the loop. Deliberately just eval() on a tightly
    restricted expression — no framework, direct contact with what a
    tool call actually is: code that runs and returns a result.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"error: {e}"


def strip_thinking(reply: str) -> str:
    """
    Drop everything up to and including a </think> block. The command
    the model actually commits to only ever appears after its reasoning
    trace — searching the raw reply risks matching instruction text the
    model echoes/paraphrases while thinking, not a real command.
    """
    return re.split(r"</think>", reply, maxsplit=1)[-1]


def plot_topk_distribution(probs: torch.Tensor, step: int, token_idx: int, k: int = 15):
    """
    Full next-token distribution at the decision point, top-k by
    probability — not just the two tokens we care about. Shows what
    else the model was weighing.
    """
    top_probs, top_ids = torch.topk(probs, k)
    labels = [repr(tokenizer.decode([i]))[1:-1] for i in top_ids.tolist()]
    colors = [
        "tab:orange" if i == CALCULATE_ID else "tab:blue" if i == ANSWER_ID else "tab:gray"
        for i in top_ids.tolist()
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(k), top_probs.float().cpu().numpy(), color=colors)
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("probability")
    ax.set_title(f"step {step} — top-{k} next-token distribution @ token {token_idx}")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"step{step}_token{token_idx}_topk.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_logit_lens(hidden_states: tuple, step: int, token_idx: int):
    """
    Logit lens: pass the hidden state from every layer (not just the
    final one) through the model's own final norm + lm_head, as if
    generation stopped there. Shows P(CALCULATE) vs P(ANSWER) building
    up (or not) layer by layer — real use of internals access, since a
    black-box .generate() call only ever exposes the final layer.
    """
    calc_probs, answer_probs = [], []
    with torch.no_grad():
        for layer_hidden in hidden_states:
            last_token_hidden = layer_hidden[0, -1, :]
            normed = model.model.norm(last_token_hidden)
            layer_logits = model.lm_head(normed)
            layer_probs = torch.softmax(layer_logits, dim=-1)
            calc_probs.append(layer_probs[CALCULATE_ID].item())
            answer_probs.append(layer_probs[ANSWER_ID].item())

    layers = range(len(hidden_states))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(layers, calc_probs, marker="o", color="tab:orange", label="P(CALCULATE)")
    ax.plot(layers, answer_probs, marker="o", color="tab:blue", label="P(ANSWER)")
    ax.set_xlabel("layer (0 = embeddings, last = final hidden state)")
    ax.set_ylabel("probability (via logit lens)")
    ax.set_title(f"step {step} — CALCULATE vs ANSWER through the layers @ token {token_idx}")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"step{step}_token{token_idx}_logitlens.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def ask_model(messages: list[dict], step: int) -> str:
    """
    Manual token-by-token generation (greedy). At the decision point
    (first token where CALCULATE or ANSWER becomes a live candidate),
    saves both charts before continuing generation. Prints each token
    as it's generated so a long run never looks dead.
    """
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    eos_id = tokenizer.eos_token_id

    generated_ids = []
    logged_decision = False

    print("model: ", end="", flush=True)
    for i in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=not logged_decision,
            )

        last_logits = output.logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        chosen_id = int(torch.argmax(probs).item())

        if not logged_decision:
            calc_p = probs[CALCULATE_ID].item()
            answer_p = probs[ANSWER_ID].item()
            if calc_p > 1e-6 or answer_p > 1e-6:
                print(
                    f"\n  [decision point @ token {i}] "
                    f"P(CALCULATE)={calc_p:.4f}  P(ANSWER)={answer_p:.4f}",
                    flush=True,
                )
                topk_path = plot_topk_distribution(probs, step, i)
                lens_path = plot_logit_lens(output.hidden_states, step, i)
                print(f"  [charts] {topk_path}")
                print(f"  [charts] {lens_path}")
                logged_decision = True

        generated_ids.append(chosen_id)
        print(tokenizer.decode([chosen_id]), end="", flush=True)

        input_ids = torch.cat(
            [input_ids, torch.tensor([[chosen_id]], device=device)], dim=1
        )
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )

        if chosen_id == eos_id:
            break
    print()

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is 12 + 7?"},
]

for step in range(MAX_STEPS):
    print(f"--- step {step + 1} ---")
    reply = ask_model(messages, step + 1)
    command_text = strip_thinking(reply)

    """
    Each iteration is: ask the model, look at what it said, decide
    whether that was an action request or a final answer, act
    accordingly. This is the entire 'agentic loop' — everything else
    (frameworks, tool schemas, structured calling) is convenience
    layered on top of this same reason -> act -> observe -> repeat shape.
    """

    calc_match = re.search(r"CALCULATE:\s*(.+)", command_text)
    answer_match = re.search(r"ANSWER:\s*(.+)", command_text)

    messages.append({"role": "assistant", "content": reply})

    if answer_match:
        print(f"\nfinal answer: {answer_match.group(1).strip()}")
        break
    elif calc_match:
        expr = calc_match.group(1).strip()
        result = run_calculator(expr)
        print(f"tool result: {expr} = {result}")
        messages.append({"role": "user", "content": f"TOOL RESULT: {result}"})
    else:
        print("(model produced neither a tool call nor a final answer — stopping)")
        break
else:
    print(f"\n(hit MAX_STEPS={MAX_STEPS} without a final answer)")
