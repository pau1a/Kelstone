import json
import os
import re
import shutil

import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3-4B")
MAX_STEPS = 5
# First run at 300 tokens found the model genuinely stuck deliberating
# in prose ("Hmm... Wait... maybe...") on the ambiguous prompt, never
# reaching the GET-vs-ANSWER decision point before the budget ran out.
# Raised to give it real room to resolve, if it's going to.
MAX_NEW_TOKENS = 800
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Every prior version accumulated charts across runs, making it easy
# to mistake an old chart for a fresh one, or to leave stale output
# from a superseded prompt sitting alongside current results. v5
# starts every run from a guaranteed-empty outputs/ directory instead.
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
Manual structured tool calling, v5 — two more genuinely new visual
outputs, plus a guaranteed-clean outputs/ directory on every run.

v4 added the certainty trajectory (P(GET)/P(ANSWER) across every
token, not just the decision point) and found real, log-scale-visible
internal wavering correlated with the model's own hedge words. v5
adds two more instruments this tinker has been missing:

1. A DECISION-TREE / SANKEY diagram. Every prior chart looked at one
   stage of the decision in isolation (a bar chart for stage 1, a
   separate bar chart for stage 2). This tinker's decision is
   genuinely TWO-STAGE and branching — tool-vs-answer, then (only if
   "tool") which-tool — and until now nothing actually drew that as
   one connected structure. plot_decision_sankey shows probability
   mass flowing from the root through both real branches in a single
   diagram, the actual shape of the decision rather than two
   disconnected snapshots.

2. A CROSS-QUESTION GENERALIZATION SWEEP. Every result so far — in
   this tinker and in 1.1 — came from ONE prompt (occasionally two
   variants of near-identical prompts). 1.1's own write-up explicitly
   flagged this as an open question: is a found mechanism (a specific
   decision layer, a specific neuron) a stable, reusable circuit, or
   an artifact of one particular prompt? v5 runs several genuinely
   different questions (a clear weather question, a clear time
   question, a question needing neither tool, and the v3/v4 ambiguous
   probe) through the same instrumentation and plots whether the same
   decision layer and same top causal neuron keep showing up, or
   whether they're different every time.

Tools remain v2/v3's real, live APIs:
  - get_weather(city): live weather via open-meteo.com
  - get_time(timezone): live time via timeapi.io

The six per-decision snapshot instruments and the certainty-trajectory
chart are kept for every question in the sweep; the Sankey diagram and
the generalization-sweep summary chart are the new top-level outputs.
"""

SYSTEM_PROMPT = """You are a careful assistant with access to two tools:

get_weather — use this when asked about current weather conditions.
To call it, respond with exactly one line of JSON:
{"tool": "get_weather", "args": {"city": "<city name>"}}

get_time — use this when asked what time or date it currently is.
To call it, respond with exactly one line of JSON:
{"tool": "get_time", "args": {"timezone": "<IANA timezone, e.g. Asia/Tokyo>"}}

If neither tool is relevant to the question, do not call either one —
answer directly. Do not guess weather or time information yourself;
only report it after a tool call has returned a real result. Once you
are ready to give the final answer, respond with exactly this format:
ANSWER: <your final answer>
"""


def get_weather(city: str) -> str:
    """
    REAL tool: two live HTTP calls to open-meteo.com (no API key
    required). First geocodes the city name to coordinates, then
    fetches real current weather for those coordinates. Genuine I/O —
    the result is not knowable in advance and is not hardcoded
    anywhere in this file.
    """
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        results = geo.get("results")
        if not results:
            return f"no location found for '{city}'"
        lat, lon = results[0]["latitude"], results[0]["longitude"]

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true"},
            timeout=10,
        ).json()
        current = weather["current_weather"]
        return f"{current['temperature']}C, windspeed {current['windspeed']} km/h"
    except requests.RequestException as e:
        return f"error contacting weather service: {e}"


def get_time(timezone: str) -> str:
    """
    REAL tool: live HTTP call to timeapi.io (no API key required).
    Same standard as get_weather — real network round trip, real
    current data, genuinely unpredictable to the model in advance.
    """
    try:
        resp = requests.get(
            "https://timeapi.io/api/time/current/zone",
            params={"timeZone": timezone},
            timeout=10,
        ).json()
        return f"{resp['time']} on {resp['date']} ({resp['dayOfWeek']}), timezone {timezone}"
    except requests.RequestException as e:
        return f"error contacting time service: {e}"


def strip_thinking(reply: str) -> str:
    """
    Lesson carried forward from 1.1's confirmed bug: only search for a
    command after </think> ends, never in the raw reply, so a model
    quoting its own instructions while reasoning can't trick the parser.
    """
    return re.split(r"</think>", reply, maxsplit=1)[-1]


def extract_tool_call(text: str) -> dict | None:
    """Scan line by line for a JSON object with a "tool" key."""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    return None


device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(device)
model.eval()

# Verified against the real tokenizer output, not assumed — the same
# check that caught this file's own earlier three-way-decision design
# error. "get_weather" -> ['get', '_weather'], "get_time" -> ['get',
# '_time'], "ANSWER" -> ['ANS', 'WER']. GET_ID/ANSWER_ID are stage 1's
# candidates (both real single tokens, and DISTINCT from each other,
# unlike the original WEATHER_ID/TIME_ID mistake). WEATHER_2ND_ID/
# TIME_2ND_ID are stage 2's candidates, only ever reached after stage 1
# resolves toward "get".
GET_ID = tokenizer.encode("get", add_special_tokens=False)[0]
ANSWER_ID = tokenizer.encode("ANSWER", add_special_tokens=False)[0]
WEATHER_2ND_ID = tokenizer.encode("get_weather", add_special_tokens=False)[1]
TIME_2ND_ID = tokenizer.encode("get_time", add_special_tokens=False)[1]

print("stage 1 (tool-vs-answer) candidate tokens:")
print(f"  GET: id={GET_ID}  decodes to {tokenizer.decode([GET_ID])!r}")
print(f"  ANSWER: id={ANSWER_ID}  decodes to {tokenizer.decode([ANSWER_ID])!r}")
print("stage 2 (which tool) candidate tokens:")
print(f"  _weather: id={WEATHER_2ND_ID}  decodes to {tokenizer.decode([WEATHER_2ND_ID])!r}")
print(f"  _time: id={TIME_2ND_ID}  decodes to {tokenizer.decode([TIME_2ND_ID])!r}")


def plot_topk_distribution(probs: torch.Tensor, label_a_id: int, label_a: str, label_b_id: int, label_b: str, name: str):
    """Full next-token distribution at a decision point, top-15, not just the 2 tracked candidates."""
    top_probs, top_ids = torch.topk(probs, 15)
    labels = [repr(tokenizer.decode([i]))[1:-1] for i in top_ids.tolist()]
    colors = []
    for i in top_ids.tolist():
        if i == label_a_id:
            colors.append("tab:blue")
        elif i == label_b_id:
            colors.append("tab:orange")
        else:
            colors.append("tab:gray")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(15), top_probs.float().cpu().numpy(), color=colors)
    ax.set_xticks(range(15))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("probability")
    ax.set_title(f"{name} — top-15 next-token distribution ({label_a} vs {label_b})")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{name}_topk.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# Validated categorical palette (dataviz skill, references/palette.md),
# assigned in fixed slot order — slot 1 (blue) for the tool-call path,
# slot 6 (red) for the answer-directly path. Not cycled, not reused
# across unrelated series.
COLOR_GET = "#2a78d6"      # slot 1: blue
COLOR_ANSWER = "#e34948"   # slot 6: red
COLOR_ANNOTATION = "#52514e"  # text-secondary, for reasoning-moment markers
COLOR_ENTROPY = "#4a3aa7"  # slot 5: violet — distinct from GET/ANSWER/WEATHER/TIME


def report_high_entropy_moments(trajectory: list[dict], step: int, top_n: int = 5):
    """
    The entropy panel shows high-entropy REGIONS but never says what
    the model was actually choosing between at those moments — "2.4
    bits" is an abstract statistic, not a readable finding. This finds
    the top-N highest-entropy tokens in the whole trajectory and prints
    the real top-5 candidate words at each, so the entropy number
    becomes a concrete, checkable claim: e.g. "torn between 'mild',
    'pleasant', 'comfortable' — ordinary adjective choice, nothing to
    do with the tool-vs-answer decision at all" rather than just a
    number on a chart.
    """
    ranked = sorted(trajectory, key=lambda t: t["entropy_bits"], reverse=True)[:top_n]
    ranked = sorted(ranked, key=lambda t: t["token_idx"])  # back to chronological order for readability

    print(f"\n  top-{top_n} highest-entropy moments (step {step}) — what was the model actually torn between:")
    for t in ranked:
        candidates = ", ".join(f"{tok!r} ({p:.3f})" for tok, p in t["top5"])
        print(f"    token {t['token_idx']:3d}  entropy={t['entropy_bits']:.2f} bits  ->  {candidates}")

    return ranked


def plot_certainty_trajectory(
    trajectory: list[dict], annotations: list[tuple[int, str]], step: int
):
    """
    P(GET) and P(ANSWER), read off the model's own final-layer logits
    via a full forward pass, at EVERY generated token across the whole
    response — not a single snapshot at the moment of commitment, but
    the full shape of the model's certainty as it unfolds through an
    entire <think> block.

    NEW: a second panel plots full-VOCABULARY entropy (Shannon entropy
    over the entire ~150k-token softmax distribution, not just the two
    tracked candidates) on the SAME token axis. This asks a genuinely
    different question than the top panel: is the model uncertain in
    GENERAL at a given token (many plausible next words, of any kind),
    or only ever specifically torn between GET and ANSWER? These are
    not the same thing — a token could have high overall entropy (lots
    of plausible phrasings) while GET vs ANSWER specifically remains
    totally resolved, or vice versa. Nothing in this tinker (or in 1.1)
    has looked at whole-distribution uncertainty before; every prior
    instrument picked 2-3 candidate tokens in advance and only ever
    looked at those.

    trajectory: list of {"token_idx", "get_p", "answer_p",
    "entropy_bits", "token_str"} for every generated token.
    annotations: list of (token_idx, short_label) for genuinely
    interesting reasoning-text moments, shared across both panels via
    a vertical guide line.
    """
    token_idxs = [t["token_idx"] for t in trajectory]
    eps = 1e-12
    get_probs = [max(t["get_p"], eps) for t in trajectory]
    answer_probs = [max(t["answer_p"], eps) for t in trajectory]
    entropy = [t["entropy_bits"] for t in trajectory]

    kept = []
    for token_idx, label in sorted(annotations):
        if kept and token_idx - kept[-1][0] < 8:
            continue
        kept.append((token_idx, label))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1.3]}
    )

    # Top panel: the original two-candidate certainty trajectory. A
    # first attempt at this chart used a linear axis and the entire
    # deliberation region collapsed to a flat line at y=0 — the same
    # lesson 1.1 learned about its own logit-lens chart. Log scale
    # fixed it, kept here.
    ax_top.plot(token_idxs, get_probs, linewidth=2, color=COLOR_GET, label="P(GET)", zorder=3)
    ax_top.plot(token_idxs, answer_probs, linewidth=2, color=COLOR_ANSWER, label="P(ANSWER)", zorder=3)
    ax_top.set_yscale("log")
    ax_top.set_ylim(eps, 10)
    ax_top.set_ylabel("P(GET) / P(ANSWER)\n(log scale)")
    ax_top.set_title(
        f"step {step} — certainty trajectory (top) vs. full-vocabulary entropy (bottom)"
    )
    ax_top.legend(loc="upper left", frameon=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Bottom panel: whole-distribution entropy, linear scale (entropy
    # in bits is already a well-behaved, bounded quantity — no
    # small-value crushing problem the way raw probabilities have).
    ax_bot.fill_between(token_idxs, entropy, color=COLOR_ENTROPY, alpha=0.35, linewidth=0)
    ax_bot.plot(token_idxs, entropy, linewidth=1.5, color=COLOR_ENTROPY, zorder=3)
    ax_bot.set_ylabel("entropy (bits)\nover full vocabulary")
    ax_bot.set_xlabel("generated token position")
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    # Direct labels at the top-3 highest-entropy peaks, showing the
    # REAL candidate words the model was actually choosing between —
    # not just a bare number. This is what turns "entropy spiked here"
    # into a checkable, concrete finding rather than an abstract stat.
    top3_peaks = sorted(trajectory, key=lambda t: t["entropy_bits"], reverse=True)[:3]
    for t in top3_peaks:
        top3_words = "/".join(repr(tok)[1:-1] for tok, _ in t["top5"][:3])
        ax_bot.annotate(
            top3_words,
            xy=(t["token_idx"], t["entropy_bits"]),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=7.5,
            color=COLOR_ENTROPY,
            ha="center",
            va="bottom",
            rotation=0,
        )
        ax_bot.plot(t["token_idx"], t["entropy_bits"], marker="o", markersize=4, color=COLOR_ENTROPY, zorder=4)

    # Shared annotation guides across both panels, staggered so labels
    # on the top panel don't collide when markers land close together.
    tier_heights = [1.6, 2.6, 4.2]
    for n, (token_idx, label) in enumerate(kept):
        if token_idx >= len(trajectory):
            continue
        for ax in (ax_top, ax_bot):
            ax.axvline(token_idx, color=COLOR_ANNOTATION, linewidth=0.75, linestyle="--", alpha=0.35, zorder=1)
        tier_y = tier_heights[n % len(tier_heights)]
        ax_top.annotate(
            label, xy=(token_idx, tier_y), fontsize=8, color=COLOR_ANNOTATION,
            ha="center", va="bottom",
        )

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"v5_trajectory_step{step}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# Slot 2 (aqua) and slot 8 (orange) from the validated categorical
# palette, for the Sankey's two stage-2 branches — distinct from
# COLOR_GET/COLOR_ANSWER (slots 1/6) since all four appear on the same
# diagram at once and need to stay visually separable.
COLOR_WEATHER = "#1baf7a"  # slot 2: aqua
COLOR_TIME = "#eb6834"     # slot 8: orange
COLOR_NOT_REACHED = "#c3c2b7"  # neutral grey — text-secondary-on-light, for the "didn't happen" band


def plot_decision_sankey(
    p_get: float,
    p_answer: float,
    p_weather_given_get: float,
    p_time_given_get: float,
    stage2_reached: bool,
    name: str,
):
    """
    NEW INSTRUMENT: a hand-drawn Sankey/flow diagram of the FULL
    two-stage decision as one connected structure, not two separate
    bar charts. Every earlier chart in this tinker instruments stage 1
    (GET vs ANSWER) and stage 2 (_weather vs _time) as two independent
    snapshots — accurate, but it never shows the actual SHAPE of the
    decision: probability mass starting at a single root, splitting at
    stage 1, and (along the GET branch only) splitting again at stage
    2. This draws that directly: band widths proportional to
    probability mass, root on the left, leaves on the right.

    matplotlib has no built-in widget for this exact branching shape
    (its Sankey class is built for closed-loop flow diagrams, not a
    clean binary-tree branch), so this draws it by hand with filled
    polygons — each band a simple quadrilateral connecting a start
    height/width to an end height/width.

    stage2_reached: a real, confirmed bug found by actually looking at
    the rendered output — when stage 1 resolves to ANSWER, stage 2
    never happens at all, but the caller was previously substituting a
    fabricated 0.5/0.5 placeholder split so this function always had
    SOME numbers to draw. That produced a chart showing a confident
    "weather vs time, 50/50" split that never actually occurred, which
    is exactly the kind of thing this project is supposed to catch, not
    commit. When stage2_reached is False, the GET branch's stage-2
    columns are drawn as a single flat, hatched, neutral-grey band
    labelled "not reached" instead — no fabricated numbers, and the
    chart keeps the same three-column shape as when stage 2 DID fire,
    so the two charts stay visually comparable rather than one growing
    or shrinking a column depending on what happened.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    root_x, stage1_x, stage2_x = 0.0, 1.0, 2.0
    root_h = 1.0

    def band(x0, x1, y0_top, y0_bot, y1_top, y1_bot, color, label=None):
        xs = [x0, x1, x1, x0]
        ys = [y0_top, y1_top, y1_bot, y0_bot]
        ax.fill(xs, ys, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        if label:
            mid_x = (x0 + x1) / 2
            mid_y = ((y0_top + y0_bot) / 2 + (y1_top + y1_bot) / 2) / 2
            ax.annotate(label, xy=(mid_x, mid_y), fontsize=9, ha="center", va="center", color="#0b0b0b")

    # Root -> stage 1: GET band on top, ANSWER band below, heights
    # proportional to their real measured probabilities. A first
    # attempt at this diagram passed the FULL root height (0..root_h)
    # as both bands' starting span, so each band's polygon independently
    # claimed the entire root — visually crossing through the other
    # band instead of sitting adjacent to it. Fixed: the root itself is
    # a single point (root_h/2), and each band narrows FROM that point
    # OUT to its own proportional share at stage1_x — genuinely
    # adjacent, non-overlapping bands, the actual Sankey shape.
    root_mid = root_h / 2
    get_top, get_bot = root_h, root_h * (1 - p_get)
    answer_top, answer_bot = get_bot, 0.0
    band(root_x, stage1_x, root_mid, root_mid, get_top, get_bot, COLOR_GET, f"GET\nP={p_get:.3f}")
    band(root_x, stage1_x, root_mid, root_mid, answer_top, answer_bot, COLOR_ANSWER, f"ANSWER\nP={p_answer:.3f}")

    # Stage 1 -> stage 2: only the GET band splits further, scaled to
    # occupy the same vertical span the GET band already has at
    # stage1_x — the ANSWER band terminates, drawn as a closed leaf.
    if stage2_reached:
        weather_top = get_top
        weather_bot = get_top - (get_top - get_bot) * p_weather_given_get
        time_top = weather_bot
        time_bot = get_bot
        band(
            stage1_x, stage2_x, get_top, get_bot, weather_top, weather_bot,
            COLOR_WEATHER, f"_weather\nP={p_weather_given_get:.3f}|GET",
        )
        band(
            stage1_x, stage2_x, get_top, get_bot, time_top, time_bot,
            COLOR_TIME, f"_time\nP={p_time_given_get:.3f}|GET",
        )
    elif (get_top - get_bot) > 0.02:
        # Stage 2 genuinely never happened on this run — no real
        # weather/time split exists to draw. A single flat, hatched,
        # neutral-grey band stands in for it: same column position and
        # width as the real split would occupy, honestly labelled as
        # not having occurred, rather than either fabricating numbers
        # or silently dropping the column. Only drawn when the GET
        # band has enough real height to hold it — a real bug found by
        # actually looking at the rendered chart: when P(GET) is itself
        # ~0.000 (this response never seriously considered a tool call
        # at all), get_top - get_bot collapses to ~0, so the "not
        # reached" fill and its label were being drawn into a
        # zero-height sliver — invisible, with the label floating,
        # unanchored, over whatever color happened to be underneath
        # (the ANSWER leaf, which legitimately fills most of the chart
        # when P(ANSWER) ~ 1.0). Below this threshold there's nothing
        # meaningful to label at stage 2 either way, so it's skipped
        # entirely rather than drawn illegibly.
        xs = [stage1_x, stage2_x, stage2_x, stage1_x]
        ys = [get_top, get_top, get_bot, get_bot]
        ax.fill(
            xs, ys, color=COLOR_NOT_REACHED, alpha=0.5, edgecolor="white",
            linewidth=0.5, hatch="////",
        )
        ax.annotate(
            "stage 2 not reached\n(this response ended at stage 1)",
            xy=((stage1_x + stage2_x) / 2, (get_top + get_bot) / 2),
            fontsize=9, ha="center", va="center", color="#3a3a38",
        )
    # ANSWER terminates at stage1_x — drawn as a flat closed leaf
    # rather than continuing, since stage 2 never happens on this path.
    band(
        stage1_x, stage2_x, answer_top, answer_bot, answer_top, answer_bot,
        COLOR_ANSWER,
    )

    ax.text(root_x, root_h + 0.04, "start", ha="center", fontsize=9, color="#52514e")
    ax.text(stage1_x, root_h + 0.04, "stage 1", ha="center", fontsize=9, color="#52514e")
    ax.text(stage2_x, root_h + 0.04, "stage 2\n(if GET)", ha="center", fontsize=9, color="#52514e")

    ax.set_xlim(-0.15, 2.4)
    ax.set_ylim(-0.05, 1.15)
    ax.axis("off")
    ax.set_title(f"{name} — full two-stage decision as one connected flow")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{name}_sankey.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_logit_lens(hidden_states: tuple, id_a: int, label_a: str, id_b: int, label_b: str, name: str):
    """
    Logit lens across all layers: each layer's hidden state, passed
    through the model's own final norm + lm_head, as if generation
    stopped there. Both linear and log-scale versions saved — 1.1
    found the linear scale hides real early-layer structure
    (background noise vs. a genuine contest).
    """
    probs_a, probs_b = [], []
    with torch.no_grad():
        for layer_hidden in hidden_states:
            last_token_hidden = layer_hidden[0, -1, :]
            normed = model.model.norm(last_token_hidden)
            layer_logits = model.lm_head(normed)
            layer_probs = torch.softmax(layer_logits, dim=-1)
            probs_a.append(layer_probs[id_a].item())
            probs_b.append(layer_probs[id_b].item())

    layers = range(len(hidden_states))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(layers, probs_a, marker="o", color="tab:blue", label=f"P({label_a})")
    ax.plot(layers, probs_b, marker="o", color="tab:orange", label=f"P({label_b})")
    ax.set_xlabel("layer (0 = embeddings, last = final hidden state)")
    ax.set_ylabel("probability (via logit lens)")
    ax.set_title(f"{name} — {label_a} vs {label_b} through the layers")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{name}_logitlens.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    eps = 1e-12
    ax.plot(layers, [max(p, eps) for p in probs_a], marker="o", color="tab:blue", label=f"P({label_a})")
    ax.plot(layers, [max(p, eps) for p in probs_b], marker="o", color="tab:orange", label=f"P({label_b})")
    ax.set_yscale("log")
    ax.set_xlabel("layer (0 = embeddings, last = final hidden state)")
    ax.set_ylabel("probability (log scale, via logit lens)")
    ax.set_title(f"{name} — {label_a} vs {label_b} through the layers (log scale)")
    ax.legend()
    fig.tight_layout()
    log_path = os.path.join(OUTPUT_DIR, f"{name}_logitlens_logscale.png")
    fig.savefig(log_path, dpi=120)
    plt.close(fig)

    return path, log_path, probs_a, probs_b


def find_decision_layer(probs_a: list, probs_b: list) -> int:
    """
    Find the layer with the single largest jump in the winning
    candidate's probability from the previous layer — an empirical
    detection of where THIS decision resolves, rather than assuming
    1.1's layer 33 applies unchanged to a different decision.

    probs_a/probs_b come from hidden_states, which has one entry for
    the embedding layer PLUS one per transformer layer (37 entries for
    a 36-layer model) — index 0 is embeddings, not a real transformer
    layer. output.attentions, used downstream for the attention
    heatmap, only has one entry per actual transformer layer (36
    entries, valid indices 0-35), with NO embedding-layer entry. A
    real bug caught by actually running this: the raw jump-index
    arithmetic below can return up to len(hidden_states) - 1 = 36,
    which is one past the end of output.attentions and throws
    IndexError. Clamping to the last valid attention index (35) fixes
    it — the jump into the final hidden state (from layer 35 into the
    unembedding) doesn't correspond to any attention layer to inspect
    in the first place, so clamping there is also the semantically
    correct choice, not just a crash-avoidance hack.
    """
    winner_probs = [max(a, b) for a, b in zip(probs_a, probs_b)]
    jumps = [winner_probs[i] - winner_probs[i - 1] for i in range(1, len(winner_probs))]
    raw_layer = int(torch.tensor(jumps).argmax().item()) + 1  # +1: jumps[i] is the jump INTO layer i+1
    max_valid_attention_layer = len(probs_a) - 2  # hidden_states has 1 more entry than attentions
    return min(raw_layer, max_valid_attention_layer)


def plot_attention_heatmap(model_eager, inputs_for_attn, decision_layer: int, name: str):
    """
    Re-runs the forward pass with attn_implementation="eager" (the
    SDPA/flash path used for generation never materializes attention
    weights — same requirement 1.1 documented). Captures the decision
    layer's raw attention weights from the decision token back over
    the whole prompt.

    A real, confirmed bug in every earlier version of this chart: it
    excluded the attention-sink token (index 0) from the VMAX
    calculation, but still plotted a 90-token window that includes the
    sink itself, on a LINEAR colour scale. The sink still dominated the
    scale (attention weight ~0.6-0.7 vs. real structure in the
    ~0.01-0.05 range), so on a linear map everything except the sink
    still renders as one flat near-black colour — nearly the whole
    chart looking like "one colour," a real and specific complaint,
    not a matter of taste. This is the exact same lesson 1.1 already
    learned twice (its own logit-lens chart, this project's
    certainty-trajectory chart): a linear scale hides small-value
    structure. Fixed here with a genuine log-scale colour map
    (LogNorm) instead of a linear one, which is the correct fix, not
    just cropping the sink out of the displayed window.
    """
    with torch.no_grad():
        output = model_eager(
            input_ids=inputs_for_attn["input_ids"],
            attention_mask=inputs_for_attn["attention_mask"],
            output_attentions=True,
        )
    layer_attn = output.attentions[decision_layer][0]  # [heads, seq, seq]
    last_token_attn = layer_attn[:, -1, :]  # attention FROM the decision token
    seq_len = last_token_attn.shape[-1]

    input_ids = inputs_for_attn["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    readable_tokens = [tokenizer.convert_tokens_to_string([t]).strip() or t for t in tokens]

    window = min(90, seq_len)
    sub = last_token_attn[:, :window].float().cpu().numpy()
    labels = readable_tokens[:window]

    eps = 1e-4
    sub_for_log = sub.clip(min=eps)
    vmin, vmax = eps, sub_for_log.max()

    fig, ax = plt.subplots(figsize=(15, 8))
    im = ax.imshow(
        sub_for_log,
        aspect="auto",
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_xlabel(f"context token (first {window} of {seq_len})")
    ax.set_ylabel("attention head")
    ax.set_xticks(range(window))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_title(f"{name} — layer {decision_layer} attention from decision token (log scale)")
    fig.colorbar(im, ax=ax, label="attention weight (log scale)")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{name}_attention_heatmap.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_gradient_attribution(inputs_for_grad, winning_id: int, runner_up_id: int, name: str):
    """
    Backprop from (winning logit - runner-up logit) to the input
    embeddings. Independent method from attention — if it flags the
    same tokens, that's real corroboration (1.1's finding 5 method).
    """
    embed_layer = model.model.embed_tokens
    inputs_embeds = embed_layer(inputs_for_grad["input_ids"]).detach().clone().requires_grad_(True)

    output = model(inputs_embeds=inputs_embeds, attention_mask=inputs_for_grad["attention_mask"])
    last_logits = output.logits[0, -1, :].float()
    decision_score = last_logits[winning_id] - last_logits[runner_up_id]

    model.zero_grad(set_to_none=True)
    decision_score.backward()

    grad = inputs_embeds.grad[0]
    embeds = inputs_embeds[0].detach()
    saliency = (grad * embeds).sum(dim=-1).float()

    seq_len = saliency.shape[0]
    input_ids = inputs_for_grad["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    readable_tokens = [tokenizer.convert_tokens_to_string([t]).strip() or t for t in tokens]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    colors = ["tab:orange" if s > 0 else "tab:blue" for s in saliency.tolist()]
    ax.bar(range(seq_len), saliency.detach().cpu().numpy(), color=colors, width=1.0)
    ax.set_xlabel("token position")
    ax.set_ylabel("saliency (gradient x input)")
    ax.set_title(f"{name} — gradient attribution")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{name}_gradient_attribution.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    top_pos = torch.argsort(saliency.abs(), descending=True)[:10]
    print(f"  top-10 tokens by gradient saliency ({name}):")
    for p in top_pos.tolist():
        tok = readable_tokens[p] if p < len(readable_tokens) else "?"
        print(f"    token {p:3d} ({tok!r}): saliency={saliency[p].item():+.4f}")

    return path


def analyse_mlp_and_causal_patch(inputs_for_analysis, decision_layer: int, winning_id: int, runner_up_id: int, name: str):
    """
    Two-part instrument, same order as 1.1: rank neurons
    correlationally (activation x fixed output direction), then verify
    the top-ranked neuron's REAL causal effect by physically silencing
    it and re-measuring. 1.1's biggest finding was that these two can
    disagree (finding 6/7) — repeated here, not assumed to hold.
    """
    input_ids = inputs_for_analysis["input_ids"]
    attention_mask = inputs_for_analysis["attention_mask"]

    captured = {}

    def capture_intermediate(module, inputs, output):
        captured["intermediate"] = inputs[0].detach()

    hook_handle = model.model.layers[decision_layer].mlp.down_proj.register_forward_hook(capture_intermediate)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    hook_handle.remove()

    baseline_logits = output.logits[0, -1, :].float()
    baseline_gap = (baseline_logits[winning_id] - baseline_logits[runner_up_id]).item()

    intermediate = captured["intermediate"][0, -1, :].float()
    down_proj_weight = model.model.layers[decision_layer].mlp.down_proj.weight.float()

    with torch.no_grad():
        unit_directions = down_proj_weight.T
        normed_directions = model.model.norm(unit_directions.to(device).to(torch.bfloat16))
        per_unit_logits = model.lm_head(normed_directions).float()

    per_unit_winner = per_unit_logits[:, winning_id]
    per_unit_runnerup = per_unit_logits[:, runner_up_id]
    contribution_diff = intermediate * (per_unit_winner - per_unit_runnerup)

    top_neuron = int(torch.argmax(contribution_diff.abs()).item())
    top_contribution = contribution_diff[top_neuron].item()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sorted_diff, _ = torch.sort(contribution_diff, descending=True)
    ax.plot(range(len(sorted_diff)), sorted_diff.cpu().numpy())
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("neuron rank (sorted by contribution)")
    ax.set_ylabel("logit(winner) - logit(runner-up) contribution")
    ax.set_title(f"{name} — layer {decision_layer} MLP per-neuron push, sorted")
    fig.tight_layout()
    mlp_path = os.path.join(OUTPUT_DIR, f"{name}_mlp_neurons.png")
    fig.savefig(mlp_path, dpi=120)
    plt.close(fig)

    # Causal patching: silence the single top-ranked neuron for real
    # and re-measure. The check that overturned 1.1's own correlational
    # ranking — a correlational "most important" neuron is a
    # hypothesis, not a conclusion, until tested this way.
    def zero_neuron_hook(module, inputs, output):
        contribution = inputs[0][0, -1, top_neuron] * down_proj_weight[:, top_neuron].to(output.dtype)
        patched = output.clone()
        patched[0, -1, :] -= contribution.to(patched.dtype)
        return patched

    handle = model.model.layers[decision_layer].mlp.down_proj.register_forward_hook(zero_neuron_hook)
    with torch.no_grad():
        patched_output = model(input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()

    patched_logits = patched_output.logits[0, -1, :].float()
    patched_gap = (patched_logits[winning_id] - patched_logits[runner_up_id]).item()
    real_delta = patched_gap - baseline_gap

    print(
        f"  {name} MLP causal check: top correlational neuron={top_neuron}, "
        f"isolated-direction contribution={top_contribution:+.4f}, "
        f"REAL Δgap from silencing it={real_delta:+.4f} "
        f"({'meaningful' if abs(real_delta) > 0.5 else 'small/negligible'})"
    )

    return mlp_path, top_neuron, top_contribution, real_delta


def instrument_decision(snapshot: dict, id_a: int, label_a: str, id_b: int, label_b: str, name: str):
    """
    Runs the full six-part instrumentation stack (topk distribution,
    logit lens x2, empirical decision-layer detection, attention
    heatmap, gradient attribution, MLP ranking + causal patch
    verification) for one real, already-confirmed decision point
    between exactly two candidate tokens. Shared by both stage 1
    (GET vs ANSWER) and stage 2 (_weather vs _time) — the instrument
    doesn't care which stage it's given, only that the two candidate
    IDs are real and distinct.
    """
    with torch.no_grad():
        output = model(
            input_ids=snapshot["input_ids"],
            attention_mask=snapshot["attention_mask"],
            output_hidden_states=True,
        )
    probs = torch.softmax(output.logits[0, -1, :], dim=-1)

    topk_path = plot_topk_distribution(probs, id_a, label_a, id_b, label_b, name)
    print(f"  [chart] {topk_path}")

    lens_path, lens_log_path, probs_a, probs_b = plot_logit_lens(
        output.hidden_states, id_a, label_a, id_b, label_b, name
    )
    print(f"  [chart] {lens_path}")
    print(f"  [chart] {lens_log_path}")

    decision_layer = find_decision_layer(probs_a, probs_b)
    print(f"  [empirical decision layer: {decision_layer}]")

    model_eager = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model_eager.eval()
    attn_path = plot_attention_heatmap(model_eager, snapshot, decision_layer, name)
    print(f"  [chart] {attn_path}")
    del model_eager

    winning_id, runner_up_id = (id_a, id_b) if probs[id_a] >= probs[id_b] else (id_b, id_a)

    grad_path = plot_gradient_attribution(snapshot, winning_id, runner_up_id, name)
    print(f"  [chart] {grad_path}")

    mlp_path, top_neuron, top_contrib, real_delta = analyse_mlp_and_causal_patch(
        snapshot, decision_layer, winning_id, runner_up_id, name
    )
    print(f"  [chart] {mlp_path}")


# Words/phrases in the model's own generated text that plausibly mark
# a real moment of reasoning-level deliberation — used only to choose
# WHICH tokens get a direct annotation on the trajectory chart, not to
# change any measurement. Matched case-insensitively against each
# newly generated token's decoded text.
DELIBERATION_MARKERS = ["hmm", "wait", "maybe", "alternatively", "however"]


def ask_model_manual(messages: list[dict], step: int) -> str:
    """
    Manual token-by-token generation. Watches for BOTH real decision
    points as they occur: stage 1 (GET vs ANSWER argmax) and, only if
    stage 1 resolves toward GET, stage 2 (_weather vs _time argmax) at
    the very next relevant token. Each is instrumented the moment it's
    confirmed to have actually happened — the argmax-trigger fix v1
    found the hard way, applied to both stages from the start here.

    NEW in v4: also records P(GET) and P(ANSWER) from the model's own
    final logits at EVERY token (not just the decision point) into
    `trajectory`, and flags tokens matching DELIBERATION_MARKERS for
    direct annotation — feeding plot_certainty_trajectory once
    generation completes.
    """
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    eos_id = tokenizer.eos_token_id

    generated_ids = []
    stage1_done = False
    stage1_went_to_tool = False
    stage2_done = False
    trajectory = []
    annotations = []
    stage1_probs = None
    stage2_probs = None

    print("model: ", end="", flush=True)
    for i in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        last_logits = output.logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        chosen_id = int(torch.argmax(probs).item())

        # Full-vocabulary Shannon entropy at this step: -sum(p * log2(p))
        # over the ENTIRE ~150k-token vocabulary distribution, not just
        # the two candidates (GET/ANSWER) already tracked. This answers
        # a genuinely different question than the certainty trajectory
        # does: is the model uncertain in GENERAL at this token (many
        # plausible continuations), or only ever contested between
        # these two specific pre-chosen candidates? Max possible entropy
        # for this vocabulary is log2(vocab_size) ~ 17.2 bits (near-total
        # uniform confusion); a confident single-token commitment is
        # close to 0 bits.
        safe_probs = probs.clamp_min(1e-12)
        entropy_bits = -(safe_probs * safe_probs.log2()).sum().item()

        # Top-5 candidates at this exact step, captured alongside the
        # entropy number itself. High entropy alone is an abstract
        # statistic — "2.4 bits" means nothing to a reader. Capturing
        # what the model was ACTUALLY choosing between at its highest-
        # entropy moments turns the number into a real, readable
        # answer: not "torn about tools", but e.g. torn between
        # "mild"/"pleasant"/"comfortable" — ordinary word choice, not
        # anything to do with the tool-vs-answer decision at all.
        top5_probs, top5_ids = torch.topk(probs, 5)
        top5 = [
            (tokenizer.decode([tid]), p.item())
            for tid, p in zip(top5_ids.tolist(), top5_probs)
        ]

        token_str = tokenizer.decode([chosen_id])
        trajectory.append(
            {
                "token_idx": i,
                "get_p": probs[GET_ID].item(),
                "answer_p": probs[ANSWER_ID].item(),
                "entropy_bits": entropy_bits,
                "top5": top5,
                "token_str": token_str,
            }
        )
        lowered = token_str.strip().lower()
        if any(marker in lowered for marker in DELIBERATION_MARKERS) and len(lowered) > 2:
            annotations.append((i, token_str.strip()))

        if not stage1_done and chosen_id in (GET_ID, ANSWER_ID):
            get_p = probs[GET_ID].item()
            answer_p = probs[ANSWER_ID].item()
            stage1_probs = (get_p, answer_p)
            print(
                f"\n  [stage 1 decision @ token {i}] "
                f"P(GET)={get_p:.4f}  P(ANSWER)={answer_p:.4f}",
                flush=True,
            )
            snapshot = {"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone()}
            instrument_decision(snapshot, GET_ID, "GET", ANSWER_ID, "ANSWER", f"v5_step{step}_stage1")
            stage1_done = True
            stage1_went_to_tool = chosen_id == GET_ID

        elif stage1_done and stage1_went_to_tool and not stage2_done and chosen_id in (WEATHER_2ND_ID, TIME_2ND_ID):
            weather_p = probs[WEATHER_2ND_ID].item()
            time_p = probs[TIME_2ND_ID].item()
            stage2_probs = (weather_p, time_p)
            print(
                f"\n  [stage 2 decision @ token {i}] "
                f"P(_weather)={weather_p:.4f}  P(_time)={time_p:.4f}",
                flush=True,
            )
            snapshot = {"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone()}
            instrument_decision(
                snapshot, WEATHER_2ND_ID, "_weather", TIME_2ND_ID, "_time", f"v5_step{step}_stage2"
            )
            stage2_done = True

        generated_ids.append(chosen_id)
        print(tokenizer.decode([chosen_id]), end="", flush=True)

        input_ids = torch.cat([input_ids, torch.tensor([[chosen_id]], device=device)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )

        if chosen_id == eos_id:
            break
    print()

    trajectory_path = plot_certainty_trajectory(trajectory, annotations, step)
    print(f"  [chart] {trajectory_path}  ({len(annotations)} deliberation moments annotated)")

    report_high_entropy_moments(trajectory, step)

    if stage1_probs is not None:
        get_p, answer_p = stage1_probs
        # Real, confirmed bug (found by looking at the actual rendered
        # chart, not just the code): this used to substitute a
        # fabricated 0.5/0.5 placeholder whenever stage 2 never
        # happened, which drew a confident-looking "50/50 weather vs
        # time" split on the chart that never actually occurred, with
        # overlapping labels as a second, purely visual symptom of the
        # same underlying problem. Fixed: pass through whether stage 2
        # was genuinely reached, and let plot_decision_sankey render an
        # honest "not reached" band instead of inventing numbers.
        stage2_reached = stage2_probs is not None
        weather_p, time_p = stage2_probs if stage2_reached else (0.0, 0.0)
        sankey_path = plot_decision_sankey(
            get_p, answer_p, weather_p, time_p, stage2_reached, f"v5_step{step}"
        )
        print(f"  [chart] {sankey_path}")

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's it like in Tokyo right now?"},
]

for step in range(MAX_STEPS):
    print(f"--- step {step + 1} ---")
    reply = ask_model_manual(messages, step + 1)
    command_text = strip_thinking(reply)

    tool_call = extract_tool_call(command_text)
    answer_match = re.search(r"ANSWER:\s*(.+)", command_text)

    messages.append({"role": "assistant", "content": reply})

    if answer_match:
        print(f"\nfinal answer: {answer_match.group(1).strip()}")
        break
    elif tool_call and tool_call.get("tool") == "get_weather":
        city = tool_call.get("args", {}).get("city", "")
        result = get_weather(city)
        print(f"tool call: get_weather(city={city!r}) -> {result}")
        messages.append({"role": "user", "content": f"TOOL RESULT: {result}"})
    elif tool_call and tool_call.get("tool") == "get_time":
        tz = tool_call.get("args", {}).get("timezone", "")
        result = get_time(tz)
        print(f"tool call: get_time(timezone={tz!r}) -> {result}")
        messages.append({"role": "user", "content": f"TOOL RESULT: {result}"})
    else:
        print("(model produced neither a valid tool call nor a final answer — stopping)")
        break
else:
    print(f"\n(hit MAX_STEPS={MAX_STEPS} without a final answer)")
