# Two real tools, and general vs. specific uncertainty

Code behind the write-up on giving a model two real tools (a live
weather API and a live time API) and a genuine three-way choice
between them — plus a deeper instrument that separates a model's
*general* linguistic uncertainty from its *specific* uncertainty about
which tool to call.

Both tools are real, live HTTP calls (no API key required):
- `get_weather(city)` via open-meteo.com
- `get_time(timezone)` via timeapi.io

The decision is genuinely two-stage, not three-way at a single token
(the model's own tokenizer splits `get_weather`/`get_time` on a shared
first token, `get`) — stage 1 is tool-vs-answer, stage 2 (only reached
if stage 1 picks a tool) is which-tool.

Requires `transformers`, `torch`, `matplotlib`, and `requests`. Set
`MODEL_PATH` to point at your own Qwen3-4B checkpoint, or place it at
`./models/Qwen3-4B` relative to wherever you run the script.

- `manual_tool_call_v5.py` — the full investigation: two real tools,
  the two-stage decision, six snapshot instruments (next-token
  distribution, logit lens, attention heatmap, gradient attribution,
  MLP neuron ranking, causal activation patching), a full-response
  certainty trajectory, a full-vocabulary entropy trajectory with its
  peaks decoded into real candidate words, and a two-stage decision
  Sankey diagram.

`outputs/` contains every chart the script produces, including the
ones referenced in the article.
