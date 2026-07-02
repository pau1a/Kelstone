# Kelstone

Code from a series of hands-on investigations into what's actually
happening inside real, running language models — direct PyTorch access
to hidden states, attention weights, logits, and gradients, run
locally on consumer hardware.

## agentic_loop/

Code behind "Inside the Moment an AI Decides" — a minimal AI agent
(one tool, one instruction) examined at the exact moment it decides
whether to call its tool or answer directly, using the logit lens,
attention analysis, gradient attribution, and direct activation
patching (ablation) to distinguish correlation from causation inside
the model.

Requires `transformers`, `torch`, and `matplotlib`. Each script expects
a local Qwen3-4B checkpoint; set the `MODEL_PATH` environment variable
to point at your own copy, or place it at `./models/Qwen3-4B` relative
to wherever you run the script from.

- `bare_react_loop_v6.py` — the agent loop itself
- `inspect_early_layers.py` — logit-lens trace across all layers
- `inspect_layer33_attention.py` — attention weights at the decision layer
- `inspect_gradient_attribution.py` — gradient-based input attribution
- `inspect_layer33_mlp.py` — per-neuron correlational ranking
- `inspect_causal_patching.py` — direct ablation to test causation
- `inspect_all_neurons_causal.py` — systematic causal ranking across all neurons
