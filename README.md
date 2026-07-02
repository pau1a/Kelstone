# Kelstone

Code from a series of hands-on investigations into what's actually
happening inside real, running language models — direct PyTorch access
to hidden states, attention weights, logits, and gradients, run
locally on consumer hardware.

Organized into 10 broad categories, each containing individual
investigations as they're published:

1. [Agentic loop](01_agentic_loop/) — reason, act, observe, repeat
2. Tool use — structured tools and how a model chooses between them
3. Constrained generation — forcing valid output at the token level
4. Structured output — reliable schema-shaped output
5. Retrieval-augmented generation — giving a model outside knowledge
6. Multi-turn conversation — what persists across a conversation
7. Fine-tuning with adapters — LoRA, without retraining a full model
8. Embeddings — turning meaning into numbers
9. Model confidence — reading a model's own certainty
10. Sampling and randomness — temperature, top-k, top-p

Each investigation lives in its own subfolder with the code behind a
published write-up. Requires `transformers`, `torch`, and
`matplotlib`; each script expects a local model checkpoint and reads
its path from the `MODEL_PATH` environment variable (see each
investigation's own scripts for the specific model used).
