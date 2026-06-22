# nn-zero-to-hero

Follow-along work for Andrej Karpathy's **Neural Networks: Zero to Hero**, plus a couple
of Hugging Face / PEFT side projects. Single shared `uv` environment at the repo root
(`pyproject.toml` / `uv.lock`), PyTorch CUDA (cu128) on an RTX 5080.

## Projects

| Dir | What |
|---|---|
| [`gpt/`](gpt/) | char-level **decoder-only Transformer** built from scratch (self-attention, multi-head, causal mask, residual + LayerNorm) trained on tiny Shakespeare. |
| [`peft-lora/`](peft-lora/) | **LoRA** fine-tuning of DistilBERT on SST-2 — 1.09% trainable params, ~0.888 acc vs 0.904 full FT. |
| `hf-intro/` | Hugging Face `transformers`/`datasets` intro. |
| `hf-finetune/` | Full fine-tuning walkthrough (Trainer API). |

See each subdirectory's README for details, results, and how to run it.
