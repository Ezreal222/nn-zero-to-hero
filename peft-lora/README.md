# LoRA demo — DistilBERT + SST-2

Parameter-efficient fine-tuning (PEFT) of **DistilBERT** on the **SST-2** sentiment
task using **LoRA**. The headline: training only **1.09%** of the weights gets within
~2 points of full fine-tuning.

| Method | Trainable params | % of total | SST-2 val acc |
|---|---|---|---|
| Full fine-tune (baseline) | ~67.0 M | 100% | **0.904** |
| **LoRA (r=8)** | **0.74 M** | **1.09%** | **0.888** |

> Trainable-param count is exact from the run log:
> `trainable params: 739,586 || all params: 67,694,596 || trainable%: 1.0925`.
> Accuracy comes from the W&B run below; the saved `lora-out/` checkpoint logs
> `eval_accuracy = 0.884` at epoch 1.

## What LoRA does

Instead of updating a weight matrix `W` directly, LoRA freezes `W` and learns a
low-rank update `ΔW = B·A`, where `A` is `(r × d)` and `B` is `(d × r)` with `r` small
(here `r=8`). Only `A` and `B` train, so:

```
W_eff = W (frozen) + (alpha / r) · B·A   # A, B are the only trainable tensors
```

With `r=8` over DistilBERT's attention `q_lin`/`v_lin` projections, that's 0.74M params
vs 67M — a ~90x reduction in what the optimizer has to track, while the frozen backbone
stays shared and reusable across tasks.

## Setup

```python
from peft import LoraConfig, get_peft_model, TaskType

lora = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],   # DistilBERT attention projections
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
# -> trainable params: 739,586 || all params: 67,694,596 || trainable%: 1.0925
```

- **Base model:** `distilbert-base-uncased` (`AutoModelForSequenceClassification`, 2 labels)
- **Dataset:** `nyu-mll/glue`, config `sst2`
- **Training:** 2 epochs, batch size 32, lr `2e-4` (LoRA tolerates a larger lr), eval per epoch
- **Hardware:** single RTX 5080 — full run ≈ 77 s

## Run it

```bash
# from repo root, using the shared uv env
../.venv/bin/jupyter nbconvert --to notebook --execute qlora.ipynb
# or open qlora.ipynb in VS Code / Jupyter and run all cells
```

The SST-2 LoRA experiment is the second half of `qlora.ipynb`. Outputs land in
`lora-out/` (gitignored — adapter checkpoints are regenerable) and metrics stream to
Weights & Biases.

## Weights & Biases

- **Run:** `distilbert-sst2-lora`
- **Link:** https://wandb.ai/ezrealzheng11-new-york-university/huggingface/runs/su0dd3pr

The training/eval loss and `eval_accuracy` curves (per-epoch) are logged there via the
`report_to="wandb"` integration in `TrainingArguments`.

## Notebooks

- **`qlora.ipynb`** — the SST-2 LoRA run above, plus a QLoRA setup (4-bit NF4 quantized
  Mistral-7B via `BitsAndBytesConfig` + `prepare_model_for_kbit_training`) showing how
  LoRA stacks on top of a quantized backbone.
- **`d.ipynb`** — a smaller LoRA exploration on `meta-llama/Llama-3.2-1B`
  (`trainable% = 0.0424`), illustrating how the trainable fraction shrinks further as the
  base model grows.

## Takeaway

LoRA recovers most of full fine-tuning's quality (0.888 vs 0.904) while training ~1% of
the parameters and producing a tiny (~MB-scale) adapter instead of a full model copy —
the core argument for parameter-efficient fine-tuning.
