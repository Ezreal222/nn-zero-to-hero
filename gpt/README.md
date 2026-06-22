# char-GPT — a decoder-only Transformer from scratch

A character-level **decoder-only Transformer** (GPT) built from scratch in PyTorch,
trained on **tiny Shakespeare**. Follows Karpathy's "Let's build GPT" — every piece
(self-attention, multi-head, causal mask, residual + LayerNorm) is implemented by hand,
no `nn.Transformer`.

## Result

Trained `v2.py` for 5000 iterations on a single RTX 5080 (~7.9 min):

```
step    0: train loss 4.4753, val loss 4.4709
step 1000: train loss 1.6623, val loss 1.8284
step 2000: train loss 1.3871, val loss 1.6101
step 3000: train loss 1.2604, val loss 1.5272
step 4000: train loss 1.1758, val loss 1.4958
step 4500: train loss 1.1384, val loss 1.4852   <- final
```

**Final validation loss ≈ 1.485** (down from 4.47 at init).

### Generation sample

Sampling 500 chars from a cold-start context (`\n`):

```
DUKE VINCENTIO:
Reward, for I shall not his exterer Henry;
But that, this hopeful with pilease to be ,
And Menenius beggar-watery's topsy of his foot.

LADY CAPULET:
Now, fetcher: therefore no more it, no more lordship.
Farewell; flattion wast your son's love sbanishmen'd.
```

Not real words everywhere, but it has learned Shakespearean structure — speaker labels,
line breaks, punctuation, and roughly English-shaped tokens — purely from characters.

## Architecture

Decoder-only, pre-LayerNorm Transformer. Config in `v2.py`:

| Hyperparameter | Value |
|---|---|
| context length (`block_size`) | 256 |
| embedding dim (`n_embd`) | 384 |
| heads (`n_head`) | 6 |
| layers (`n_layer`) | 6 |
| dropout | 0.2 |
| batch size | 64 |
| learning rate | 3e-4 (AdamW) |
| iterations | 5000 |

The pieces, each a small `nn.Module`:

- **`Head`** — one self-attention head. Computes `wei = q·kᵀ / √d`, applies a **causal
  mask** (`tril` → `-inf` on future positions) so each token only attends to the past,
  then `softmax` and weighted sum of values.
- **`MultiHeadAttention`** — `n_head` heads in parallel, concatenated and projected.
- **`FeedForward`** — position-wise MLP (`n_embd → 4·n_embd → n_embd`, ReLU).
- **`Block`** — one Transformer block with **residual connections** around both
  sub-layers and **pre-LayerNorm**:
  ```python
  x = x + self.sa(self.ln1(x))     # communication (attention)
  x = x + self.ffwd(self.ln2(x))   # computation (MLP)
  ```
- **Model** — token embedding + learned positional embedding → 6 stacked blocks →
  final LayerNorm → linear head to vocab logits. `generate()` samples autoregressively,
  cropping the context to the last `block_size` tokens each step.

## Files

- **`v2.py`** — the full GPT (above). Run it to reproduce the result.
- **`build_gpt.ipynb`** — step-by-step build: bigram → single head → multi-head →
  blocks → full GPT, the way the model was developed.
- **`bigram.py`** — the simplest baseline (token embedding straight to logits), the
  starting point before attention.
- **`tokenizer.ipynb`** — tokenizer exploration (BPE), separate from the char-level model.
- **`input.txt`** — tiny Shakespeare (~1 MB), the training corpus.

## Run it

```bash
# from repo root, uses the shared uv env (PyTorch cu128)
cd gpt && ../.venv/bin/python v2.py
```

Prints train/val loss every 500 steps and a 500-character generation sample at the end.
