# PARSeq full hybrid-tokenizer/data-path patch

This is a combined patch for hybrid LaTeX tokenization.

It includes:

- Hybrid tokenizer support, including non-backslash tokens like `_{` and `^{`.
- Dataset/data-module fixes so raw LaTeX labels are preserved until `HybridLatexTokenizer` sees them.
- Hybrid token-length filtering, so `max_label_length` is measured in hybrid tokens instead of raw string characters.
- Validation/test metric fix so decoded LaTeX predictions are not stripped by `CharsetAdapter`.
- Train-without-validation support from the no-val patch.

## Apply

Copy this patch folder's contents into the PARSeq repo root, overwriting files.

Your Colab patch-loader can use the zip directly if it looks for a patch root containing:

```text
strhub/data/latex_tokenizer.py
configs/charset/latex_hybrid.yaml
```

## Important

After applying this patch, retrain from scratch or from the pretrained PARSeq weights. Old checkpoints trained before this fix may have learned stripped labels such as `theta` instead of the hybrid token `\theta`.

## 2026-05-27 token-aware pretrained row transfer

The training loader now handles resized/custom vocabularies more carefully.
Instead of skipping the entire vocab-dependent tensors when the hybrid charset
changes shape, it transfers rows by token string:

```text
pretrained "a" row  -> current "a" row
pretrained "t" row  -> current "t" row
pretrained "=" row  -> current "=" row
new "\\theta" row   -> initialized from spelling pieces when possible
new "_{" row        -> initialized from spelling pieces when possible
```

This applies to:

```text
head.weight
head.bias
text_embed.embedding.weight
```

The loader also uses token-name-aware transfer even if a future custom vocabulary
happens to have the same shape as the pretrained vocabulary. This prevents silent
class-meaning misalignment when token order changes.

Config options in `configs/main.yaml`:

```yaml
pretrained_charset: 94_full
pretrained_charset_train: null
pretrained_transfer_vocab_by_token: true
pretrained_init_new_tokens_from_spelling: true
```

For normal `pretrained=parseq-tiny`, leave these defaults alone.
