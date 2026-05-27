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
