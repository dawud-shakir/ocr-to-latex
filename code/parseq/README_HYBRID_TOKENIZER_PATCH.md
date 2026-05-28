# PARSeq hybrid LaTeX tokenizer patch

This patch keeps the original character-level tokenizer as the default and adds an optional `latex_hybrid` tokenizer.

## Files to copy into your repo

Copy these into the matching paths in your PARSeq repo:

```text
strhub/data/latex_tokenizer.py
strhub/models/base.py
strhub/models/parseq/system.py
strhub/models/parseq/model.py
configs/model/parseq.yaml
configs/charset/latex_hybrid.yaml
```

The copied `train.py`, `strhub/data/utils.py`, `configs/experiment/parseq-tiny.yaml`, and `configs/charset/94_full.yaml` include the tokenizer-aware pretrained transfer and train-without-validation support used by this patch.

## Default behavior

`configs/model/parseq.yaml` now has:

```yaml
tokenizer_type: char
latex_tokens: []
```

So existing character-level training should behave the same unless you opt in.

## Enable hybrid LaTeX tokenization

Use the new charset config override:

```bash
HYDRA_FULL_ERROR=1 python train.py \
  +experiment=parseq-tiny \
  pretrained=parseq-tiny \
  charset=latex_hybrid \
  'data.root_dir="/path/to/Training/data"' \
  data.train_dir=real \
  data.remove_whitespace=false \
  data.normalize_unicode=false \
  data.augment=true \
  model.max_label_length=44
```

With `tokenizer_type: latex_hybrid`, `model.max_label_length` is measured in hybrid tokens, not raw characters.
For example:

```text
\frac{x+1}{\alpha}
```

becomes:

```text
[\frac] [{] [x] [+] [1] [}] [{] [\alpha] [}]
```

## Pretrained loading

`train.py` uses token-aware pretrained transfer for vocabulary-dependent tensors:

```text
head.weight
head.bias
text_embed.embedding.weight
```

Matching tokens are copied by token string instead of raw row index. New multi-character hybrid tokens can optionally be initialized from the average of their spelling-piece rows when those characters exist in the pretrained charset.

`train.py` also partially transfers PARSeq `pos_queries` when only `model.max_label_length` changes. Public PARSeq checkpoints usually have 26 position queries because they were trained with `max_label_length=25` plus EOS. If you train with a longer value such as `model.max_label_length=32`, the first 26 positions are copied and only the extra positions stay randomly initialized. This is controlled by:

```yaml
pretrained_transfer_pos_queries_prefix: true
```

## 2026-05-25 tokenizer update

`HybridLatexTokenizer.tokenize()` now performs longest-match over every entry in
`latex_tokens`, not only backslash-starting commands. This allows structural
hybrid tokens such as `_{` and `^{` to work even when `_` and `^` are not present
as standalone characters in `charset_train`.

Example with `_` and `^` intentionally absent from `charset_train`:

```text
x_{2}        -> ['x', '_{', '2', '}']
y^{(0 + 0)}  -> ['y', '^{', '(', '0', ' ', '+', ' ', '0', ')', '}']
```


## 2026-05-26 full hybrid-data-path patch

This patch includes the full set of files needed for hybrid LaTeX token training:

- `strhub/data/latex_tokenizer.py`: longest-match hybrid tokens, including non-backslash tokens like `_{` and `^{`.
- `strhub/data/dataset.py`: accepts optional label adapter and token-length function; hybrid max length is measured in tokenizer units.
- `strhub/data/module.py`: when `tokenizer_type=latex_hybrid`, preserves raw LaTeX labels until `HybridLatexTokenizer` validates/tokenizes them. This prevents `\theta` from being stripped to `theta` before training.
- `strhub/models/base.py`: keeps raw LaTeX predictions/labels for validation metrics instead of applying `CharsetAdapter`.
- `configs/main.yaml`: passes `model.tokenizer_type` and `model.latex_tokens` into the data module.
- `configs/charset/latex_hybrid.yaml`: keeps `latex_tokens` under `model` so Hydra passes them into both the model and data module.

This patch also includes the train-without-validation convenience changes from the no-val patch, so it can be applied as one combined patch instead of applying hybrid and no-val patches separately.
