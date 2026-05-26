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

The copied `train.py`, `strhub/data/utils.py`, `configs/experiment/parseq-tiny.yaml`, and `configs/charset/94_full.yaml` are included only for context and are unchanged.

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

Your current `train.py` already loads only shape-compatible pretrained tensors. That is important because hybrid tokenization changes the output vocabulary size, so token-specific layers such as `head` and `text_embed` will be reinitialized while compatible encoder/decoder weights are reused.
