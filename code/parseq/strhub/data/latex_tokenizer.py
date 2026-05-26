# Scene Text Recognition Model Hub
# Hybrid LaTeX tokenizer extension for PARSeq fine-tuning.
#
# This file intentionally leaves the original character tokenizer untouched.
# It adds a tokenizer that can treat selected LaTeX commands/structures
# (for example \frac, \alpha, _{, or ^{) as single sequence tokens while
# still falling back to ordinary character tokens for everything else.

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from strhub.data.utils import BaseTokenizer


class HybridLatexTokenizer(BaseTokenizer):
    """Hybrid tokenizer for LaTeX-like OCR labels.

    Predictable tokens are ordered like PARSeq's original Tokenizer:

        [E], predictable tokens..., [B], [P]

    This matters because the PARSeq output head predicts every token except
    BOS and PAD. EOS must therefore remain inside the output-head range.

    `latex_tokens` should contain multi-character LaTeX commands/structures
    such as r"\\frac", r"\\sqrt", r"\\alpha", r"_{", r"^{", etc.
    Any text not matched by those tokens is encoded character-by-character
    using `char_tokens`.
    """

    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, char_tokens: str, latex_tokens: Optional[Sequence[str]] = None) -> None:
        latex_tokens = list(latex_tokens or [])

        # Keep only meaningful multi-character commands here. Single-character
        # entries already belong in char_tokens, and duplicates would make the
        # id-to-token table ambiguous.
        seen_latex: set[str] = set()
        cleaned_latex_tokens: list[str] = []
        for tok in latex_tokens:
            tok = str(tok)
            if len(tok) <= 1:
                continue
            if tok in seen_latex:
                continue
            seen_latex.add(tok)
            cleaned_latex_tokens.append(tok)

        # Longest first prevents partial matches when one token is a prefix
        # of another, e.g. r"\\right" before r"\\rightarrow". It also lets
        # structural tokens such as r"_{" and r"^{" be recognized before their
        # individual fallback characters.
        self.latex_tokens = tuple(sorted(cleaned_latex_tokens, key=len, reverse=True))

        # Preserve charset order while removing duplicates.
        self.char_tokens = tuple(dict.fromkeys(char_tokens))

        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        predictable_tokens = self.latex_tokens + self.char_tokens

        self._itos = specials_first + predictable_tokens + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def tokenize(self, label: str) -> list[str]:
        """Tokenize one label using longest-match hybrid-token matching.

        Matching is intentionally *not* limited to backslash-starting commands.
        This lets structural tokens like r"_{" and r"^{" work even when "_"
        and "^" are not standalone fallback characters in `char_tokens`.

        Unknown LaTeX commands are not fatal as long as their characters are in
        `char_tokens`; for example, if r"\\operatorname" is not listed as one token,
        it becomes '\\', 'o', 'p', ... instead.
        """
        tokens: list[str] = []
        i = 0
        while i < len(label):
            matched = None

            # Longest-match over every hybrid token, not just tokens beginning
            # with backslash. This supports both command tokens (\frac) and
            # structural tokens (_{, ^{).
            for tok in self.latex_tokens:
                if label.startswith(tok, i):
                    matched = tok
                    break

            if matched is not None:
                tokens.append(matched)
                i += len(matched)
                continue

            ch = label[i]
            if ch not in self._stoi:
                raise KeyError(
                    f"Unsupported label character {ch!r} at index {i} in {label!r}. "
                    "Add it to charset_train, add a matching latex_tokens entry, "
                    "or convert it before training."
                )
            tokens.append(ch)
            i += 1

        return tokens

    def _tok2ids(self, tokens: Sequence[str] | str) -> list[int]:
        # Accept either a pre-tokenized list or a raw label string.
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)
        return [self._stoi[s] for s in tokens]

    def encode(self, labels: list[str], device: Optional[torch.device] = None) -> Tensor:
        batch = [
            torch.as_tensor(
                [self.bos_id] + self._tok2ids(label) + [self.eos_id],
                dtype=torch.long,
                device=device,
            )
            for label in labels
        ]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> tuple[Tensor, list[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)
        ids = ids[:eos_idx]
        probs = probs[: eos_idx + 1]
        return probs, ids
