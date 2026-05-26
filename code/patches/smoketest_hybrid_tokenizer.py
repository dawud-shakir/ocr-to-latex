#!/usr/bin/env python3
r"""
Smoke-test PARSeq's hybrid LaTeX tokenizer patch.

This test focuses on HybridLatexTokenizer behavior, especially hybrid tokens
that do not start with a backslash, such as _{ and ^{.

It also tests the important overlap case:

    char_tokens includes '_' and '{'
    latex_tokens includes '_{'

In that case the tokenizer should still choose the longer hybrid token '_{'
instead of falling back to '_' then '{'.

Usage:
    python smoketest_hybrid_tokenizer_patch_overlap_20260526.py \
      --patch /path/to/parseq_hybrid_tokenizer_patch_non_backslash_20260525.zip \
      --verbose

Or:
    python smoketest_hybrid_tokenizer_patch_overlap_20260526.py --repo /path/to/patched/parseq --verbose
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Iterable, Optional


def _purge_strhub_modules() -> None:
    """Force a clean import from the requested patch/repo path."""
    for name in list(sys.modules):
        if name == "strhub" or name.startswith("strhub."):
            del sys.modules[name]


def _looks_like_parseq_root(path: Path) -> bool:
    return (path / "strhub" / "data" / "utils.py").is_file()


def _resolve_patch_root(path: Path) -> tuple[Path, Optional[tempfile.TemporaryDirectory[str]]]:
    """Return a directory that should be inserted onto sys.path."""
    path = path.expanduser().resolve()
    tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None

    if path.is_file() and path.suffix.lower() == ".zip":
        tmpdir = tempfile.TemporaryDirectory(prefix="parseq_hybrid_patch_smoketest_")
        extract_dir = Path(tmpdir.name)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)

        candidates = [extract_dir]
        candidates += [p for p in extract_dir.rglob("*") if p.is_dir()]
        for cand in candidates:
            if _looks_like_parseq_root(cand):
                return cand.resolve(), tmpdir
        raise FileNotFoundError(f"Could not find strhub/data/utils.py inside zip: {path}")

    if path.is_dir():
        candidates = [path]
        candidates += [path / "parseq_hybrid_tokenizer_patch", path / "parseq", path / "PARSeq"]
        candidates += [p for p in path.rglob("strhub") if p.is_dir()][:10]
        for cand in candidates:
            root = cand.parent if cand.name == "strhub" else cand
            if _looks_like_parseq_root(root):
                return root.resolve(), None
        raise FileNotFoundError(f"Could not find a PARSeq/patch root under: {path}")

    raise FileNotFoundError(f"Patch/repo path does not exist: {path}")


def _import_tokenizer(root: Path):
    _purge_strhub_modules()
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    module = importlib.import_module("strhub.data.latex_tokenizer")
    tokenizer_cls = getattr(module, "HybridLatexTokenizer")
    return tokenizer_cls, module


def _check_equal(name: str, got, expected) -> None:
    if got != expected:
        raise AssertionError(f"{name}\n  got:      {got!r}\n  expected: {expected!r}")


def _check_raises(name: str, fn, expected_substring: str = "") -> None:
    try:
        fn()
    except Exception as exc:
        if expected_substring and expected_substring not in str(exc):
            raise AssertionError(
                f"{name}\n  raised expected exception type, but message did not contain "
                f"{expected_substring!r}: {exc!r}"
            ) from exc
        return
    raise AssertionError(f"{name}\n  expected an exception, but none was raised")


def _roundtrip_decode(tokenizer, labels: list[str]) -> list[str]:
    """Encode labels, build one-hot token distributions, decode them back."""
    import torch

    encoded = tokenizer.encode(labels)
    # PARSeq predicts target positions after BOS: label tokens + EOS + PAD.
    target_ids = encoded[:, 1:]
    probs = torch.zeros(
        target_ids.shape[0],
        target_ids.shape[1],
        len(tokenizer),
        dtype=torch.float32,
    )
    for i in range(target_ids.shape[0]):
        for j in range(target_ids.shape[1]):
            probs[i, j, int(target_ids[i, j])] = 1.0
    decoded, _conf = tokenizer.decode(probs)
    return decoded


def run_tests(tokenizer_cls, verbose: bool = False) -> None:
    """Run focused checks for non-backslash, mixed, and overlapping hybrid tokens."""
    tests_run = 0

    def show(label: str, tokens: list[str]) -> None:
        if verbose:
            print(f"  {label!r} -> {tokens}")

    # 1) Minimal charset: underscore/caret/backslash are intentionally absent.
    #    The only way x_{2} and y^{...} can pass is if _{ and ^{ are matched
    #    as complete hybrid tokens.
    tok = tokenizer_cls(
        char_tokens="0123456789xy()+ =}",
        latex_tokens=["_{", "^{"],
    )
    got = tok.tokenize("x_{2}")
    show("x_{2}", got)
    _check_equal("non-backslash token _{ should be recognized", got, ["x", "_{", "2", "}"])
    tests_run += 1

    got = tok.tokenize("y^{(0 + 0)}")
    show("y^{(0 + 0)}", got)
    _check_equal(
        "non-backslash token ^{ should be recognized",
        got,
        ["y", "^{", "(", "0", " ", "+", " ", "0", ")", "}"],
    )
    tests_run += 1

    _check_raises(
        "x_2 should fail when '_' is not in charset and '_{' is not present in the label",
        lambda: tok.tokenize("x_2"),
        "Unsupported label character '_'",
    )
    tests_run += 1

    # 2) Mixed command tokens and structural tokens. The charset below does not
    #    contain backslash, underscore, or caret, so \sum, _{, and ^{ must all
    #    be matched as hybrid tokens.
    tok = tokenizer_cls(
        char_tokens="12345n=}",
        latex_tokens=[r"\sum", "_{", "^{"],
    )
    label = r"\sum_{n=1}^{5}"
    got = tok.tokenize(label)
    show(label, got)
    _check_equal(
        "mixed command + non-backslash structure tokens",
        got,
        [r"\sum", "_{", "n", "=", "1", "}", "^{", "5", "}"],
    )
    tests_run += 1

    # 3) Longest-match with non-backslash tokens. If sorting/matching is wrong,
    #    this would tokenize as _{ n = 1 } instead of _{n=1}.
    tok = tokenizer_cls(
        char_tokens="x5}",
        latex_tokens=["_{", "_{n=", "_{n=1}", "^{"],
    )
    label = r"x_{n=1}^{5}"
    got = tok.tokenize(label)
    show(label, got)
    _check_equal(
        "longest-match should work for tokens that do not start with backslash",
        got,
        ["x", "_{n=1}", "^{", "5", "}"],
    )
    tests_run += 1

    # 4) Longest-match with command tokens too, for sanity.
    tok = tokenizer_cls(
        char_tokens="",
        latex_tokens=[r"\right", r"\rightarrow", r"\to"],
    )
    label = r"\rightarrow"
    got = tok.tokenize(label)
    show(label, got)
    _check_equal(
        "longest-match should prefer \\rightarrow over \\right or \\to",
        got,
        [r"\rightarrow"],
    )
    tests_run += 1

    # 5) Overlap case: every character in the hybrid token is also available
    #    as a fallback char. The tokenizer should still choose the longer hybrid
    #    token first.
    tok = tokenizer_cls(
        char_tokens="x2_{}",
        latex_tokens=["_{"],
    )
    label = "x_{2}"
    got = tok.tokenize(label)
    show(label, got)
    _check_equal(
        "hybrid token _{ should win even when '_' and '{' are fallback chars",
        got,
        ["x", "_{", "2", "}"],
    )
    tests_run += 1

    tok = tokenizer_cls(
        char_tokens="y0^{}",
        latex_tokens=["^{"],
    )
    label = "y^{0}"
    got = tok.tokenize(label)
    show(label, got)
    _check_equal(
        "hybrid token ^{ should win even when '^' and '{' are fallback chars",
        got,
        ["y", "^{", "0", "}"],
    )
    tests_run += 1

    # 6) Overlap with a backslash command. This ensures \theta wins over the
    #    fallback character sequence '\', 't', 'h', ... when both are possible.
    tok = tokenizer_cls(
        char_tokens=r"\theta2_{ }",
        latex_tokens=[r"\theta", "_{"],
    )
    label = r"\theta_{2}"
    got = tok.tokenize(label)
    show(label, got)
    _check_equal(
        "command token \\theta should win over fallback characters when both are available",
        got,
        [r"\theta", "_{", "2", "}"],
    )
    tests_run += 1

    # 7) Encode/decode round trip with a hybrid tokenizer.
    tok = tokenizer_cls(
        char_tokens="0123456789xyn()+ =}",
        latex_tokens=[r"\sum", "_{", "^{"],
    )
    labels = ["x_{2}", "y^{(0 + 0)}", r"\sum_{n=1}^{5}"]
    decoded = _roundtrip_decode(tok, labels)
    if verbose:
        print("  roundtrip decoded:", decoded)
    _check_equal("encode/decode round trip", decoded, labels)
    tests_run += 1

    print(f"All hybrid-tokenizer smoke tests passed ({tests_run} checks).")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test PARSeq HybridLatexTokenizer patch.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--patch", type=Path, help="Path to patch zip or extracted patch folder.")
    src.add_argument("--repo", type=Path, help="Path to a patched PARSeq repo root.")
    parser.add_argument("--verbose", action="store_true", help="Print tokenized examples.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        root, tmpdir = _resolve_patch_root(args.patch or args.repo)
        print(f"Using PARSeq/patch root: {root}")
        tokenizer_cls, module = _import_tokenizer(root)
        print(f"Imported HybridLatexTokenizer from: {Path(module.__file__).resolve()}")
        run_tests(tokenizer_cls, verbose=args.verbose)
        return 0
    except Exception:
        print("\nSMOKE TEST FAILED", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
