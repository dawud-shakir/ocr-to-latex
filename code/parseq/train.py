#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

import torch

from pytorch_lightning import Trainer

# To enable SWA, uncomment this import:
# from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.data.utils import Tokenizer
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights


# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    'Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.'
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000  # Can be anything. We use 1000 for convenience.
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


VOCAB_DEPENDENT_KEYS = {
    'head.weight',
    'head.bias',
    'text_embed.embedding.weight',
}


def _copy_pos_queries_prefix(
    pretrained_tensor: torch.Tensor,
    current_tensor: torch.Tensor,
) -> tuple[torch.Tensor | None, int, int, int]:
    """Partially copy learned PARSeq output-position queries.

    Public PARSeq checkpoints use max_label_length=25, so their pos_queries
    tensor normally has 26 positions: 25 label positions plus EOS. When fine-
    tuning with a longer or shorter max_label_length, the only expected shape
    change is the sequence-position dimension. Copy the shared prefix and keep
    any extra destination positions at their normal random initialization.

    Return (patched_tensor, copied_positions, source_positions, dest_positions).
    patched_tensor is None when the tensors are incompatible, e.g. different
    embedding dimensions from mixing parseq-tiny weights with parseq config.
    """
    if pretrained_tensor.ndim < 2 or current_tensor.ndim < 2:
        return None, 0, 0, 0
    if pretrained_tensor.ndim != current_tensor.ndim:
        return None, 0, int(pretrained_tensor.shape[1]), int(current_tensor.shape[1])

    # pos_queries is [1, max_label_length + 1, embed_dim] in PARSeq. Keep this
    # generic over leading/trailing dimensions, but only allow dim=1 to differ.
    if pretrained_tensor.shape[0] != current_tensor.shape[0]:
        return None, 0, int(pretrained_tensor.shape[1]), int(current_tensor.shape[1])
    if pretrained_tensor.shape[2:] != current_tensor.shape[2:]:
        return None, 0, int(pretrained_tensor.shape[1]), int(current_tensor.shape[1])

    src_positions = int(pretrained_tensor.shape[1])
    dst_positions = int(current_tensor.shape[1])
    copy_positions = min(src_positions, dst_positions)
    if copy_positions <= 0:
        return None, 0, src_positions, dst_positions

    out = current_tensor.clone()
    out[:, :copy_positions, ...].copy_(pretrained_tensor[:, :copy_positions, ...])
    return out, copy_positions, src_positions, dst_positions


def _get_cfg_value(config: DictConfig, key: str, default=None):
    """Best-effort OmegaConf get that also checks config.model."""
    try:
        value = config.get(key, default)
    except Exception:
        value = default
    if value is not None:
        return value
    try:
        return config.model.get(key, default)
    except Exception:
        return default


def _resolve_pretrained_charset_train(config: DictConfig) -> str:
    """Return the tokenizer charset used by the pretrained PARSeq checkpoint.

    The public PARSeq pretrained checkpoints, including parseq-tiny, use the
    repo's 94_full charset. Keep this configurable so a future checkpoint with a
    different source charset can still use token-aware row transfer safely.
    """
    explicit_charset = _get_cfg_value(config, 'pretrained_charset_train', None)
    if explicit_charset:
        return str(explicit_charset)

    charset_name = str(_get_cfg_value(config, 'pretrained_charset', '94_full') or '94_full')
    if charset_name.endswith('.yaml'):
        charset_name = charset_name[:-5]
    charset_path = Path(__file__).resolve().parent / 'configs' / 'charset' / f'{charset_name}.yaml'
    if not charset_path.exists():
        raise FileNotFoundError(
            f"Could not resolve pretrained charset config {charset_name!r}: {charset_path}. "
            "Set pretrained_charset_train explicitly if this checkpoint did not use configs/charset/94_full.yaml."
        )
    charset_cfg = OmegaConf.load(charset_path)
    return str(charset_cfg.model.charset_train)


def _tokenizer_maps(tokenizer) -> tuple[dict[str, int], dict[int, str]]:
    stoi = {str(k): int(v) for k, v in getattr(tokenizer, '_stoi', {}).items()}
    itos_raw = getattr(tokenizer, '_itos', ())
    itos = {int(i): str(tok) for i, tok in enumerate(itos_raw)}
    return stoi, itos


def _is_predicted_row(tokenizer, token_id: int, num_rows: int) -> bool:
    """Return True when a tokenizer id has a corresponding output-head row."""
    # PARSeq's head predicts every token except BOS and PAD. In the tokenizer
    # layouts used here, BOS/PAD are the final ids and therefore outside the
    # head row count, but keep this explicit for readability/safety.
    bos_id = getattr(tokenizer, 'bos_id', None)
    pad_id = getattr(tokenizer, 'pad_id', None)
    return 0 <= int(token_id) < int(num_rows) and token_id not in {bos_id, pad_id}


def _copy_vocab_rows_by_token(
    key: str,
    pretrained_tensor: torch.Tensor,
    current_tensor: torch.Tensor,
    source_tokenizer: Tokenizer,
    dest_tokenizer,
    *,
    init_new_from_spelling: bool = True,
) -> tuple[torch.Tensor, int, int, int]:
    """Copy vocab-dependent rows by token string rather than by raw row index.

    This avoids both failure modes:
      1. shape mismatch -> all character rows skipped;
      2. same shape but different token order -> silent class-meaning mismatch.

    For new multi-character hybrid tokens such as r"\theta", optionally initialize
    their rows from the average of their source character rows (e.g. '\\', 't',
    'h', 'e', 't', 'a') when those characters exist in the pretrained tokenizer.
    """
    out = current_tensor.clone()
    src_stoi, _src_itos = _tokenizer_maps(source_tokenizer)
    dst_stoi, dst_itos = _tokenizer_maps(dest_tokenizer)

    if pretrained_tensor.ndim != current_tensor.ndim:
        return out, 0, 0, 0
    if pretrained_tensor.ndim > 1 and pretrained_tensor.shape[1:] != current_tensor.shape[1:]:
        return out, 0, 0, 0

    src_rows = int(pretrained_tensor.shape[0])
    dst_rows = int(current_tensor.shape[0])
    is_head = key in {'head.weight', 'head.bias'}

    direct = 0
    spelling = 0
    skipped = 0

    for dst_id in range(dst_rows):
        token = dst_itos.get(dst_id)
        if token is None:
            skipped += 1
            continue
        if is_head and not _is_predicted_row(dest_tokenizer, dst_id, dst_rows):
            continue

        src_id = src_stoi.get(token)
        if src_id is not None:
            if (not is_head or _is_predicted_row(source_tokenizer, src_id, src_rows)) and 0 <= src_id < src_rows:
                out[dst_id].copy_(pretrained_tensor[src_id])
                direct += 1
            else:
                skipped += 1
            continue

        if not init_new_from_spelling or len(token) <= 1:
            skipped += 1
            continue

        piece_ids: list[int] = []
        can_spell = True
        for ch in token:
            ch_id = src_stoi.get(ch)
            if ch_id is None or ch_id < 0 or ch_id >= src_rows:
                can_spell = False
                break
            if is_head and not _is_predicted_row(source_tokenizer, ch_id, src_rows):
                can_spell = False
                break
            piece_ids.append(ch_id)

        if can_spell and piece_ids:
            rows = pretrained_tensor[piece_ids]
            out[dst_id].copy_(rows.mean(dim=0))
            spelling += 1
        else:
            skipped += 1

    return out, direct, spelling, skipped


def _load_pretrained_token_aware(model: BaseSystem, inner_model: torch.nn.Module, config: DictConfig) -> None:
    """Load pretrained weights with token-aware row transfer for resized vocab layers."""
    pretrained = get_pretrained_weights(config.pretrained)
    current = inner_model.state_dict()

    transfer_vocab_by_token = bool(_get_cfg_value(config, 'pretrained_transfer_vocab_by_token', True))
    init_new_from_spelling = bool(_get_cfg_value(config, 'pretrained_init_new_tokens_from_spelling', True))
    transfer_pos_queries_prefix = bool(_get_cfg_value(config, 'pretrained_transfer_pos_queries_prefix', True))

    dest_tokenizer = getattr(model, 'tokenizer', None)
    source_tokenizer = None
    if transfer_vocab_by_token and dest_tokenizer is not None:
        pretrained_charset_train = _resolve_pretrained_charset_train(config)
        source_tokenizer = Tokenizer(pretrained_charset_train)
        print(
            "Token-aware pretrained transfer: "
            f"source pretrained_charset={_get_cfg_value(config, 'pretrained_charset', '94_full')!r}, "
            f"source_vocab={len(source_tokenizer)}, dest_vocab={len(dest_tokenizer)}, "
            f"init_new_from_spelling={init_new_from_spelling}",
            flush=True,
        )

    loadable: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    row_transfer_report: list[str] = []
    pos_query_transfer_report: list[str] = []

    for key, pretrained_tensor in pretrained.items():
        if key not in current:
            continue

        current_tensor = current[key]

        # Vocab-dependent tensors need token-name-aware transfer even when the
        # shapes happen to match. Same shape does not guarantee same token ids.
        if key in VOCAB_DEPENDENT_KEYS and source_tokenizer is not None and dest_tokenizer is not None:
            patched, direct, spelling, row_skipped = _copy_vocab_rows_by_token(
                key,
                pretrained_tensor,
                current_tensor,
                source_tokenizer,
                dest_tokenizer,
                init_new_from_spelling=init_new_from_spelling,
            )
            loadable[key] = patched
            row_transfer_report.append(
                f"{key}: copied_matching={direct}, initialized_from_spelling={spelling}, left_random={row_skipped}"
            )
            continue

        if pretrained_tensor.shape == current_tensor.shape:
            loadable[key] = pretrained_tensor
        elif key == 'pos_queries' and transfer_pos_queries_prefix:
            patched, copied_positions, src_positions, dst_positions = _copy_pos_queries_prefix(
                pretrained_tensor,
                current_tensor,
            )
            if patched is not None:
                loadable[key] = patched
                pos_query_transfer_report.append(
                    f"pos_queries: copied_prefix_positions={copied_positions}, "
                    f"source_positions={src_positions}, dest_positions={dst_positions}, "
                    f"left_random={max(dst_positions - copied_positions, 0)}"
                )
            else:
                skipped.append(key)
        else:
            skipped.append(key)

    if row_transfer_report:
        print("Token-aware vocab row transfer:", flush=True)
        for line in row_transfer_report:
            print("  " + line, flush=True)

    if pos_query_transfer_report:
        print("Partial pos_queries transfer:", flush=True)
        for line in pos_query_transfer_report:
            print("  " + line, flush=True)

    if skipped:
        print("Skipped pretrained layers due to shape mismatch:", skipped, flush=True)
    else:
        print("Skipped pretrained layers due to shape mismatch: []", flush=True)

    inner_model.load_state_dict(loadable, strict=False)


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = 'auto'
    train_without_val = bool(config.get('train_without_val', False))

    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)

        # Optional train-only mode. This is useful for quick experiments or a final
        # train-on-everything run where root_dir/val is intentionally absent.
        # Normal behavior is unchanged when train_without_val is false.
        config.data.allow_missing_val = bool(config.data.get('allow_missing_val', False) or train_without_val)
        if train_without_val:
            config.trainer.limit_val_batches = 0
            config.trainer.num_sanity_val_steps = 0

        # Special handling for GPU-affected config
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        if gpu:
            # Use mixed-precision training
            config.trainer.precision = 'bf16-mixed' if torch.get_autocast_gpu_dtype() is torch.bfloat16 else '16-mixed'
        if gpu and devices > 1:
            # Use DDP with optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            # Scale steps-based config
            if not train_without_val and config.trainer.get('val_check_interval') is not None:
                config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
    # If specified, use pretrained weights to initialize the model.
    if config.pretrained is not None:
        m = model.model if config.model._target_.endswith('PARSeq') else model

        # Hybrid LaTeX fine-tuning changes the tokenizer/vocabulary. A plain
        # shape-compatible load is not enough: if token order changes, same-shaped
        # rows can silently mean the wrong class. Instead, copy vocab-dependent
        # rows by token string. Existing characters such as "a", "t", "=" keep
        # their pretrained rows, while new hybrid tokens such as r"\theta" or
        # r"_{" are initialized fresh or from the average of their spelling pieces.
        _load_pretrained_token_aware(model, m, config)

    print(summarize(model, max_depth=2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    if train_without_val:
        # Train-only mode: validation metrics will not exist, so checkpoint by
        # epoch-aggregated training loss instead of val_NED.
        checkpoint = ModelCheckpoint(
            monitor='train_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            save_on_train_epoch_end=True,
            filename='{epoch}-{step}-{train_loss:.4f}',
        )
        if trainer_strategy == 'auto':
            print(
                "Training without validation: checkpointing by train_loss. "
                "No val_accuracy/val_NED metrics will be logged.",
                flush=True,
            )
    else:
        # Select checkpoints by validation NED instead of exact-match accuracy. (dawud 2026-05-19)
        checkpoint = ModelCheckpoint(
            monitor='val_NED',
            mode='max',
            save_top_k=3,
            save_last=True,
            filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
        )
        # checkpoint = ModelCheckpoint(
        #     monitor='val_accuracy',
        #     mode='max',
        #     save_top_k=3,
        #     save_last=True,
        #     filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
        # )

    # To enable SWA, uncomment the three lines below, restore the import above, and
    # change callbacks=[checkpoint] to callbacks=[checkpoint, swa].  (dawud, 2026-05-19)
    #
    # swa_epoch_start = 0.75
    # swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    # swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
    cwd = (
        HydraConfig.get().runtime.output_dir
        if config.ckpt_path is None
        else str(Path(config.ckpt_path).parents[1].absolute())
    )
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=TensorBoardLogger(cwd, '', '.'),
        strategy=trainer_strategy,
        enable_model_summary=False,
        callbacks=[checkpoint],
        # If SWA is enabled above, use this instead:
        # callbacks=[checkpoint, swa],
        
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
