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
import glob
import io
import logging
import unicodedata
from pathlib import Path, PurePath
from typing import Callable, Optional, Union

import lmdb
from PIL import Image

from torch.utils.data import ConcatDataset, Dataset

from strhub.data.utils import CharsetAdapter

log = logging.getLogger(__name__)


def build_tree_dataset(root: Union[PurePath, str], *args, **kwargs):
    try:
        kwargs.pop('root')  # prevent 'root' from being passed via kwargs
    except KeyError:
        pass
    root = Path(root).absolute()
    log.info(f'dataset root:\t{root}')
    datasets = []
    for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
        mdb = Path(mdb)
        ds_name = str(mdb.parent.relative_to(root))
        ds_root = str(mdb.parent.absolute())
        dataset = LmdbDataset(ds_root, *args, **kwargs)
        log.info(f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}')
        datasets.append(dataset)
    return ConcatDataset(datasets)


class LmdbDataset(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(
        self,
        root: str,
        charset: str,
        max_label_len: int,
        min_image_dim: int = 0,
        remove_whitespace: bool = True,
        normalize_unicode: bool = True,
        unlabelled: bool = False,
        transform: Optional[Callable] = None,
        label_adapter: Optional[Callable[[str], str]] = None,
        label_length: Optional[Callable[[str], int]] = None,
    ):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = []
        self.filtered_index_list = []
        self.num_samples = self._preprocess_labels(
            charset,
            remove_whitespace,
            normalize_unicode,
            max_label_len,
            min_image_dim,
            label_adapter=label_adapter,
            label_length=label_length,
        )

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(
            self.root, max_readers=1, readonly=True, create=False, readahead=False, meminit=False, lock=False
        )

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(
        self,
        charset,
        remove_whitespace,
        normalize_unicode,
        max_label_len,
        min_image_dim,
        *,
        label_adapter: Optional[Callable[[str], str]] = None,
        label_length: Optional[Callable[[str], int]] = None,
    ):
        # Default PARSeq behavior: adapt raw labels with a character-level
        # CharsetAdapter. Hybrid LaTeX training passes its own adapter/length
        # function so labels such as r"\theta" and r"x_{2}" are validated by
        # the tokenizer instead of being character-stripped before tokenization.
        charset_adapter = label_adapter or CharsetAdapter(charset)
        length_fn = label_length or len
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return num_samples
            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                # Normally, whitespace is removed from the labels.
                if remove_whitespace:
                    label = ''.join(label.split())
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                if normalize_unicode:
                    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
                try:
                    label = charset_adapter(label)
                    label_len = length_fn(label)
                except Exception as exc:
                    log.debug('Skipping label %r from %s: %s', label, self.root, exc)
                    continue
                # We filter out samples which don't contain any supported tokens/characters.
                if not label:
                    continue
                # For hybrid tokenizers, max_label_len is measured in tokenizer
                # units, not raw Python characters. Example: r"\theta" has raw
                # length 6 but hybrid-token length 1.
                if label_len > max_label_len:
                    continue
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < min_image_dim or h < min_image_dim:
                        continue
                self.labels.append(label)
                self.filtered_index_list.append(index)
        return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.unlabelled:
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]

        img_key = f'image-{index:09d}'.encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label
