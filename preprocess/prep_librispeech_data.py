#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse, os
import logging
import shutil
from pathlib import Path
from typing import Tuple, Union
from torch import Tensor
from tempfile import NamedTemporaryFile

import pandas as pd
from examples.speech_to_text.data_utils import (
    save_df_to_tsv,
)
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm


log = logging.getLogger(__name__)

SPLITS_TEST = [
    "test-clean",
    "test-other",
    "dev-clean",
    "dev-other",
]

SPLITS_TRAIN = [
    "train-clean-100",
    "train-other-500",
    "train-clean-360",
]

MANIFEST_COLUMNS = ["id", "audio", "tgt_text", "speaker"]

class librispeech(LIBRISPEECH):
    def __init__(self, path, folder_in_archive, url):
        super(librispeech, self).__init__(path, folder_in_archive=folder_in_archive, url=url)
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        return self.get_metadata(n)

def process_test(args):
    data_root = Path(args.data_root).absolute()
    tgt_dir = Path(args.tgt_dir).absolute()
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    for split in SPLITS_TEST:
        print(f"Processing {split}...")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = librispeech(data_root, folder_in_archive='.', url=split)
        for path, sr, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
            sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
            manifest["id"].append(sample_id)
            manifest["audio"].append(os.path.join(data_root,path))
            manifest["tgt_text"].append(utt.lower())
            manifest["speaker"].append(spk_id)
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), tgt_dir / f"{split}.tsv"
        )

def process_train(args):
    data_root = Path(args.data_root).absolute()
    tgt_dir = Path(args.tgt_dir).absolute()
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    for split in SPLITS_TRAIN:
        print(f"Processing {split}...")
        dataset = librispeech(data_root, folder_in_archive='.', url=split)
        for path, sr, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
            sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
            manifest["id"].append(sample_id)
            manifest["audio"].append(os.path.join(data_root,path))
            manifest["tgt_text"].append(utt.lower())
            manifest["speaker"].append(spk_id)
    save_df_to_tsv(
        pd.DataFrame.from_dict(manifest), tgt_dir / "train-all.tsv"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-o", required=True, type=str)
    parser.add_argument("--tgt-dir", required=True, type=str)
    parser.add_argument("--mode", required=True, type=str, choices=["train", "test"],)
    args = parser.parse_args()
    if args.mode == 'test':
        process_test(args)
    else:
        process_train(args)

if __name__ == "__main__":
    main()
