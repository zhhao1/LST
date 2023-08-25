#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from fairseq.data.audio.audio_utils import get_waveform, convert_waveform


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["dev","train", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)

        # Load source and target utterances
        for _lang in ["en", lang]:
            with open(txt_root / f"{split}.{_lang}") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)  
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(
            self, n: int
    ) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, \
            utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)  # n_frames = waveform.size(1)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id, wav_path, offset

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
    tgt_dir = Path(args.tgt_dir).absolute()
    for lang in args.languages:
        tgt_dir = tgt_dir / f"en-{lang}"
        os.makedirs(tgt_dir, exist_ok=True)
        
        # Generate TSV manifest
        print("Generating manifest...")
        train_text = []
        for split in MUSTC.SPLITS:
            print(f"Fetching split {split}...")
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = MUSTC(args.data_root, lang, split)
            for waveform, sr, src_utt, tgt_utt, spk_id, utt_id, wav_path, offset in tqdm(dataset):
                length = int(waveform.size(1))
                manifest["id"].append(utt_id)
                manifest["audio"].append(f"{wav_path}:{offset}:{length}")
                manifest["n_frames"].append(length)
                manifest["src_text"].append(src_utt)
                manifest["tgt_text"].append(tgt_utt)
                manifest["speaker"].append(spk_id)
            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, is_train_split=is_train_split, min_n_frames=1600, max_n_frames=480000)
            save_df_to_tsv(df, tgt_dir / f"{split}.tsv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--tgt-dir", required=True, type=str)
    parser.add_argument("--languages", type=str, nargs='+')
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
