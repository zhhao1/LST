import argparse
from pathlib import Path
from typing import Callable, Union
import pandas as pd
import os
import warnings

import sys
from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv
from asr.asr_inference import asr_inference
from filtering_utils import find_noisy_examples_zhhao, mustc_utterance_cleaner


DATASETS = ["MUSTC", "LIBRISPEECH"]
TRAIN_SPLITS = {
    "MUSTC": "train.tsv",
    "LIBRISPEECH": "train-all.tsv"}
CLEANER_FUNC = {
    "MUSTC": mustc_utterance_cleaner,
    "LIBRISPEECH": None,}
MODEL_NAME = "/home/zhhao/ssl_model/wav2vec_large_960h_lv60_self"

def filter(df: pd.DataFrame, tsv_path: Union[Path, str], dataset_name: str,
utterance_cleaner: Callable, asr_batch_size: int, asr_wer_threshold: float, max_example_number: int) -> pd.DataFrame:
    # text cleaning
    if dataset_name == 'MUSTC':
        df["src_text"] = df.apply(lambda x: utterance_cleaner(x["src_text"]), axis = 1)

    # removal of empty examples after cleaning
    empty_examples_bool = df.tgt_text == ""
    df = df.loc[~empty_examples_bool]
    print(f"removed {empty_examples_bool.sum()} empty examples. remaining: {len(df)}")

    # removal of noisy examples (based on ASR system predictions)
    asr_predictions_path = tsv_path.parent / f"{tsv_path.stem}_wer_results.json"
    if os.path.exists(asr_predictions_path):
        print("Already generate inference reults, skip this stage")
    else:
        print("Starting ASR inference with Wav2Vev_2.0 ...")
        asr_inference(tsv_path, asr_batch_size, dataset_name, MODEL_NAME)
    examples_bool = find_noisy_examples_zhhao(df, asr_predictions_path, asr_wer_threshold, max_example_number)
    df = df.loc[examples_bool]   
    print(f"preserve {examples_bool.sum()} examples")
    
    return df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type = str, required = True, choices = DATASETS)
    parser.add_argument("--tsv_root", type = str, required = True,
        help = "Path for the directory containing the TSV files for this DATASET.")
    parser.add_argument("--asr_batch_size", type = int, default = 24,
        help = "Batch size to be used during the ASR inference with Wav2Vec.")
    parser.add_argument("--asr_wer_threshold", type = float, default = .5,
        help = "Word-Error-Rate above which an example is considered noisy.")
    parser.add_argument("--max_example_number", type = int, default = 50000,
        help = "Preserved max example number.")
    args = parser.parse_args()

    tsv_root = Path(args.tsv_root)
    file_name = TRAIN_SPLITS[args.dataset_name]

    utterance_cleaner = CLEANER_FUNC[args.dataset_name]

    tsv_path = tsv_root / file_name
    df_split = load_df_from_tsv(tsv_path)

    print(f"Running filtering script for {file_name} of {args.dataset_name} from file {tsv_path}")
    df_split_filtered = filter(df_split, tsv_path, args.dataset_name, utterance_cleaner,
    args.asr_batch_size, args.asr_wer_threshold, args.max_example_number)

    new_tsv_path = tsv_root / Path(tsv_path.stem + "_filtered.tsv")
    save_df_to_tsv(df_split_filtered, new_tsv_path)
    print(f"Saved filtered TSV for: {file_name} of {args.dataset_name} at: {new_tsv_path}")


if __name__ == "__main__":
    main()
