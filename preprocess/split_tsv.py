import csv, argparse, os
import pandas as pd
from pathlib import Path
from examples.speech_to_text.data_utils import (
    save_df_to_tsv,
)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "speaker",
                    "src_text", "tgt_text", "src_lang", "tgt_lang"]

def load_samples_from_tsv(root: str, split: str):
    tsv_path = Path(root) / f"{split}.tsv"
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {tsv_path}")
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples = [dict(e) for e in reader]
    if len(samples) == 0:
        raise ValueError(f"Empty manifest: {tsv_path}")
    return samples

root = '/home/zhhao/data_source/MUST-C/'
tsv_file = 'tst-COMMON_st'

samples = load_samples_from_tsv(root, tsv_file)

manifest0_5 = {c: [] for c in MANIFEST_COLUMNS}
manifest5_10 = {c: [] for c in MANIFEST_COLUMNS}
manifest10_15 = {c: [] for c in MANIFEST_COLUMNS}
manifest15_20 = {c: [] for c in MANIFEST_COLUMNS}
manifest20_50 = {c: [] for c in MANIFEST_COLUMNS}

for sample in samples:
    duration = int(sample['n_frames'])/16000 
    if duration < 5:
        for column in MANIFEST_COLUMNS:
            manifest0_5[column].append(sample[column])
    elif duration < 10:
        for column in MANIFEST_COLUMNS:
            manifest5_10[column].append(sample[column])        
    elif duration < 15:
        for column in MANIFEST_COLUMNS:
            manifest10_15[column].append(sample[column])  
    elif duration < 20:
        for column in MANIFEST_COLUMNS:
            manifest15_20[column].append(sample[column])
    else:
        for column in MANIFEST_COLUMNS:
            manifest20_50[column].append(sample[column])      
                    
save_df_to_tsv(
    pd.DataFrame.from_dict(manifest0_5), os.path.join(root, tsv_file + '0_5.tsv')
)
            
save_df_to_tsv(
    pd.DataFrame.from_dict(manifest5_10), os.path.join(root, tsv_file + '5_10.tsv')
)

save_df_to_tsv(
    pd.DataFrame.from_dict(manifest10_15), os.path.join(root, tsv_file + '10_15.tsv')
)

save_df_to_tsv(
    pd.DataFrame.from_dict(manifest15_20), os.path.join(root, tsv_file + '15_20.tsv')
)

save_df_to_tsv(
    pd.DataFrame.from_dict(manifest20_50), os.path.join(root, tsv_file + '20_50.tsv')
)


