import json, csv
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re
import numpy as np

df = pd.DataFrame(np.random.rand(10, 2), columns=list('ab'))

df = df.assign(WER = 999)
print(df)
df = df.sort_values(by='b', ascending=True) 
print(df)
examples_bool = df.b < 0.3
print(examples_bool)
print(examples_bool.sum())
if examples_bool.sum() > 2:
    examples_bool[2:] = False
print(examples_bool)
df = df.loc[examples_bool]
print(df)
sd




asr_predictions_file = './train-all_wer_results.json'

def load_df_from_tsv(path: Union[str, Path]) -> pd.DataFrame:
    _path = path if isinstance(path, str) else path.as_posix()
    return pd.read_csv(
        _path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )

df = load_df_from_tsv('./train-all.tsv')

with open(asr_predictions_file, "r") as file:
    ids = [json.loads(line)["id"] for line in file]
with open(asr_predictions_file, "r") as file:
    wer = [float(json.loads(line)["WER"]) for line in file]
ids_to_wer = dict(zip(ids, wer))

df = df.assign(WER = 999)
for index, row in df.iterrows():
    df.loc[index, "WER"] = ids_to_wer.get(row.id, 999)
df = df.sort_values(by='WER', ascending=True)    
examples_bool = df.WER < 0.3
print(examples_bool)
print(examples_bool.sum())
if examples_bool.sum() > 50000:
    examples_bool[50000:] = False

print(df)
print(examples_bool)

sd

#ids_to_wer_sorted = dict(sorted(ids_to_wer.items(), key = lambda x: x[1]), reverse = True)
ids_to_wer_sorted = sorted(ids_to_wer.items(), key = lambda x: x[1])
print(len(ids_to_wer_sorted))
print(ids_to_wer_sorted[100000])
print(ids_to_wer_sorted[0:10])
