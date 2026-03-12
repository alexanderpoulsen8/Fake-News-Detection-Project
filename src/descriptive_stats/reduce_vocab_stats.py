import pandas as pd
from pathlib import Path

StartPath = Path.cwd().parents[1]

_FILEPATH = StartPath / "data" / "vocabulary_stats.csv"

_OUTPUT_PATH = StartPath / "data" / "reduced_vocabulary_stats.csv"


df = pd.read_csv(_FILEPATH, low_memory=False)


df['count'] = df['count'].astype(int)
df = df.sort_values(by=['count'], ascending=False)
df.head(10000).to_csv(_OUTPUT_PATH,mode="w",header=True, index=False)