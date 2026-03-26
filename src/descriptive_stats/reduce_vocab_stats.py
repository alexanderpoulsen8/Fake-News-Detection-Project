import pandas as pd
from pathlib import Path

Top_k_words = 10000

StartPath = Path.cwd().parents[1]

_FILEPATH = StartPath / "data" / "vocabulary_stats.csv"

_OUTPUT_PATH = StartPath / "data" / "reduced_vocabulary_stats.csv"


df = pd.read_csv(_FILEPATH, low_memory=False)


df['count'] = df['count'].astype(int)
df = df.sort_values(by=['count'], ascending=False)
df.head(Top_k_words).to_csv(_OUTPUT_PATH,mode="w",header=True, index=False)