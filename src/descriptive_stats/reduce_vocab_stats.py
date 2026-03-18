import pandas as pd
from pathlib import Path

StartPath = Path.cwd().parents[1]

_FILEPATH = StartPath / "data" / "vocabulary_stats.csv"

_OUTPUT_PATH = StartPath / "data" / "reduced_vocabulary_stats.csv"


df = pd.read_csv(_FILEPATH, low_memory=False)
df['count'] = df['count'].astype(int)
df = df.sort_values(by=['count'], ascending=False).head(30000)


ALL_LABELS = {'unreliable', 'hate', 'junksci', 'fake', 'satire', 'conspiracy', 'bias', 'reliable', 'political', 'state', 'clickbait'}
df['types'] = df['types'].str.lstrip("{'").str.rstrip("}").str.split(r", '")
df['types'] = df['types'].apply(lambda row: {key: int(val) for key, val in [label.split("': ") for label in row]})


df['count'] = df['types'].apply(lambda r: sum([r.get(l, 0) for l in ALL_LABELS]))
# df = df.mask(df['count'] < 1).dropna()
df = df.sort_values(by=['count'], ascending=False)
df.head(50000).to_csv(_OUTPUT_PATH,mode="w",header=True, index=False)