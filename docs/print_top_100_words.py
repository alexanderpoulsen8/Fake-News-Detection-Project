import pandas as pd
from pathlib import Path


StartPath = Path.cwd().parents[0]
data_dir = StartPath / 'data' / 'medium_dataset'
preprocessed_reduced_vocab_stats_path = data_dir / "preprocessed_reduced_vocabulary_stats.csv"

df = pd.read_csv(preprocessed_reduced_vocab_stats_path, names=['word','count'], header=0).head(100)
df['count'] = df['count'].astype(int)
for word, count in zip(df['word'], df['count']):
    print(f'{word}: {count:,}')