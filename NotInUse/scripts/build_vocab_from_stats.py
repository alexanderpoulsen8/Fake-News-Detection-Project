import pandas as pd
import pickle
from pathlib import Path

StartPath = Path.cwd()
DATA_DIR = StartPath / "data"

VOCAB_STATS_FILE = DATA_DIR / "reduced_vocabulary_stats.csv"
OUTPUT_FILE = DATA_DIR / "top_10000_vocab.pkl"
TOP_K = 10000

print(f"Reading {VOCAB_STATS_FILE}...")
df = pd.read_csv(VOCAB_STATS_FILE, index_col=0)

print(f"Total words in vocabulary_stats.csv: {len(df):,}")

df['count'] = pd.to_numeric(df['count'], errors='coerce')
df = df.dropna(subset=['count'])
df = df[df['count'] > 0]

print(f"Words with valid counts: {len(df):,}")

df_sorted = df.sort_values('count', ascending=False)

top_words = df_sorted.head(TOP_K).index.tolist()
top_words_clean = [w for w in top_words if isinstance(w, str) and w.strip()]
vocab_set = set(top_words_clean)

print(f"\nTop {TOP_K} words by frequency:")
print(f"Most frequent: {df_sorted.head(10)[['count']].to_string()}")

print(f"\nSaving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(vocab_set, f)

print(f"Saved {len(vocab_set)} words to {OUTPUT_FILE}")
print(f"\nSample words from vocab: {list(vocab_set)[:10]}")
