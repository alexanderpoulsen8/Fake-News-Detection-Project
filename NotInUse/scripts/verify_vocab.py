import pickle
from pathlib import Path

StartPath = Path.cwd()
DATA_DIR = StartPath / "data"

vocab_file = DATA_DIR / "top_10000_vocab.pkl"

print(f"Loading {vocab_file}...")
with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)

print(f"Type: {type(vocab)}")
print(f"Size: {len(vocab)}")
print(f"\nTop 20 words (sorted):")
sorted_vocab = sorted(list(vocab))[:20]
for word in sorted_vocab:
    print(f"  {word}")

print(f"\nChecking for expected high-frequency words:")
expected_words = ['<num>', 'not', 'said', 'one', 'new', 'us', 'would', 'state', 'year', 'peopl']
for word in expected_words:
    status = "FOUND" if word in vocab else "MISSING"
    print(f"  {word}: {status}")
