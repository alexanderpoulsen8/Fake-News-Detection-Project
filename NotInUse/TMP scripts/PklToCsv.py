import pickle
import pandas as pd
# Script brugt til at conventere pkl filerne så vi kunne visualisere dem og kigge manuelt efter fejl
pkl_file = r"C:\Users\jespe\GDS eksamen\Fake-News-Detection-Project\data\top_10000_vocab.pkl"
csv_file = r"C:\Users\jespe\GDS eksamen\Fake-News-Detection-Project\data\top_10000_vocab.csv"

with open(pkl_file, 'rb') as f:
    vocab = pickle.load(f)

if isinstance(vocab, set):
    vocab_list = sorted(list(vocab))
    df = pd.DataFrame({'word': vocab_list})
elif isinstance(vocab, list):
    df = pd.DataFrame({'word': vocab})
elif isinstance(vocab, dict):
    df = pd.DataFrame.from_dict(vocab, orient='index')
else:
    df = pd.DataFrame(vocab)

df.to_csv(csv_file, index=False)
print(f"Converted {pkl_file} to {csv_file}")
print(f"Total words: {len(df)}")
print(f"\nFirst 10 words:")
print(df.head(10))
