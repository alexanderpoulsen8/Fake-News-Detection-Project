import numpy as np
import pandas as pd
from pathlib import Path

StartPath = Path.cwd().parents[0]
_VOCAB_PATH = StartPath / "data" / "vocabulary_reliable_frequency.csv"
_PROCESSED_PATH = StartPath / "data" / "preprocessed_dataset.csv"
_OUTPUT_PATH = StartPath / "data" / "predictions_by_vocab.csv"
_CHUNKSIZE = 10000
ALPHA = 1  # Laplace smoothing

# --- Load vocabulary ---
vocab_df = pd.read_csv(_VOCAB_PATH, index_col='word')
vocab_df = vocab_df[vocab_df['count'] >= 5]  # remove rare words
vocab_size = len(vocab_df)

# log-odds
log_odds = np.log((vocab_df['reliable_count'] + ALPHA) / (vocab_df['fake_count'] + ALPHA))
log_odds_dict = log_odds.to_dict()

# --- Class priors ---
types = pd.read_csv(_PROCESSED_PATH, usecols=['type'], low_memory=False)['type']
types = types[types.notna()]
types_bool = types.map(lambda t: t in {'reliable'})
log_prior_true = np.log(types_bool.mean())
log_prior_fake = np.log(1 - types_bool.mean())

# --- Initialize output ---
pd.DataFrame(columns=['prediction','target']).to_csv(_OUTPUT_PATH, index=False)

# --- Prediction function ---
def predict_tokens(tokens):
    # sum of log-odds for tokens in vocab
    token_scores = [log_odds_dict.get(t, 0) for t in tokens]
    score = sum(token_scores) / max(len(tokens), 1)  # normalize by length
    score += log_prior_true - log_prior_fake      # add priors
    return score > 0

# --- Process in chunks ---
with pd.read_csv(_PROCESSED_PATH, chunksize=_CHUNKSIZE, usecols=['content','type']) as reader:
    for i, chunk in enumerate(reader, 1):
        print(f"Processing chunk {i} ({i*len(chunk)} rows)")
        chunk = chunk[chunk['type'].isin({'reliable','fake','conspiracy','hate','junksci','clickbait','unreliable','bias','satire','political'})].dropna()
        chunk['target'] = chunk['type'].map(lambda t: t in {'reliable'})

        # parse tokens
        tokens_series = chunk['content'].str.lstrip("['").str.rstrip("']").str.split(r"', '")

        # vectorized scoring
        chunk['prediction'] = tokens_series.apply(predict_tokens)

        # save
        chunk[['prediction','target']].to_csv(_OUTPUT_PATH, mode='a', header=False, index=False)