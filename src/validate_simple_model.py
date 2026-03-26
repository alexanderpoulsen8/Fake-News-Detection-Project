import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.metrics import classification_report, f1_score
from collections import Counter
import pickle
from pathlib import Path
import re

StartPath = Path.cwd()
DATA_DIR = StartPath / "data"
MODEL_PATH = StartPath / "data" / "models" / "logistic_model.pkl"
VOCAB_PATH = StartPath / "data" / "models" / "top_10000_vocab.pkl"
EVAL_PATH = StartPath / "data" / "val.csv"

TOP_K_WORDS = 10000

LIAR_FAKE_LABELS = {'false', 'pants-fire', 'barely-true', 'half-true', 'mostly-true'}
LIAR_TRUE_LABELS = {'true'}

_LIAR_cols = ['id', 'type', 'content', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']


def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"\b\w+\b", text.lower())


def load_liar_combined_data():
    print(f"Loading evaluation data from {EVAL_PATH}...")
    df = pd.read_csv(
        EVAL_PATH,
        sep='\t',
        header=None,
        names=_LIAR_cols,
        low_memory=False
    )

    df = df.dropna(subset=['content', 'type'])

    df['label'] = df['type'].apply(
        lambda x: 0 if x in LIAR_FAKE_LABELS else (1 if x in LIAR_TRUE_LABELS else -1)
    )

    df = df[df['label'] != -1]

    print(f"Loaded {len(df)} labeled examples")
    print(df['label'].value_counts())

    return df[['content', 'label']]


def load_vocabulary():
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_PATH}")

    print(f"Loading vocabulary from {VOCAB_PATH}...")
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    print(f"Loaded {len(vocab)} words")
    return vocab


def create_features(df, vocab):
    print(f"Creating features for {len(df)} documents...")
    vocab_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
    X = lil_matrix((len(df), len(vocab)), dtype=np.int32)

    for i, content in enumerate(df['content']):
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i}/{len(df)} documents...")

        tokens = simple_tokenize(content)
        for word, count in Counter(tokens).items():
            if word in vocab_to_idx:
                X[i, vocab_to_idx[word]] = count

    print("Converting to CSR format for efficient computation...")
    return X.tocsr()


def main():
    eval_df = load_liar_combined_data()
    vocab = load_vocabulary()

    X_eval = create_features(eval_df, vocab)
    y_eval = eval_df['label'].values

    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    print("\nCombined Evaluation Results:")
    y_pred = model.predict(X_eval)
    print(f"F1 Score: {f1_score(y_eval, y_pred):.4f}")
    print(classification_report(y_eval, y_pred, target_names=['FAKE', 'TRUE']))


if __name__ == "__main__":
    main()