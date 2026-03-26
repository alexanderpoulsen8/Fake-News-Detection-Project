import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from collections import Counter
import ast
import pickle
from pathlib import Path

StartPath = Path.cwd().parents[0]
DATA_DIR = StartPath / "data" / "LIAR"
MODEL_PATH = DATA_DIR / 'models' / 'logistic_model.pkl'

TOP_K_WORDS = 10000

FAKE_LABELS = {
    "fake", "conspiracy", "hate", "junksci", "unreliable",
    "bias", "satire", "political", "clickbait", "rumor", "unknown"
}
TRUE_LABELS = {"reliable"}
LIAR_FAKE_LABELS = {'false', 'pants-fire', 'barely-true', 'half-true', 'mostly-true'}
LIAR_TRUE_LABELS = {'true'}
_LIAR_cols = ['id', 'type', 'content', '4','5','6','7','8','9','10','11','12','13','14']

def load_LIAR_data(split):
    """Load train/val/test split and create binary labels."""
    print(f"Loading {split} data...")
    df = pd.read_csv(
        f"{DATA_DIR}/{split}.tsv",
        usecols=['content', 'type'],
        header=None,
        names=_LIAR_cols,
        low_memory=False,
        sep='\t'
    )
    df = df.dropna(subset=['content', 'type'])

    df['label'] = df['type'].apply(
        lambda x: 0 if x in LIAR_FAKE_LABELS else (1 if x in LIAR_TRUE_LABELS else -1)
    )
    df = df[df['label'] != -1]

    return df[['content', 'label']]


def load_data(split):
    """Load train/val/test split and create binary labels."""
    print(f"Loading {split} data...")
    df = pd.read_csv(f"{DATA_DIR}/{split}.csv", usecols=['content', 'type'])
    df = df.dropna(subset=['content', 'type'])

    df['label'] = df['type'].apply(
        lambda x: 0 if x in FAKE_LABELS else (1 if x in TRUE_LABELS else -1)
    )
    df = df[df['label'] != -1]

    return df[['content', 'label']]


def load_vocabulary():
    """Load pre-built vocabulary from pickle file.

    Note: Vocabulary must be built separately using scripts/build_vocab_from_stats.py
    """
    vocab_file = DATA_DIR / "models" / f"top_{TOP_K_WORDS}_vocab.pkl"

    if not vocab_file.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_file}\n"
            f"Please run scripts/build_vocab_from_stats.py first to build the vocabulary."
        )

    print(f"Loading vocabulary from {vocab_file}...")
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    print(f"Loaded {len(vocab)} words")
    return vocab


def create_features(df, vocab):
    """Create bag-of-words feature matrix using sparse representation."""
    print(f"Creating features for {len(df)} documents...")
    vocab_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
    X = lil_matrix((len(df), len(vocab)), dtype=np.int32)

    for i, content in enumerate(df['content']):
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i}/{len(df)} documents...")
        try:
            tokens = ast.literal_eval(content) if isinstance(content, str) else []
            for word, count in Counter(tokens).items():
                if word in vocab_to_idx:
                    X[i, vocab_to_idx[word]] = count
        except:
            continue

    print("Converting to CSR format for efficient computation...")
    return X.tocsr()


def main(LIAR=False):
    train_df = load_LIAR_data('train') if LIAR else load_data('train')
    val_df = load_LIAR_data('valid') if LIAR else load_data('val')
    test_df = load_LIAR_data('test') if LIAR else load_data('test')

    vocab = load_vocabulary()

    X_train = create_features(train_df, vocab)
    y_train = train_df['label'].values

    X_val = create_features(val_df, vocab)
    y_val = val_df['label'].values

    X_test = create_features(test_df, vocab)
    y_test = test_df['label'].values

    print("\Loading logistic regression model...")
    model = pickle.load(open(MODEL_PATH, 'rb'))

    print("\nValidation Results:")
    y_val_pred = model.predict(X_val)
    print(f"F1 Score: {f1_score(y_val, y_val_pred):.4f}")
    print(classification_report(y_val, y_val_pred, target_names=['FAKE', 'TRUE']))

    print("\nTest Results:")
    y_test_pred = model.predict(X_test)
    print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, target_names=['FAKE', 'TRUE']))


if __name__ == "__main__":
    main(LIAR=True)
