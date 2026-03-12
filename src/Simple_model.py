import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from collections import Counter
import ast
import pickle
from pathlib import Path

StartPath = Path.cwd()
_DATA_DIR = StartPath / "data"

TOP_K_WORDS = 10000

FAKE_LABELS = {'unreliable', 'hate', 'junksci', 'fake', 'satire', 'conspiracy', 'bias'}
TRUE_LABELS = {'reliable', 'political', 'state', 'clickbait'}


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


def build_vocabulary(train_df, load_if_exists=True):
    """Extract top K most frequent words from training data."""
    vocab_file = f"{DATA_DIR}/top_{TOP_K_WORDS}_vocab.pkl"
    
    if load_if_exists and pd.io.common.file_exists(vocab_file):
        print(f"Loading pre-computed vocabulary from {vocab_file}...")
        with open(vocab_file, 'rb') as f:
            return pickle.load(f), None
    
    print(f"Building vocabulary (top {TOP_K_WORDS} words)...")
    word_counter = Counter()
    
    for content in train_df['content']:
        try:
            tokens = ast.literal_eval(content) if isinstance(content, str) else []
            word_counter.update(tokens)
        except:
            continue
    
    vocab = set([word for word, _ in word_counter.most_common(TOP_K_WORDS)])
    
    print(f"Saving vocabulary to {vocab_file}...")
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    
    return vocab, word_counter


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


def main():
    train_df = load_data('train')
    val_df = load_data('val')
    test_df = load_data('test')
    
    vocab, word_counter = build_vocabulary(train_df)
    
    X_train = create_features(train_df, vocab)
    y_train = train_df['label'].values
    
    X_val = create_features(val_df, vocab)
    y_val = val_df['label'].values
    
    X_test = create_features(test_df, vocab)
    y_test = test_df['label'].values
    
    print("\nTraining logistic regression...")
    model = LogisticRegression(max_iter=5000, random_state=42, verbose=1)
    model.fit(X_train, y_train)
    
    print("\nValidation Results:")
    y_val_pred = model.predict(X_val)
    print(f"F1 Score: {f1_score(y_val, y_val_pred):.4f}")
    print(classification_report(y_val, y_val_pred, target_names=['FAKE', 'TRUE']))
    
    print("\nTest Results:")
    y_test_pred = model.predict(X_test)
    print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, target_names=['FAKE', 'TRUE']))
    
    with open(f"{DATA_DIR}/logistic_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(f"{DATA_DIR}/vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)
    
    print("\nModel saved!")


if __name__ == "__main__":
    main()
