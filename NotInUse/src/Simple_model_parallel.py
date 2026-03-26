import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from collections import Counter
import ast
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path

StartPath = Path.cwd()
DATA_DIR = StartPath / "data"

TOP_K_WORDS = 10000

FAKE_LABELS = {'unreliable', 'hate', 'junksci', 'fake', 'satire', 'conspiracy', 'bias', 'political', 'state', 'clickbait'}
TRUE_LABELS = {'reliable'}


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
    vocab_file = DATA_DIR / f"top_{TOP_K_WORDS}_vocab.pkl"
    
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


def process_chunk_features(args):
    """Process a chunk of documents in parallel."""
    chunk_df, vocab, chunk_idx = args
    vocab_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
    X_chunk = lil_matrix((len(chunk_df), len(vocab)), dtype=np.int32)
    
    for i, content in enumerate(chunk_df['content'].values):
        try:
            tokens = ast.literal_eval(content) if isinstance(content, str) else []
            for word, count in Counter(tokens).items():
                if word in vocab_to_idx:
                    X_chunk[i, vocab_to_idx[word]] = count
        except:
            continue
    
    print(f"  Chunk {chunk_idx} completed")
    return X_chunk.tocsr()


def create_features_parallel(df, vocab, n_workers=None):
    """Create bag-of-words feature matrix using parallel processing."""
    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)
    
    print(f"Creating features for {len(df)} documents using {n_workers} workers...")
    
    # Split dataframe into chunks
    chunk_size = len(df) // n_workers
    chunks = []
    for i in range(n_workers):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_workers - 1 else len(df)
        chunk = df.iloc[start_idx:end_idx].reset_index(drop=True)
        chunks.append((chunk, vocab, i + 1))
    
    # Process chunks in parallel
    with Pool(n_workers) as pool:
        results = pool.map(process_chunk_features, chunks)
    
    # Combine results
    print("Combining chunks...")
    X = vstack(results)
    
    return X


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


def main(use_parallel=True, n_workers=None):
    train_df = load_data('train')
    val_df = load_data('val')
    test_df = load_data('test')
    
    vocab = load_vocabulary()
    
    # Choose parallel or sequential feature creation
    feature_func = create_features_parallel if use_parallel else create_features
    
    if use_parallel:
        X_train = feature_func(train_df, vocab, n_workers)
        X_val = feature_func(val_df, vocab, n_workers)
        X_test = feature_func(test_df, vocab, n_workers)
    else:
        X_train = feature_func(train_df, vocab)
        X_val = feature_func(val_df, vocab)
        X_test = feature_func(test_df, vocab)
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    print("\nTraining logistic regression...")
    model = LogisticRegression(
        C=0.5,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=5000,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("\nValidation Results:")
    y_val_pred = model.predict(X_val)
    print(f"F1 Score: {f1_score(y_val, y_val_pred):.4f}")
    print(classification_report(y_val, y_val_pred, target_names=['FAKE', 'TRUE']))
    
    print("\nTest Results:")
    y_test_pred = model.predict(X_test)
    print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, target_names=['FAKE', 'TRUE']))
    
    with open(DATA_DIR / "logistic_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    print("\nModel saved!")


if __name__ == "__main__":
    # Set use_parallel=True to use all CPU cores
    # Set n_workers to specific number (e.g., 8) to limit cores used
    main(use_parallel=True, n_workers=None)
