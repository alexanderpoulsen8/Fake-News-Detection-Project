import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix, vstack, save_npz
from pathlib import Path

# Prefer loading idf and other artifacts relative to this module if they exist
module_dir = Path(__file__).resolve().parent

# Project root is three levels up from this module: repo_root/src/advanced_model/... -> parents[3] == repo_root
project_root = Path(__file__).resolve().parents[3]
data_dir = project_root / 'data' / 'big_dataset'

_TRAIN_PATH = data_dir / 'big_preprocessed_split' / 'train.csv'

# Look for idf_vector.csv next to this module first, otherwise look in repo_root/data/models,
# then fall back to data/big_dataset/tf_idf
_IDF_PATH = module_dir / 'idf_vector.csv'
if not _IDF_PATH.exists():
    alt_path = project_root / 'data' / 'models' / 'idf_vector.csv'
    if alt_path.exists():
        _IDF_PATH = alt_path
    else:
        _IDF_PATH = data_dir / 'tf_idf' / 'idf_vector.csv'

_OUTPUT_MATRIX_PATH = data_dir / 'tf_idf' / 'vectorized_training_set.npz'
_OUTPUT_Y_TRUE_PATH = data_dir / 'tf_idf' / 'y_true_labels_training_set.csv'

_CHUNKSIZE = 1000

FAKE_LABELS = {
    "fake", "conspiracy", "hate", "junksci", "unreliable",
    "bias", "satire", "political", "clickbait", "rumor", "unknown"
}
REAL_LABELS = {"reliable"}
LIAR_FAKE_LABELS = {'false', 'pants-fire', 'barely-true', 'half-true', 'mostly-true'}
LIAR_REAL_LABELS = {'true'}
_LIAR_cols = ['id', 'type', 'content', '4','5','6','7','8','9','10','11','12','13','14']
idf = pd.read_csv(
    _IDF_PATH,
    usecols=['term','idf'],
    low_memory=False,
    na_filter=False
)


vocab_idx = {word: i for i, word in enumerate(idf['term'])}
idf_array = idf['idf'].values


def map_label(label):
    if pd.isna(label):
        return None
    label = str(label).strip().lower()
    if label in FAKE_LABELS:
        return 1
    if label in REAL_LABELS:
        return 0
    return None

def map_LIAR_label(label):
    if pd.isna(label):
        return None
    label = str(label).strip().lower()
    if label in LIAR_FAKE_LABELS:
        return 1
    if label in LIAR_REAL_LABELS:
        return 0
    return None

def prepare_df(df):
    text_col = "content"
    label_col = "type"
    required_cols = [text_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[text_col, label_col])
    df["label"] = df[label_col].apply(map_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df

def prepare_df_return_only_labels(df):
    text_col = "content"
    label_col = "type"
    required_cols = [text_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[text_col, label_col])
    df["label"] = df[label_col].apply(map_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df['label']

def prepare_LIAR_df(df):
    text_col = "content"
    label_col = "type"
    required_cols = [text_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[text_col, label_col])
    df["label"] = df[label_col].apply(map_LIAR_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df

def prepare_LIAR_df_return_only_labels(df):
    text_col = "content"
    label_col = "type"
    required_cols = [text_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[text_col, label_col])
    df["label"] = df[label_col].apply(map_LIAR_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df['label']


def vectorize_doc(tokens, ngram_range=(1,2), sublinear=True):
    def get_ngrams(tokens):
        min_n, max_n = ngram_range
        result = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                result.append(' '.join(tokens[i:i+n]))
        return result

    if not tokens:
        return [], []

    tokens = get_ngrams(tokens)
    counts = Counter(tokens)

    indices = []
    values = []

    for token, tf in counts.items():
        idx = vocab_idx.get(token)
        if idx is not None:
            indices.append(idx)
            values.append(tf)

    if not values:
        return [], []

    values = np.array(values, dtype=float)

    if sublinear:
        values = 1 + np.log(values)

    values *= idf_array[indices]

    norm = np.linalg.norm(values)
    if norm > 0:
        values /= norm

    return indices, values


def vectorize_chunk(chunk):
    chunk = prepare_df(chunk)
    chunk['content'] = chunk['content'].str.split(' ')

    rows, cols, data = [], [], []

    for i, doc in enumerate(chunk['content']):
        indices, values = vectorize_doc(doc)

        rows.extend([i] * len(indices))
        cols.extend(indices)
        data.extend(values)

    return (
        csr_matrix((data, (rows, cols)), shape=(len(chunk), len(vocab_idx))),
        chunk['label']
    )

def vectorize_LIAR_chunk(chunk):
    chunk = prepare_LIAR_df(chunk)
    chunk['content'] = chunk['content'].str.split(' ')

    rows, cols, data = [], [], []

    for i, doc in enumerate(chunk['content']):
        indices, values = vectorize_doc(doc)

        rows.extend([i] * len(indices))
        cols.extend(indices)
        data.extend(values)

    return (
        csr_matrix((data, (rows, cols)), shape=(len(chunk), len(vocab_idx))),
        chunk['label']
    )

def vectorize_chunk_only_articles(chunk):
    chunk = prepare_df(chunk)
    chunk['content'] = chunk['content'].str.split(' ')

    rows, cols, data = [], [], []

    for i, doc in enumerate(chunk['content']):
        indices, values = vectorize_doc(doc)

        rows.extend([i] * len(indices))
        cols.extend(indices)
        data.extend(values)

    return (
        csr_matrix((data, (rows, cols)), shape=(len(chunk), len(vocab_idx))),
        len(chunk['label'])
    )

def vectorize_articles(
    articles_filepath,
    n_workers: None | int = None,
    chunksize=_CHUNKSIZE,
    multiprocessing=True,
    LIAR=False
):
    print('\nidf and vocab index map are stored as global variables.')
    print('\nProcessing articles in training set...')
    if LIAR:
        reader = pd.read_csv(
            articles_filepath,
            usecols=['content', 'type'],
            chunksize=chunksize,
            header=None,
            names=_LIAR_cols,
            low_memory=False,
            sep='\t'
        )
    else:
        reader = pd.read_csv(
            articles_filepath,
            usecols=['content', 'type'],
            chunksize=chunksize,
            low_memory=False,
        )
    total_articles = 0
    chunks = []
    y_true = []
    if multiprocessing:
        from multiprocessing import Pool, cpu_count
        if n_workers == None:
            n_workers = max(cpu_count() - 1, 1)
        with Pool(n_workers) as pool:
            if LIAR:
                for i, (X, y) in enumerate(
                    pool.imap(vectorize_LIAR_chunk, reader, chunksize=1),
                    1
                ):
                    total_articles += len(y)
                    chunks.append(X)
                    y_true.append(y)
                    print(f'Vectorized {i:,} chunks = {total_articles:,} articles')
            else:
                for i, (X, y) in enumerate(
                    pool.imap(vectorize_chunk, reader, chunksize=1),
                    1
                ):
                    total_articles += len(y)
                    chunks.append(X)
                    y_true.append(y)
                    print(f'Vectorized {i:,} chunks = {total_articles:,} articles')
    else:
        for i, (X, y) in enumerate(map(vectorize_chunk, reader), 1):
            total_articles += len(y)
            chunks.append(X)
            y_true.append(y)
            print(f'Vectorized {i:,} chunks = {total_articles:,} articles')

    print(f'\nCombining all tf_idf matrix chunks and true y labels...')
    return (
        vstack(chunks, format='csr'),
        pd.concat(y_true, ignore_index=True)
    )

def vectorize_only_articles(
    articles_filepath,
    n_workers: None | int = None,
    chunksize=_CHUNKSIZE,
    multiprocessing=True,
):
    print('\nidf and vocab index map are stored as global variables.')
    print('\nProcessing articles in training set...')
    reader = pd.read_csv(
        articles_filepath,
        usecols=['content', 'type'],
        chunksize=chunksize,
        low_memory=False
    )
    total_articles = 0
    chunks = []
    if multiprocessing:
        from multiprocessing import Pool, cpu_count
        if n_workers == None:
            n_workers = max(cpu_count() - 1, 1)
        with Pool(n_workers) as pool:
            for i, (X, articles_in_chunk) in enumerate(pool.imap(vectorize_chunk_only_articles, reader, chunksize=1), 1):
                total_articles += articles_in_chunk
                chunks.append(X)
                print(f'Vectorized {i:,} chunks = {total_articles:,} articles')
    else:
        for i, (X, articles_in_chunk) in enumerate(map(vectorize_chunk_only_articles, reader), 1):
            total_articles += articles_in_chunk
            chunks.append(X)
            print(f'Vectorized {i:,} chunks = {total_articles:,} articles')

    print(f'\nCombining all tf_idf matrix chunks...')
    return vstack(chunks, format='csr'),

def save_y_true_vector(
    articles_filepath,
    n_workers: None | int = None,
    chunksize=_CHUNKSIZE,
    multiprocessing=True,
):
    print('\nidf and vocab index map are stored as global variables.')
    print('\nProcessing articles in training set...')
    reader = pd.read_csv(
        articles_filepath,
        usecols=['content', 'type'],
        chunksize=chunksize,
        low_memory=False
    )
    total_articles = 0
    y_true = []
    if multiprocessing:
        from multiprocessing import Pool, cpu_count
        if n_workers == None:
            n_workers = max(cpu_count() - 1, 1)
        with Pool(n_workers) as pool:
            for i, y in enumerate(pool.imap(prepare_df_return_only_labels, reader, chunksize=1), 1):
                total_articles += len(y)
                y_true.append(y)
                print(f'Processed {i:,} chunks = {total_articles:,} articles')
    else:
        for i, y in enumerate(map(prepare_df_return_only_labels, reader), 1):
            total_articles += len(y)
            y_true.append(y)
            print(f'Processed {i:,} chunks = {total_articles:,} articles')

    print(f'\nCombining all true y labels...')
    return pd.concat(y_true, ignore_index=True)