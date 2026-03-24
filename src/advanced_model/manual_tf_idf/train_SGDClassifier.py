import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import SGDClassifier
from multiprocessing import Pool, cpu_count
from pathlib import Path
import joblib

start_path = Path.cwd().parents[2]
data_dir = start_path / 'data' / 'big_dataset'
_TRAIN_PATH = data_dir / 'big_preprocessed_split' / 'train.csv'
_IDF_PATH = data_dir / 'tf_idf' / 'big_pruned_idf_vector.csv'
_OUTPUT_MODEL_PATH = data_dir / 'models' / 'tf_idf_SGDClasssifier.joblib'



_N_WORKERS = max(cpu_count() - 1, 1)
_CHUNKSIZE = 100000

FAKE_LABELS = {
    "fake", "conspiracy", "hate", "junksci", "unreliable", "bias", "satire", "political", "clickbait"
}
REAL_LABELS = {"reliable"}
idf = pd.read_csv(
    _IDF_PATH,
    usecols=['word','idf'],
    low_memory=False,
    na_filter=False
)
vocab_idx = {word: i for i, word in enumerate(idf['word'])}

def map_label(label):
    if pd.isna(label):
        return None
    label = str(label).strip().lower()
    if label in FAKE_LABELS:
        return 1
    if label in REAL_LABELS:
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

def vectorize_doc(tokens, vocab_idx, idf, ngram_range=(1,1), sublinear=True):
    def get_ngrams(tokens):
        min_n, max_n = ngram_range
        length = len(tokens)
        result = set()
        for n in range(min_n, max_n + 1):
            for i in range(length - n + 1):
                result.add(' '.join(tokens[i:i+n]))
        return result

    if not isinstance(tokens, str) or not tokens:
        return [], []

    if ngram_range[1] > 1:
        tokens = get_ngrams(tokens)

    indices = []
    values = []
    vocab_get = vocab_idx.get

    for token in tokens:
        idx = vocab_get
        if idx is not None:
            indices.append(idx)
            values.append(1.0)

    if not values:
        return [], []

    values = np.array(values, dtype=float)

    if sublinear:
        values = 1 + np.log(values)

    # apply idf
    values *= idf[indices]

    # L2
    norm = np.linalg.norm(values)
    if norm > 0:
        values /= norm

    return indices, values


def vectorize_chunk(chunk):
    chunk = prepare_df(chunk)
    chunk['content'] = chunk['content'].str.split(' ')

    rows, cols, data = [], [], []

    for i, doc in enumerate(chunk['content']):
        indices, values = vectorize_doc(doc, vocab_idx, idf['idf'])

        rows.extend([i] * len(indices))
        cols.extend(indices)
        data.extend(values)

    return (
        csr_matrix((data, (rows, cols)), shape=(len(chunk), len(vocab_idx))),
        chunk['label'].astype(int).to_numpy()
    )


def train_SGDClassifier(
    train_path=_TRAIN_PATH,
    idf_path=_IDF_PATH,
    chunksize=_CHUNKSIZE,
    ngram_range=(1,1),
    sublinear=True
):
    print('idf and vocab index map are stored as global variables. '
          '\nPreparing model, and reader...')
    clf = model = SGDClassifier(
        loss="hinge",        # same objective as LinearSVC
        penalty="l2",
        alpha=1e-4,          # tune this!
        max_iter=5,          # important for partial_fit loop
        warm_start=True,
        learning_rate='optimal',
        tol=None
    )
    classes = np.array([0,1])
    print('\nProcessing articles in training set...')
    reader = pd.read_csv(
        _TRAIN_PATH,
        usecols=['content', 'type'],
        chunksize=chunksize,
        low_memory=False
    )
    with Pool(_N_WORKERS) as pool:
        total_rows = 0
        for i, (X, y) in enumerate(
            pool.imap(vectorize_chunk, reader, chunksize=1),
            1
        ):
            total_rows += len(y)
            clf.partial_fit(X, y, classes=classes)
            print(f'Vectorized and fitted {i:,} chunks = {total_rows:,} rows')
    print(f'Saving model to file \n{_OUTPUT_MODEL_PATH}')
    joblib.dump(clf, _OUTPUT_MODEL_PATH)
    print('Finished')










if __name__ == '__main__':
    train_SGDClassifier()