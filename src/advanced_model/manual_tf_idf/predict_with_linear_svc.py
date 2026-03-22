import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score, classification_report
from multiprocessing import Pool, cpu_count
from pathlib import Path
import joblib

start_path = Path.cwd().parents[2]
data_dir = start_path / 'data' / 'big_dataset'
_VAL_PATH = data_dir / 'big_preprocessed_split' / 'val.csv'
_IDF_PATH = data_dir / 'tf_idf' / 'big_pruned_idf_vector.csv'
_MODEL_PATH = data_dir / 'models' / 'linear_svm_model.joblib'
_OUTPUT_RESULTS_PATH = data_dir / 'results' / 'advanced_model_metrics.txt'

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
        chunk['label']
    )

def predict():
    clf = joblib.load(_MODEL_PATH)
    print('\nProcessing articles in validation set...')
    reader = pd.read_csv(
        _VAL_PATH,
        usecols=['content', 'type'],
        chunksize=_CHUNKSIZE,
        low_memory=False
    )
    with Pool(_N_WORKERS) as pool:
        total_rows = 0
        y_full = []
        y_pred_full = []
        for i, (X, y) in enumerate(
            pool.imap(vectorize_chunk, reader, chunksize=1),
            1
        ):
            total_rows += len(y)
            print(X.sum(axis=1)[:10])
            y_pred = clf.predict(X)

            y_full.extend(y)
            y_pred_full.extend(y_pred)

            print(f'Vectorized and predicted {i:,} chunks = {total_rows:,} rows')

    print('\nFinished predicting all articles')
    y_full = np.array(y_full, dtype=int)
    y_pred_full = np.array(y_pred_full, dtype=int)

    print('Computing F1-score')
    print(y_full.shape, y_pred_full.shape)
    assert y_full.shape == y_pred_full.shape
    print(np.unique(y_full))
    print(np.unique(y_pred_full))
    print(y_full[:10])
    print(y_pred_full[:10])
    f1 = f1_score(y_full, y_pred_full)
    report = classification_report(y_full, y_pred_full)

    print(f"\nValidation F1: {f1:.4f}")
    print(report)

    _OUTPUT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Validation F1: {f1:.4f}\n\n")
        f.write(f"Val path: {_VAL_PATH}\n")
        f.write(f"Chunk size: {_CHUNKSIZE}\n")
        f.write("TF-IDF config:\n")
        f.write("  sublinear_tf=True\n")
        f.write("  smooth_idf=True\n")
        f.write("  stop_words=None\n")
        f.write(report)
    print(f"Saved results to {_OUTPUT_RESULTS_PATH}")




if __name__ == '__main__':
    predict()