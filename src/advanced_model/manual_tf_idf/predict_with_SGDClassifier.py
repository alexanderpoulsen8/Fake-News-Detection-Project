import pandas as pd
import numpy as np
import tf_idf_vectorizer as Vec
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score, classification_report
from multiprocessing import Pool, cpu_count
from pathlib import Path
import joblib

start_path = Path.cwd().parents[2]
data_dir = start_path / 'data' / 'big_dataset'
_VAL_PATH = data_dir / 'big_preprocessed_split' / 'val.csv'
_IDF_PATH = data_dir / 'tf_idf' / 'big_pruned_idf_vector.csv'
_MODEL_PATH = data_dir / 'models' / 'SGDClassifier.joblib'
_OUTPUT_RESULTS_PATH = data_dir / 'results' / 'advanced_model_metrics.txt'

_N_WORKERS = max(cpu_count() - 1, 1)
_CHUNKSIZE = 1000



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
            pool.imap(Vec.vectorize_chunk, reader, chunksize=1),
            1
        ):
            total_rows += len(y)
            print(f'\nMean value of X: {X.mean()}')
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