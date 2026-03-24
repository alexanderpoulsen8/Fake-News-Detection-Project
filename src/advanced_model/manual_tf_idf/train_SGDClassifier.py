import pandas as pd
import numpy as np
import tf_idf_vectorizer as Vec
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import SGDClassifier
from multiprocessing import Pool, cpu_count
from pathlib import Path
import joblib

start_path = Path.cwd().parents[2]
data_dir = start_path / 'data' / 'big_dataset'
_TRAIN_PATH = data_dir / 'big_preprocessed_split' / 'train.csv'

_OUTPUT_MODEL_PATH = data_dir / 'models' / 'SGDClasssifier.joblib'

_N_WORKERS = max(cpu_count() - 1, 1)
_CHUNKSIZE = 10000

def train_SGDClassifier(
    train_path=_TRAIN_PATH,
    chunksize=_CHUNKSIZE,
    ngram_range=(1,1),
    sublinear=True
):
    print('idf and vocab index map are stored as global variables. '
          '\nPreparing model, and reader...')
    clf = SGDClassifier(
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
            pool.imap(Vec.vectorize_chunk, reader, chunksize=1),
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