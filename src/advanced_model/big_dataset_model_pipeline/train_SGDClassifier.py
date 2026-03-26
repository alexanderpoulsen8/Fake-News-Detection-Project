import pandas as pd
import numpy as np
import tf_idf_vectorizer as Vec
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
from multiprocessing import Pool, cpu_count
from pathlib import Path
import joblib

start_path = Path.cwd().parents[2]
data_dir = start_path / 'data' / 'big_dataset'
_TRAIN_PATH = data_dir / 'big_preprocessed_split' / 'train.csv'
_OUTPUT_MODEL_PATH = data_dir / 'models' / 'SGDClassifier.joblib'


_BUFFER_SIZE = 800_000
_CHUNKSIZE = 1_000
_BATCH_SIZE = 512
_EPOCHS = 5
_N_WORKERS = max(cpu_count() - 2, 1)

def get_class_weights(
    train_path=_TRAIN_PATH,
    chunksize=_BUFFER_SIZE,
    n_workers=_N_WORKERS
):
    class_counts = np.zeros(2, dtype=np.int64)
    reader = pd.read_csv(
        train_path,
        usecols=['content', 'type'],
        chunksize=chunksize,
        low_memory=False
    )

    with Pool(n_workers) as pool:
        for chunk_labels in pool.imap(
            Vec.prepare_df_return_only_labels,
            reader,
            chunksize=1
        ):
            counts = np.bincount(chunk_labels, minlength=2)
            class_counts += counts

    total = class_counts.sum()
    n_classes = len(class_counts)
    class_weight_dict = {
        i: total / (n_classes * count) for i, count in enumerate(class_counts)
    }

    print(f'Class counts: {class_counts}, Class weights: {class_weight_dict}')
    return class_weight_dict

def train_SGDClassifier(
    train_path=_TRAIN_PATH,
    output_path=_OUTPUT_MODEL_PATH,
    buffer_size=_BUFFER_SIZE,
    chunksize=_CHUNKSIZE,
    batch_size=_BATCH_SIZE,
    n_epochs=_EPOCHS,
    n_workers=_N_WORKERS
):
    print('Preparing class weights...\n')
    class_weight_dict = get_class_weights(
        train_path,
        chunksize=buffer_size,
        n_workers=n_workers
    )

    print('Preparing SGDClassifier and reader...')

    clf = SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=1e-7,
        learning_rate='constant',
        eta0=0.001,
        fit_intercept=True,
        max_iter=1,
        tol=None,
        shuffle=False,
        average=True,
        class_weight=class_weight_dict,
        random_state=42
    )

    classes = np.array([0,1])

    with Pool(n_workers) as pool:
        for epoch in range(n_epochs):
            print(f'\nEpoch {epoch + 1}/{n_epochs}')

            reader = pd.read_csv(
                train_path,
                usecols=['content', 'type'],
                chunksize=chunksize,
                low_memory=False
            )

            buffer_X, buffer_y = [], []
            total_rows = 0

            for i, (X_chunk, y_chunk) in enumerate(
                pool.imap(Vec.vectorize_chunk, reader, chunksize=1),
                1
            ):
                if X_chunk.shape[0] == 0:
                    continue

                buffer_X.append(X_chunk)
                buffer_y.append(y_chunk)
                total_rows += len(y_chunk)

                # Once buffer reaches chunk size, shuffle and train
                if sum(b.shape[0] for b in buffer_X) >= buffer_size:
                    # Stack sparse matrices and labels
                    X_buf = vstack(buffer_X)
                    y_buf = np.concatenate(buffer_y)

                    # Shuffle buffer
                    idx = np.random.permutation(len(y_buf))
                    X_buf = X_buf[idx]
                    y_buf = y_buf[idx]

                    # Train in mini-batches
                    for start in range(0, len(y_buf), batch_size):
                        end = start + batch_size
                        clf.partial_fit(
                            X_buf[start:end],
                            y_buf[start:end],
                            classes=classes
                        )
                    # Clear buffer
                    buffer_X, buffer_y = [], []
                    print(f'\nEpoch {epoch + 1}, Fitted {i:,} chunks = {total_rows:,} rows\n')
                print(f'Epoch {epoch + 1}, vectorized {i:,} chunks = {total_rows:,} rows')
            # Train remaining buffer after epoch
            if buffer_X:
                X_buf = vstack(buffer_X)
                y_buf = np.concatenate(buffer_y)

                idx = np.random.permutation(len(y_buf))
                X_buf = X_buf[idx]
                y_buf = y_buf[idx]

                for start in range(0, len(y_buf), batch_size):
                    end = start + batch_size
                    clf.partial_fit(
                        X_buf[start:end],
                        y_buf[start:end],
                        classes=classes
                    )
            # Save progress
            joblib.dump(clf, output_path)
            print(f'Epoch {epoch + 1} completed, total rows processed: {total_rows}')

    print(f'\nSaving trained model to: {output_path}')
    joblib.dump(clf, output_path)
    print('Finished training.')


if __name__ == '__main__':
    train_SGDClassifier()