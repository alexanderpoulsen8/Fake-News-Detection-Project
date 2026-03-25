import numpy as np
from scipy.sparse import csr_matrix, load_npz
from sklearn.svm import LinearSVC
from pathlib import Path
import joblib

start_path = Path.cwd().parents[2]
data_dir = start_path / 'data' / 'big_dataset'
_TRAIN_MATRIX_PATH = data_dir / 'tf_idf' / 'vectorized_training_set.npz'
_TRAIN_Y_TRUE_PATH = data_dir / 'tf_idf' / 'y_true_labels_training_set.csv'
_OUTPUT_PATH = data_dir / 'models' / 'tf_idf_linear_svc.joblib'

C_VALUE = 0.01

def main(
    training_matrix_path=_TRAIN_MATRIX_PATH,
    training_y_true_path=_TRAIN_Y_TRUE_PATH,
    output_path=_OUTPUT_PATH,
    c_value=C_VALUE
):
    print('\nLoading training tf-idf matrix')
    X = load_npz(training_matrix_path)
    print('\nLoading true y labels')
    y = np.genfromtxt(training_y_true_path)
    print('\nBuilding and fitting LinearSVC...')
    clf = LinearSVC(
        C=c_value,
        random_state=42,
        max_iter=10000,
        verbose=1,
        class_weight='balanced',
        dual=False
    )
    clf.fit(X, y)

    print(f'\nSaving LinearSVC model to {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"\nSaved model to {output_path}")
    print('\nFinished')
    print("Shape:", X.shape)
    print("Non-zeros:", X.nnz)
    print("Density:", X.nnz / (X.shape[0] * X.shape[1]))

    row_sums = np.array(X.sum(axis=1)).flatten()
    print("Sample row sums:", row_sums[:10])

    print("Min:", X.min())
    print("Max:", X.max())
    print("Mean:", X.mean())


if __name__ == '__main__':
    main()