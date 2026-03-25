import tf_idf_vectorizer as Vec
from scipy.sparse import save_npz
from pathlib import Path

start_path = Path.cwd().parents[2]
data_dir = start_path / 'data' / 'big_dataset'
_TRAIN_PATH = data_dir / 'big_preprocessed_split' / 'train.csv'
_OUTPUT_MATRIX_PATH = data_dir / 'tf_idf' / 'vectorized_training_set.npz'
_OUTPUT_Y_TRUE_PATH = data_dir / 'tf_idf' / 'y_true_labels_training_set.csv'

def vectorize_training_set(
    train_path=_TRAIN_PATH,
    output_matrix_path=_OUTPUT_MATRIX_PATH,
    output_y_true_path=_OUTPUT_Y_TRUE_PATH,
):
    print('\nVectorizing training set...\n')
    X = Vec.vectorize_only_articles(train_path, chunksize=1_000, multiprocessing=True)

    print(f'\nSaving training matrix to \n{output_matrix_path}\n...')
    save_npz(output_matrix_path, X)

    print('\nCollecting true y labels from training set...')
    y = Vec.save_y_true_vector(train_path, chunksize=1000, multiprocessing=True)

    print(f'\nSaving y labels to \n{output_y_true_path}')
    y.to_csv(output_y_true_path, index=False, header=False)

    print('\nFinished')

if __name__ == '__main__':
    vectorize_training_set()