import pandas as pd
from numpy import log
from pathlib import Path

start_path = Path.cwd().parents[2]
_DATA_DIR = start_path / 'data' / 'big_dataset'
_FILEPATH = _DATA_DIR / 'tf_idf' / 'big_strict_pruned_doc_freq_vector.csv'
_IDF_VECTOR_PATH = _DATA_DIR / 'tf_idf' / 'idf_vector.csv'

_NROWS = 6821441


def build_idf_vector(filepath=_FILEPATH, N=_NROWS):
    print('Loading doc frequency DataFrame...\n')
    df = pd.read_csv(
        filepath,
        low_memory=False,
        na_filter=False
    )
    print('\nConverting to idf dictionary...')
    df['idf'] = log((N+1)/(df['doc_freq']+1)) + 1
    print(f'Adding idf dictionary to csv file {_IDF_VECTOR_PATH}')
    df[['term','idf']].to_csv(_IDF_VECTOR_PATH, index=False)
    print('Finished')

if __name__ == '__main__':
    build_idf_vector()