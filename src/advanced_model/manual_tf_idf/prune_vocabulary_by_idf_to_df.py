import pandas as pd
import numpy as np
from pathlib import Path

start_path = Path.cwd().parents[2]
data_dir = start_path / 'data' / 'big_dataset'
_FILEPATH = data_dir / 'tf_idf' / 'big_idf_vector.csv'
_OUTPUT_VOCAB_PATH = data_dir / 'tf_idf' / 'big_pruned_vocabulary.csv'
_OUTPUT_IDF_PATH = data_dir / 'tf_idf' / 'big_pruned_idf_vector.csv'

_N = 6821441
_MIN_DF = 5
_MAX_DF = 0.9

def prune(
    filepath=_FILEPATH,
    output_vocab_path=_OUTPUT_VOCAB_PATH,
    output_idf_path=_OUTPUT_IDF_PATH,
    N=_N,
    min_df=_MIN_DF,
    max_df=_MAX_DF
):
    print('\nLoading idf scores file...')
    df = pd.read_csv(
        _FILEPATH,
        na_filter=False,
        low_memory=False
    )
    prior_vector_size = len(df)
    print('\nComputing document frequency score from idf scores')
    df['doc_freq'] = ((N+1) / np.exp(df['idf'] - 1) - 1).round().astype(int)
    print('\nPruning vocabulary based on min and max doc frequence')
    filtered = df[
        (df["doc_freq"] >= min_df) &
        (df["doc_freq"] / N <= max_df)
    ]
    print(f'Removed {prior_vector_size-len(filtered):,} terms to '
          f'go from {prior_vector_size:,} terms to {len(filtered):,} terms')
    filtered.to_csv(output_idf_path, index=False, header=True)
    filtered['word'].to_csv(output_vocab_path, index=False, header=False)
    print(f'Reduced terms and idf scores saved to \n{output_vocab_path} \nand \n{output_idf_path}')

if __name__ == '__main__':
    prune()