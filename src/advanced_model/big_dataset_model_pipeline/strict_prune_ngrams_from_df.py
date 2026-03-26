'''
    This script was only used for ease of testing different feature spaces,
    as storing the whole document frequency vector requires around
    10-12GB and is therefore very slow to process.
'''
import pandas as pd
import numpy as np
from pathlib import Path

start_path = Path.cwd().parents[2]
data_dir = start_path / 'data' / 'big_dataset'
_FILEPATH = data_dir / 'tf_idf' / 'big_pruned_doc_freq_vector.csv'
_OUTPUT_VOCAB_PATH = data_dir / 'tf_idf' / 'big_strict_pruned_vocabulary.csv'
_OUTPUT_IDF_PATH = data_dir / 'tf_idf' / 'big_strict_pruned_doc_freq_vector.csv'

_N = 6821441 # Number of articles that vocabulary is built on
_MIN_DF = 1000 # Absolute minimal document frequency per term
_MAX_DF = 0.1 # Relative maximal document frequency per term

def prune(
    filepath=_FILEPATH,
    output_vocab_path=_OUTPUT_VOCAB_PATH,
    output_idf_path=_OUTPUT_IDF_PATH,
    N=_N,
    min_df=_MIN_DF,
    max_df=_MAX_DF
):
    print('\nLoading ngram df file...')
    df = pd.read_csv(
        _FILEPATH,
        na_filter=False,
        low_memory=False
    )
    prior_vector_size = len(df)

    print('\nPruning vocabulary based on absolute mininimum doc frequency...')
    df = df[df['doc_freq'] >= min_df]
    print(f'Absolute min_df threshold removed {prior_vector_size-len(df):,} terms to '
          f'go from {prior_vector_size:,} terms to {len(df):,} terms')

    prior_vector_size = len(df)

    print('\nPruning vocabulary based on relative maximum doc frequency...')
    df = df[df["doc_freq"] / N <= max_df]
    print(f'Relative max_df threshold removed {prior_vector_size-len(df):,} terms to '
          f'go from {prior_vector_size:,} terms to {len(df):,} terms\n')


    df.to_csv(output_idf_path, index=False, header=True)
    # df['term'].to_csv(output_vocab_path, index=False, header=False)
    print(f'Reduced terms and doc frequencies scores saved to \n'
          f'{output_vocab_path} \nand \n{output_idf_path}')

if __name__ == '__main__':
    prune()