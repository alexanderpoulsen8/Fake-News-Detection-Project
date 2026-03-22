import pandas as pd
from multiprocessing import Pool, cpu_count
from pathlib import Path

start_path = Path.cwd().parents[2]
_DATA_DIR = start_path / 'data' / 'big_dataset'
_TRAIN_PATH = _DATA_DIR / 'big_preprocessed_split' / "train.csv"
_OUTPUT_PATH = _DATA_DIR / 'tf_idf' / 'big_vocabulary.csv'

_N_WORKERS = max(cpu_count() - 2, 1)
_CHUNKSIZE = 5000

def vocabulary_of_content(df):
    df['content'] = df['content'].str.split(' ')
    unigrams = df['content'].explode(ignore_index=True).unique()
    bigrams = df['content'].apply(
        lambda doc: [' '.join([doc[i], doc[i+1]]) for i in range(len(doc)-1)]
    ).explode().unique()
    return pd.concat((unigrams, bigrams), ignore_index=True)

def build_vocab(filepath=_TRAIN_PATH):
    reader = pd.read_csv(
        filepath,
        chunksize=_CHUNKSIZE,
        quotechar='"',
        usecols=['content'],
        low_memory=False
    )
    vocab = set()
    with Pool(_N_WORKERS) as pool:
        for i, chunk_vocab in enumerate(
            pool.imap(
                vocabulary_of_content,
                reader,
                chunksize=1
            ),
            1
        ):
            vocab.update(chunk_vocab)
            print(f'Acquired vocabulary of first {i} chunks and first {i*_CHUNKSIZE:,} articles')

    print(f'Adding vocabulary to csv file in file path {_TRAIN_PATH}')
    pd.Series(list(vocab)).to_csv(_OUTPUT_PATH, mode='w', header=False, index=False)

    print("Vocabulary finished")

if __name__ == '__main__':
    build_vocab()