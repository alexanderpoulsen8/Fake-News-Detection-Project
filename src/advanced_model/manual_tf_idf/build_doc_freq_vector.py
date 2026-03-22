import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

start_path = Path.cwd().parents[2]
_DATA_DIR = start_path / 'data' / 'big_dataset'
_TRAIN_PATH = _DATA_DIR / 'big_preprocessed_split' / "train.csv"
_DOC_FREQ_PATH = _DATA_DIR / 'tf_idf' / 'big_doc_freq_vector.csv'

_N_WORKERS = 1 # max(cpu_count() - 4, 1)
_CHUNKSIZE = 1000
_NROWS = 6821441

def vocabulary_of_content_per_article(df):
    df['content'] = df['content'].fillna('').str.split(' ')
    return pd.concat([
        df['content'].apply(lambda article: np.unique(article)),
        df['content'].apply(
            lambda doc: np.unique([' '.join([doc[i], doc[i+1]]) for i in range(len(doc)-1)])
        )
    ])


def build_doc_freq(train_path=_TRAIN_PATH):
    doc_freq = {}
    reader = pd.read_csv(
        train_path,
        chunksize=_CHUNKSIZE,
        usecols=['content'],
        low_memory=False,
        dtype=str
    )

    print('Processing corpus...\n')
    with Pool(_N_WORKERS) as pool:
        for i, chunk in enumerate(
            pool.imap(
                vocabulary_of_content_per_article,
                reader,
                chunksize=1
            ),
            1
        ):
            for article_vocab in chunk:
                for term in article_vocab:
                    doc_freq[term] = doc_freq.get(term, 0) + 1
            print(f'updated doc frequency vector of first {i} chunks and '
                  f'first {i*_CHUNKSIZE:,} articles')

    print(f'Adding doc frequency to csv file {_DOC_FREQ_PATH}')
    pd.DataFrame(
        list(doc_freq.items()), columns=['word','doc_freq']
    ).to_csv(_DOC_FREQ_PATH, index=False)
    print('Finished')

if __name__ == '__main__':
    build_doc_freq()