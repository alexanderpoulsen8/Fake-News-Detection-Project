import pandas as pd
import numpy as np
from pathlib import Path

start_path = Path.cwd().parents[2]
_DATA_DIR = start_path / 'data' / 'big_dataset'
_TRAIN_PATH = _DATA_DIR / 'big_preprocessed_split' / "train.csv"
_DOC_FREQ_PATH = _DATA_DIR / 'tf_idf' / 'big_doc_freq_vector.csv'

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
    reader = pd.read_csv(
        train_path,
        chunksize=_CHUNKSIZE,
        usecols=['content'],
        low_memory=False,
        dtype=str
    )

    print('Processing corpus...\n')
    doc_freq = {}

    for i, chunk in enumerate(reader,1):
        for doc in chunk['content'].fillna(''):
            tokens = doc.split(' ')

            terms = set(tokens)
            terms |= {' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)}

            for term in terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        # prune if number of features is too much for memory to handle
        if len(doc_freq) > 100_000_000:
            print(f'Pruning after {i} chunks')
            doc_freq = {k: v for k, v in doc_freq.items() if v >= 2}

    print(f'Adding df dictionary to csv file \n{_DOC_FREQ_PATH}')
    pd.DataFrame(list(doc_freq.items()), columns=['term','doc_freq']).to_csv(_DOC_FREQ_PATH, index=False)
    print('Finished')

if __name__ == '__main__':
    build_doc_freq()