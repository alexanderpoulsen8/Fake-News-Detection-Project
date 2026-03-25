import pandas as pd
from numpy import log
from pathlib import Path
from multiprocessing import Pool, cpu_count

start_path = Path.cwd().parents[2]
_DATA_DIR = start_path / 'data' / 'big_dataset'
_TRAIN_PATH = _DATA_DIR / 'big_preprocessed_split' / "train.csv"
_VOCAB_PATH = _DATA_DIR / 'tf_idf' / 'big_vocabulary.csv'
_IDF_VECTOR_PATH = _DATA_DIR / 'tf_idf' / 'big_idf_vector.csv'

_N_WORKERS = max(cpu_count() - 2, 1)
_CHUNKSIZE = 10000
_NROWS = 6821441

def vocabulary_of_content_per_article(df):
    return df['content'].fillna('').str.split(' ').apply(lambda article: set(article))

def build_idf_vector(train_path=_TRAIN_PATH, vocab_path=_VOCAB_PATH):
    print('Building dictionary...\n')
    vocab = pd.read_csv(
        vocab_path,
        header=None,
        low_memory=False,
        dtype=str,
        na_filter=False
    )[0]
    word_doc_occurrences = {}
    for word in vocab:
        word_doc_occurrences[word] = 0
    del vocab
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
                for word in article_vocab:
                    word_doc_occurrences[word] += 1
            print(f'updated word-doc-occurrences of first {i} chunks and '
                  f'first {i*_CHUNKSIZE:,} articles')
    print('\nConverting to idf dictionary...')
    for i, key in enumerate(word_doc_occurrences.keys(), 1):
        word_doc_occurrences[key] = log((_NROWS+1)/(word_doc_occurrences[key]+1)) + 1
        if i % 100000 == 0:
            print(f'idf of first {i:,} words computed')

    print(f'Adding idf dictionary to csv file {_IDF_VECTOR_PATH}')
    pd.DataFrame(list(word_doc_occurrences.items()), columns=['word','idf']).to_csv(_IDF_VECTOR_PATH, index=False)
    print('Finished')

if __name__ == '__main__':
    build_idf_vector()