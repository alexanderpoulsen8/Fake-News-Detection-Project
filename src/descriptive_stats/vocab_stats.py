import pandas as pd
import exploratory_data_analysis as eda
from pathlib import Path

StartPath = Path.cwd()
_VOCAB_FILEPATH = StartPath / "data" / "vocabulary.csv"
_PROCESSED_FILEPATH = StartPath / "data" / "preprocessed_dataset.csv"
_OUTPUT_PATH = StartPath / "data" / "vocabulary_stats.csv"
_CHUNKSIZE = 50000

types = [
    'unreliable','hate','junksci','fake','satire','conspiracy','bias','rumor','unknown',
    'reliable','political','state','clickbait'
]
df_dict = {
    word: {
        'count': 0,
        'domains': set(),
        'types': {}
    } for word in pd.read_csv(_VOCAB_FILEPATH, header=None)[0]
}
default_row = {'count': 0, 'domains': set(), 'types': {}}

print('constructed')


with pd.read_csv(
    _PROCESSED_FILEPATH,
    chunksize=_CHUNKSIZE,
    usecols=['content','domain','type']
) as reader:
    i = 0
    for chunk in reader:
        i += 1
        print(i*_CHUNKSIZE)
        chunk['domain'] = chunk['domain'].fillna('unknown')
        chunk['type'] = chunk['type'].fillna('unknown')
        chunk_vocab = chunk['content'].str.lstrip("['").str.rstrip("']").str.split(r"', '")
        for tokens, domain, type in zip(chunk_vocab, chunk['domain'], chunk['type']):
            for token in tokens:
                df_dict.setdefault(token, default_row)
                df_dict[token]['types'].setdefault(type, 0)
                df_dict[token]['count'] += 1
                df_dict[token]['domains'].add(domain)
                df_dict[token]['types'][type] += 1

pd.DataFrame(df_dict).T.to_csv(_OUTPUT_PATH,mode="w",header=True)
