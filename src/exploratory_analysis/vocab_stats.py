import pandas as pd
import exploratory_data_analysis as eda

_VOCAB_FILEPATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\vocabulary.csv"
_PROCESSED_FILEPATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\preprocessed_dataset.csv"
_OUTPUT_PATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\vocabulary_stats.csv"
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

# df_dict['count'] = df_dict['count'].astype(int)
# df_dict.set_index('word', inplace=True)
# print(df_dict.loc['twerk'])
# df_dict.loc['null'] = {'count': 0, 'domains': set()}


# with pd.read_csv(_PROCESSED_FILEPATH, chunksize=_CHUNKSIZE, usecols=['content','domain']) as reader:
#     i = 0
#     for chunk in reader:
#         print(i)
#         chunk['domain'] = chunk['domain'].fillna('<unknown>')
#         chunk_vocab = chunk['content'].str.lstrip("['").str.rstrip("']").str.split(r"', '").dropna()
#         for tokens, domain in zip(chunk_vocab, chunk['domain']):
#             i += 1
#             if i < 700:
#                 continue
#             for token in tokens:
#                 if token not in df_dict.index:
#                     # Add a new row
#                     df_dict.loc[token] = {'count': 1, 'domains': {domain}}
#                     print(i, df_dict.loc[token])
#                 df_dict.loc[token, 'count'] = df_dict.loc[token, 'count'] + 1
#                 df_dict.loc[token, 'domains'].add(domain)
#         break
# df_dict.to_csv(_OUTPUT_PATH,mode="w",header=True)