import preprocessing as preproc
import pandas as pd

print(' --- Number of characters --- ')

df = pd.read_csv('news_sample.csv')
tokens = preproc.to_tokens(df['content'])
print(f'Before any preprocessing: {preproc.get_size(tokens)}')

df['content'] = preproc.clean_text(df['content'], tokenize_dates=True)
tokens = preproc.to_tokens(df['content'])
print(f'After cleaning articles: {preproc.get_size(tokens)}')

tokens = preproc.rm_stopwords_from_tokens(tokens)
print(f'After removing stopwords: {preproc.get_size(tokens)}')

tokens = preproc.stem_tokens(tokens)
print(f'After stemming tokens: {preproc.get_size(tokens)}')
