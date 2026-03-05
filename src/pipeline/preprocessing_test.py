import preprocessing as pp
import pandas as pd

print(' --- Number of characters --- ')
df = pd.read_csv('news_sample.csv')
tokens = pp.to_tokens(df['content'])
print(f'After only removing whitespace: {pp.get_size(tokens)}')

df['content'] = pp.clean_text(df['content'], tokenize_dates=True)
tokens = pp.to_tokens(df['content'])
print(f'After cleaning articles: {pp.get_size(tokens)}')

tokens = pp.rm_stopwords_from_tokens(tokens)
print(f'After removing stopwords: {pp.get_size(tokens)}')

tokens = pp.stem_tokens(tokens)
print(f'After stemming tokens: {pp.get_size(tokens)}')

df = pd.read_csv('news_sample.csv')
tokens = pp.preprocess(df['content'], tokenize_dates=True)
print(f'After using combined preprocess function: {pp.get_size(tokens)}')