import preprocessing as pp
import pandas as pd
from pathlib import Path
news_sample_path = Path.cwd().parents[1] / 'data' / 'small_dataset' / 'news_sample.csv'

print(' --- Number of characters --- ')
df = pd.read_csv(news_sample_path)
tokens = pp.tokenize_series(df['content'])
print(f'After only removing whitespace: {pp.token_char_size(tokens).sum()}')

df['content'] = pp.clean_text(df['content'])
tokens = pp.tokenize_series(df['content'])
print(f'After cleaning articles: {pp.token_char_size(tokens).sum()}')

tokens = pp.rm_stopwords(tokens)
print(f'After removing stopwords: {pp.token_char_size(tokens).sum()}')

tokens = pp.stem_tokens(tokens)
print(f'After stemming tokens: {pp.token_char_size(tokens).sum()}')

df = pd.read_csv(news_sample_path)
tokens = pp.preprocess(df['content'])
print(f'After using combined preprocess function: {pp.token_char_size(tokens).sum()}')