import preprocessing as preproc
import pandas as pd


df = pd.read_csv('news_sample.csv')
tokens = preproc.to_tokens(df['content'])
print(preproc.get_size(tokens))
df['content'] = preproc.rm_punctuation(df['content'])
tokens = preproc.to_tokens(df['content'])
print(preproc.get_size(tokens))
tokens = preproc.rm_stopwords_from_tokens(tokens)
print(preproc.get_size(tokens))
tokens = preproc.stem_tokens(tokens)
print(preproc.get_size(tokens))