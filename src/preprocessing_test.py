import preprocessing as preproc
import pandas as pd

df = pd.read_csv('news_sample.csv')
tokens = preproc.to_tokens(df['content'])
print(len(tokens))
df['content'] = preproc.rm_punctuation(df['content'])
tokens = preproc.to_tokens(df['content'])
print(len(tokens))
tokens = preproc.rm_stopwords(tokens)
print(len(tokens))