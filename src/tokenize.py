import re
import numpy as np
import pandas as pd



def tokenize(texts): # parameter texts is a Pandas Series
    whitespace_patt = re.compile(r'\s+')
    non_word_non_space_patt = re.compile(r'[^\w\s<>]'),
    cleaned = texts.copy()
    cleaned = cleaned.str.replace(pat=whitespace_patt, repl=r' ', regex=True)
    cleaned = cleaned.str.lower()
    cleaned = cleaned.str.replace(pat=non_word_non_space_patt, repl='', regex=True)
    tokens = cleaned.apply(lambda x: re.split(whitespace_patt, x))
    return tokens

df = pd.read_csv('news_sample.csv')
print(tokenize(df['content']))