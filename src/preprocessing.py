import re
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt_tab')

def clean_text(texts):
    whitespace_patt = re.compile(r'\s+')
    non_word_non_space_patt = re.compile(r'[^\w\s<>]')
    cleaned = texts.copy()
    cleaned = cleaned.str.replace(pat=whitespace_patt, repl=r' ', regex=True)
    cleaned = cleaned.str.lower()
    cleaned = cleaned.str.replace(pat=non_word_non_space_patt, repl='', regex=True)
    return cleaned

def to_tokens(texts): # parameter texts is a Pandas Series
    tokens = [token for text in texts for token in nltk.word_tokenize(text)]
    return tokens

df = pd.read_csv('news_sample.csv')
print(to_tokens(clean_text(df['content']))[:100])