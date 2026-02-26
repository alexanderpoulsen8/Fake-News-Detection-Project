import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download("stopwords")



def to_tokens(texts): # parameter texts is a Pandas Series
    tokens = [token for text in texts for token in nltk.word_tokenize(text)]
    return tokens


def rm_punctuation(texts):
    non_word_non_space_patt = re.compile(r'[^\w\s]')
    cleaned = texts.copy()
    cleaned = cleaned.str.lower()
    cleaned = cleaned.str.replace(pat=non_word_non_space_patt, repl='', regex=True)
    return cleaned

def rm_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    stop_words -= {"no", "nor", "not"}
    tokens_no_sw = [word for word in tokens if word not in stop_words]
    return tokens_no_sw


df = pd.read_csv('news_sample.csv')
tokens = to_tokens(df['content'])
print(len(tokens))
df['content'] = rm_punctuation(df['content'])
tokens = to_tokens(df['content'])
print(len(tokens))
tokens = rm_stopwords(tokens)
print(len(tokens))