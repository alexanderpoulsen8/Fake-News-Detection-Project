import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('punkt_tab')
nltk.download("stopwords")

def rm_punctuation(texts):
    '''Takes pandas series as input and
    removes all non-word and non-space characters and converts
    all word characters to lowercase'''
    non_word_non_space_patt = re.compile(r'[^\w\s]')
    cleaned = texts.copy()
    cleaned = cleaned.str.lower()
    cleaned = cleaned.str.replace(pat=non_word_non_space_patt, repl='', regex=True)
    return cleaned

def to_tokens(texts): # parameter texts is a Pandas Series
    tokens = [token for text in texts for token in nltk.word_tokenize(text)]
    return tokens

def rm_stopwords_from_tokens(tokens: list[str]):
    stop_words = set(stopwords.words("english"))
    stop_words -= {"no", "nor", "not"}
    tokens_no_sw = [word for word in tokens if word not in stop_words]
    return tokens_no_sw

def stem_tokens(tokens: list[str]):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(token) for token in tokens]

def get_size(tokens: list[str]):
    return sum([len(token) for token in tokens])