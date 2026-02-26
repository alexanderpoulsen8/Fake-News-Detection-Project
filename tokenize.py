import re
import numpy as np
import pandas as pd



def tokenize(texts): # parameter texts is a Pandas Series
    whitespace_patt = re.compile(r'\s+')
    cleaned = texts.copy()
    cleaned = cleaned.str.replace(pat=whitespace_patt, repl=r' ', regex=True)
    cleaned = cleaned.str.lower()


df = pd.read_csv('news_sample.csv')