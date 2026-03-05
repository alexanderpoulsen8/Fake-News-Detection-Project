import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from itertools import chain

def create_patterns():
    months = [r'january', r'february', r'march', r'april', r'may', r'june',
            r'july', r'august', r'september', r'october', r'november', r'december',
            r'jan', r'feb', r'mar', r'apr', r'jun', r'jul',
            r'aug', r'sep', r'oct', r'nov', r'dec']
    month_pattern = r'(' + '|'.join(months) + r')'
    ddmmyy_patt = r'(\d{1,2}-\d{1,2}-\d{2,4})|(\d{1,2}/\d{1,2}/\d{2,4})\W'
    m_d_y_patt = month_pattern + r'\s\d{1,2}(st|nd|rd|th)?,?\s\d{4}\W'
    d_m_y_patt = r'\s\d{1,2}(st|nd|rd|th)?\s(of\s)?' + month_pattern + r',?\s\d{4}\W'
    m_d_patt = month_pattern + r',?\s\d{1,2}(st|nd|rd|th)?\W'
    d_m_patt =  r'\s\d{1,2}(st|nd|rd|th)?\s(of\s)?' + month_pattern
    date_patts = [ddmmyy_patt, m_d_y_patt, d_m_y_patt, m_d_patt, d_m_patt]
    patts = {
        'whitespace': re.compile(r'\s+'),
        'email': re.compile(r'\S+(?![^\w@])\w@\S+(?![^\.])\.\w{2,}(\.\w{2,})?'),
        'url': re.compile(r'(http\S+)|(\S+(?![^\w\.])\w\.\w{2,}(\.\w{2,})?(\S+)?)'),
        'clean_bef_date': r'[^\w\s<>/-]',
        'date': re.compile(r'(' + r')|('.join(date_patts) + r')'),
        'non_word_non_space': re.compile(r'[^\w\s<>]'),
        'num': re.compile(r'\d+(st\s|nd\s|rd\s|th\s)?')
    }
    return patts
PATTERNS = create_patterns()
STOPWORDS = set(stopwords.words("english")) - {"no", "nor", "not"}
STEMMER = SnowballStemmer("english")

def clean_text(articles, tokenize_dates = False):
    '''
    Takes pandas string Series as input and cleans all article elements

    Parameters:
        articles - Pandas Series of strings
        tokenize_dates - Default False. True if you want dates to be tokenized as <DATE>

    Returns:
        Cleaned articles. Removes whitespace, non word/space characters, and
        tokenizes numbers, dates (if tokenize_date = True), URLs, and emails
    '''


    cleaned = articles.copy()
    cleaned = cleaned.str.replace(pat=PATTERNS['whitespace'], repl=r' ', regex=True)
    cleaned = cleaned.str.lower()
    cleaned = cleaned.str.replace(pat=PATTERNS['email'], repl=r'<EMAIL>', regex=True)
    cleaned = cleaned.str.replace(pat=PATTERNS['url'], repl=r'<URL>', regex=True)
    if tokenize_dates:
        cleaned = cleaned.str.replace(pat=PATTERNS['clean_bef_date'], repl=r'', regex=True)
        cleaned = cleaned.str.replace(pat=PATTERNS['date'], repl=r'<DATE>', regex=True)
    cleaned = cleaned.str.replace(pat=PATTERNS['non_word_non_space'], repl=r'', regex=True)
    cleaned = cleaned.str.replace(pat=PATTERNS['num'], repl=r'<NUM> ', regex=True)
    cleaned = cleaned.str.replace(pat=PATTERNS['whitespace'], repl=r' ', regex=True)

    return cleaned


def to_tokens(articles): #
    '''
    Get pandas Series of lists of tokens from collection of articles

    Parameters:
        articles - Pandas Series of longer strings

    Returns:
        Pandas Series of lists of tokens
    '''
    return articles.str.split(PATTERNS['whitespace'])


def rm_stopwords_from_tokens(tokens):
    '''
    Remove stopwords from array of tokens using nltk.corpus.stopwords

    Parameters:
        tokens - Pandas Series of tokens

    Returns:
        Pandas Series of tokens without stopwords
    '''
    return tokens.apply(lambda ls: [token for token in ls if token not in STOPWORDS])

def stem_tokens(tokens):
    '''
    Stems pandas Series of tokens using nltk.stem.SnowBallStemmer

    Paramters:
        tokens - Pandas Series of tokens

    Returns:
        Pandas Series where stemming has been applied to every element
    '''
    return tokens.apply(lambda ls: [STEMMER.stem(token) for token in ls])

def get_size(tokens):
    '''
    Get total number of characters of all elements in a Pandas Series
    '''
    return sum(len(token) for ls in tokens for token in ls)

def encode_vocabulary(token_series):
    vocab = pd.unique(token_series.explode())
    cat_dtype = pd.CategoricalDtype(categories=vocab)
    token_codes = token_series.apply(lambda ls: pd.Categorical(ls, dtype=cat_dtype).codes)
    return token_codes

def preprocess(articles):
    '''
    Combined function of all functions in preprocessing module
    '''
    cleaned = clean_text(articles)

    tokens = cleaned.str.split(PATTERNS['whitespace'])

    tokens = tokens[~tokens.isin(STOPWORDS)]

    stemmer = SnowballStemmer("english")
    tokens = tokens.map(stemmer.stem)

    return tokens
