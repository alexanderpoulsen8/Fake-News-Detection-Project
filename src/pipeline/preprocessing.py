import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


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
    patterns = {
        'whitespace': re.compile(r'\s+'),
        'email': re.compile(r'\S+(?![^\w@])\w@\S+(?![^\.])\.\w{2,}(\.\w{2,})?'),
        'url': re.compile(r'(http\S+)|(\S+(?![^\w\.])\w\.\w{2,}(\.\w{2,})?(\S+)?)'),
        'clean_bef_date': r'[^\w\s<>/-]',
        'non_word_non_space': re.compile(r'[^\w\s<>]'),
        'num': re.compile(r'\d+')
    }
    if tokenize_dates:
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
        patterns['date'] = re.compile(r'(' + r')|('.join(date_patts) + r')')

    cleaned = articles.copy()
    cleaned = cleaned.str.replace(pat=patterns['whitespace'], repl=r' ', regex=True)
    cleaned = cleaned.str.lower()
    cleaned = cleaned.str.replace(pat=patterns['email'], repl=r'<EMAIL>', regex=True)
    cleaned = cleaned.str.replace(pat=patterns['url'], repl=r'<URL>', regex=True)
    if tokenize_dates:
        cleaned = cleaned.str.replace(pat=patterns['clean_bef_date'], repl=r'', regex=True)
        cleaned = cleaned.str.replace(pat=patterns['date'], repl=r'<DATE>', regex=True)
    cleaned = cleaned.str.replace(pat=patterns['non_word_non_space'], repl=r'', regex=True)
    cleaned = cleaned.str.replace(pat=patterns['whitespace'], repl=r' ', regex=True)
    cleaned = cleaned.str.replace(pat=patterns['num'], repl=r'<NUM>', regex=True)
    return cleaned


def to_tokens(articles): #
    '''
    Get tokens from collection of articles

    Parameters:
        articles - Pandas Series of longer strings

    Returns:
        Pandas Series of tokens
    '''
    return articles.str.split(r'\s+').explode()


def rm_stopwords_from_tokens(tokens):
    '''
    Remove stopwords from array of tokens using nltk.corpus.stopwords

    Parameters:
        tokens - Pandas Series of tokens

    Returns:
        Pandas Series of tokens without stopwords
    '''
    stop_words = set(stopwords.words("english")) - {"no", "nor", "not"}
    tokens = tokens.astype("category")
    return tokens[~tokens.isin(stop_words)]

def stem_tokens(tokens):
    '''
    Stems pandas Series of tokens using nltk.stem.SnowBallStemmer

    Paramters:
        tokens - Pandas Series of tokens

    Returns:
        Pandas Series where stemming has been applied to every element
    '''
    stemmer = SnowballStemmer("english")
    return tokens.map(stemmer.stem)

def get_size(tokens):
    '''
    Get total number of characters of all elements in a Pandas Series
    '''
    return tokens.str.len().sum()

def preprocess(articles):
    '''
    Combined function of all functions in preprocessing module
    '''
    cleaned = clean_text(articles)

    tokens = cleaned.str.split(r'\s+').explode()

    stop_words = set(stopwords.words("english")) - {"no", "nor", "not"}
    tokens = tokens[~tokens.isin(stop_words)]

    stemmer = SnowballStemmer("english")
    tokens = tokens.map(stemmer.stem)

    return tokens
