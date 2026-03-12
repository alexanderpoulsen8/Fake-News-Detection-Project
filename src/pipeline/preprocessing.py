import re
import pandas as pd
from pandarallel import pandarallel
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import WhitespaceTokenizer

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


# Compile once
_PATTERNS = create_patterns()
_NON_WORD_NON_SPACE = re.compile(r"[^\w\s]")

# Build once
_STOP_WORDS = set(stopwords.words("english")) - {"no", "nor", "not"}
_STEMMER = SnowballStemmer("english")

def clean_text(articles, tokenize_dates=True):
    '''
    Takes pandas string Series as input and cleans all article elements

    Parameters:
        articles - Pandas Series of strings
        tokenize_dates - Default False. True if you want dates to be tokenized as <DATE>

    Returns:
        Cleaned articles. Removes whitespace, non word/space characters, and
        tokenizes numbers, dates (if tokenize_date = True), URLs, and emails
    '''
    cleaned = articles.fillna("").astype(str).str.lower()
    cleaned = cleaned.str.replace(pat=_PATTERNS['whitespace'], repl=r' ', regex=True)
    cleaned = cleaned.str.replace(pat=_PATTERNS['email'], repl=r'<EMAIL>', regex=True)
    cleaned = cleaned.str.replace(pat=_PATTERNS['url'], repl=r'<URL>', regex=True)
    if tokenize_dates:
        cleaned = cleaned.str.replace(pat=_PATTERNS['clean_bef_date'], repl=r'', regex=True)
        cleaned = cleaned.str.replace(pat=_PATTERNS['date'], repl=r'<DATE>', regex=True)
    cleaned = cleaned.str.replace(pat=_PATTERNS['non_word_non_space'], repl=r'', regex=True)
    cleaned = cleaned.str.replace(pat=_PATTERNS['num'], repl=r'<NUM> ', regex=True)
    cleaned = cleaned.str.replace(pat=_PATTERNS['whitespace'], repl=r' ', regex=True)

    return cleaned


def rm_punctuation(texts):
    """
    texts: pandas Series (strings)
    Returns: Series with lowercase text and punctuation removed.
    """
    s = texts.fillna("").astype(str).str.lower()
    return s.str.replace(_NON_WORD_NON_SPACE, "", regex=True)


def tokenize_series(texts):
    """
    texts: pandas Series (strings)
    Returns: Series[list[str]] where each row is tokenized separately.
    """
    s = texts.fillna("").astype(str)
    return s.apply(WhitespaceTokenizer().tokenize)


def rm_stopwords(tokens_series):
    """
    tokens_series: Series[list[str]]
    Returns: Series[list[str]] with stopwords removed (keeps no/nor/not).
    """
    return tokens_series.apply(lambda toks: [t for t in toks if t not in _STOP_WORDS])


def stem_tokens(tokens_series):
    """
    tokens_series: Series[list[str]]
    Returns: Series[list[str]] stemmed.
    """
    return tokens_series.apply(lambda toks: [_STEMMER.stem(t) for t in toks])


def token_char_size(tokens_series):
    """
    tokens_series: Series[list[str]]
    Returns: Series[int] sum of token lengths per row.
    """
    return tokens_series.apply(lambda toks: sum(len(t) for t in toks))


def process_tokens(tokens_series):
    return tokens_series.apply(lambda tokens: [_STEMMER.stem(t) for t in tokens if t not in _STOP_WORDS])

def preprocess(articles, tokenize_dates=True):
    '''
    Combined function of all functions in preprocessing module
    '''
    cleaned = clean_text(articles, tokenize_dates=tokenize_dates)
    tokens_series = tokenize_series(cleaned)

    return process_tokens(tokens_series)

def encode_vocabulary(token_series):
    vocab = pd.unique(token_series.explode())
    cat_dtype = pd.CategoricalDtype(categories=vocab)
    token_codes = token_series.apply(lambda ls: pd.Categorical(ls, dtype=cat_dtype).codes)
    return token_codes

