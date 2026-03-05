import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Compile once
_NON_WORD_NON_SPACE = re.compile(r"[^\w\s]")

# Build once
_STOP_WORDS = set(stopwords.words("english")) - {"no", "nor", "not"}
_STEMMER = SnowballStemmer("english")


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
    return s.apply(nltk.word_tokenize)


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