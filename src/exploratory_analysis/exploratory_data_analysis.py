import pandas as pd

def encode_vocabulary(token_series):
    vocab = pd.unique(token_series.explode())
    cat_dtype = pd.CategoricalDtype(categories=vocab)
    token_codes = token_series.apply(lambda ls: pd.Categorical(ls, dtype=cat_dtype).codes)
    return token_codes

