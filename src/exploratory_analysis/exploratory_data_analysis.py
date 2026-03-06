import pandas as pd

def encode_vocabulary(tokens):
    vocab = pd.unique(tokens)

    cat_dtype = pd.CategoricalDtype(categories=vocab)
    token_codes = tokens.apply(lambda ls: pd.Categorical(ls, dtype=cat_dtype).codes)
    return token_codes

