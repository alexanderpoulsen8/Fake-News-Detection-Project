import pandas as pd
import preprocessing as preproc
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "news_sample.csv")

df["content"] = preproc.rm_punctuation(df["content"])
tokens = preproc.tokenize_series(df["content"])
tokens = preproc.rm_stopwords(tokens)
tokens = preproc.stem_tokens(tokens)

print("rows:", len(df))
print("total tokens:", tokens.apply(len).sum())
print("total token chars:", preproc.token_char_size(tokens).sum())