import pandas as pd
import re

df = pd.read_csv(
    "/Users/alexanderpoulsen/Downloads/995,000_rows.csv",
    nrows=10000,
    dtype={"id": str, "title": str, "author": str, "text": str, "label": int}
)
df.head()

# print head
print(df.head())

df["content"].head(20)


pattern1 = r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4}\b"

pattern2 = r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b"

pattern3 = r"\b\d{4}-\d{2}-\d{2}\b"

pattern4 = r"\b\d{1,2}/\d{1,2}/\d{4}\b"

def extract_dates(text):
    matches = []
    for p in [pattern1, pattern2, pattern3, pattern4]:
        matches.extend(re.findall(p, str(text)))
    return matches

df["date_candidates"] = df["content"].apply(extract_dates)

df[df["date_candidates"].map(len) > 0][["date_candidates"]].head(20)

all_dates = df["date_candidates"].explode().dropna().unique()
all_dates[:100]

from dateutil import parser

sample = all_dates[:100]

for d in sample:
    try:
        print(d, "->", parser.parse(d))
    except:
        print(d, "FAILED")

