import pandas as pd

df = pd.read_csv(
    "/Users/alexanderpoulsen/Downloads/995,000_rows.csv",
    nrows=10000,
    dtype={"id": str, "title": str, "author": str, "text": str, "label": int}
)
df.head()

# print head
print(df.head())