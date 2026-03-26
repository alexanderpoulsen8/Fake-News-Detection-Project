import pandas as pd

"""df = pd.read_csv(
    "/Users/alexanderpoulsen/Downloads/995,000_rows.csv",
    engine="python",
    on_bad_lines="skip",
    nrows=1000
)"""

"""print(df.shape)
print(df.columns)"""

"""df = pd.read_csv("/Users/alexanderpoulsen/Downloads/995,000_rows.csv", nrows=100000)
print(df.memory_usage(deep=True).sum() / 1e6, "MB")"""


for chunk in pd.read_csv(
    "/Users/alexanderpoulsen/Downloads/995,000_rows.csv",
    chunksize=200_000
):
    # preprocess here
    print(chunk.shape)
    break
