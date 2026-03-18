import pandas as pd
from preprocessing import preprocess_for_vectorizer
from multiprocessing import Pool, cpu_count
from pathlib import Path

StartPath = Path.cwd().parents[1]
_OUTPUT_PATH = StartPath / "data" / "full_preprocessed_dataset.csv"
_FILEPATH = StartPath / "data" / "news_cleaned_2018_02_13.csv"

_CHUNKSIZE = 20000
_N_WORKERS = max(cpu_count() - 2, 1)

def process_chunk(chunk):
    chunk["content_clean"] = preprocess_for_vectorizer(chunk["content"])
    return chunk

def main():
    cols = pd.read_csv(_FILEPATH, nrows=0).columns.tolist()
    output_cols = cols + ["content_clean"]

    pd.DataFrame(columns=output_cols).to_csv(_OUTPUT_PATH, index=False)
    print("Added columns to csv file")

    reader = pd.read_csv(
        _FILEPATH,
        chunksize=_CHUNKSIZE,
        quotechar='"',
        usecols=cols,
        low_memory=False
    )

    with Pool(_N_WORKERS) as pool:
        print(f"Starting process using {_N_WORKERS} CPU cores")
        for i, processed_chunk in enumerate(pool.map(process_chunk, reader, chunksize=1), 1):
            print(f"chunk {i} done")
            processed_chunk.to_csv(
                _OUTPUT_PATH,
                mode="a",
                header=False,
                index=False
            )

if __name__ == "__main__":
    main()
    print("Done")