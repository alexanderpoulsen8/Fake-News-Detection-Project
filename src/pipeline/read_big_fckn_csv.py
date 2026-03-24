import pandas as pd
from preprocessing import preprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path

StartPath = Path.cwd().parents[1]
_OUTPUT_PATH = StartPath / "data" / 'small_dataset' / "preprocessed_dataset.csv"
_BIG_OUTPUT_PATH = StartPath / "data" / 'big_dataset' / "full_preprocessed_dataset.csv"
_FILEPATH = StartPath / "data" / 'small_dataset' / "995,000_rows.csv"
_BIG_FILEPATH = StartPath / "data" / 'big_dataset' / "news_cleaned_2018_02_13.csv"

_CHUNKSIZE = 5000
_N_WORKERS = max(cpu_count() - 1, 1)

def process_chunk(chunk):
    chunk['content_clean'] = preprocess(chunk["content"])
    return chunk

def preprocess_dataset(
    filepath=_FILEPATH,
    output_path=_OUTPUT_PATH,
    chunksize=_CHUNKSIZE,
    n_workers=_N_WORKERS
):
    cols = pd.read_csv(_FILEPATH, nrows=0).columns


    pd.DataFrame(columns=cols).to_csv(output_path, index=False)
    print("Added columns to csv file")

    reader = pd.read_csv(
        filepath,
        chunksize=chunksize,
        quotechar='"',
        usecols=cols,
        low_memory=False
    )

    with Pool(n_workers) as pool:
        print(f"Starting process using {n_workers} CPU processors")
        for i, processed_chunk in enumerate(pool.imap(process_chunk, reader, chunksize=1), 1):
            print(f"chunk {i} done")
            processed_chunk.to_csv(
                output_path,
                mode="a",
                header=False,
                index=False
            )
    print("Done")

if __name__ == "__main__":
    preprocess_dataset()
