import pandas as pd
from preprocessing import preprocess_for_vectorizer
from multiprocessing import Pool, cpu_count
from pathlib import Path

StartPath = Path.cwd().parents[1]
data_dir = StartPath / 'data'
_SMALL_FILEPATH = data_dir / 'small_dataset' / 'news_sample.csv'
_FILEPATH = data_dir / 'medium_dataset' / "995,000_rows.csv"
_BIG_FILEPATH = data_dir / 'big_dataset' / "news_cleaned_2018_02_13.csv"
_SMALL_OUTPUT_PATH = data_dir / 'small_dataset' / 'small_preprocessed_dataset.csv'
_OUTPUT_PATH = data_dir / 'medium_dataset' / "preprocessed_for_vectorizer_dataset.csv"
_BIG_OUTPUT_PATH = data_dir / 'big_dataset' / "full_preprocessed_dataset.csv"

_CHUNKSIZE = 20000
_N_WORKERS = max(cpu_count() - 1, 1)

def process_chunk(chunk):
    chunk['content'] = preprocess_for_vectorizer(chunk["content"])
    return chunk


def preprocess_dataset(
    filepath=_FILEPATH,
    output_path=_OUTPUT_PATH,
    chunksize=_CHUNKSIZE,
    n_workers=_N_WORKERS
):
    cols = pd.read_csv(filepath, nrows=0).columns


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
            print(f"chunk {i:,} done, approx {i*chunksize:,} articles")
            processed_chunk.to_csv(
                output_path,
                mode="a",
                header=False,
                index=False
            )
    print("Done")

if __name__ == "__main__":
    preprocess_dataset()
