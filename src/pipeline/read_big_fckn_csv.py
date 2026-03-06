import pandas as pd
from preprocessing import preprocess
from multiprocessing import Pool, cpu_count


_FILEPATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\995,000_rows.csv"
_CHUNKSIZE = 20000
_OUTPUT_PATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\preprocessed_dataset.csv"
_N_WORKERS = max(cpu_count() - 1, 1)

def process_chunk(chunk):
    chunk["content"] = preprocess(chunk["content"])
    return chunk

def main():

    _COLS = pd.read_csv(_FILEPATH, nrows=0).columns
    pd.DataFrame(columns=_COLS).to_csv(_OUTPUT_PATH, index=False)
    print('Added columns to csv file')

    reader = pd.read_csv(
        _FILEPATH,
        chunksize=_CHUNKSIZE,
        quotechar='"',
        usecols=_COLS,
        low_memory=False
    )

    with Pool(_N_WORKERS) as pool:
        print(f'Starting process using {_N_WORKERS} CPU cores')
        for i, processed_chunk in enumerate(pool.imap(process_chunk, reader, chunksize=1), 1):
            print(f"chunk {i} done")
            processed_chunk.to_csv(
                _OUTPUT_PATH,
                mode="a",
                header=False,
                index=False
            )

if __name__ == "__main__":
    main()
    print("LETS FUCKING GOOOOOOOOOOO")
