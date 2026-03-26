import pandas as pd
from .preprocessing import preprocess_for_vectorizer
from multiprocessing import Pool, cpu_count
from pathlib import Path
import os
import sys

_CHUNKSIZE = 20000
_N_WORKERS = max(cpu_count() - 1, 1)

def process_chunk(chunk):
    """Process a chunk of data by cleaning the content column."""
    chunk["content"] = preprocess_for_vectorizer(chunk["content"])
    return chunk

def preprocess_large_csv(input_path, output_path, chunksize=100000, n_workers=None):
    """
    Preprocess a large CSV file in parallel using multiprocessing.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save preprocessed CSV
        chunksize: Number of rows per chunk
        n_workers: Number of worker processes (default: CPU count - 1)
    """
    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)
    
    print(f"Preprocessing {input_path}...")
    print(f"Using {n_workers} CPU cores, chunk size: {chunksize}")
    
    # Read column names
    cols = pd.read_csv(input_path, nrows=0).columns.tolist()
    
    # Initialize output file with headers
    pd.DataFrame(columns=cols).to_csv(output_path, index=False)
    print("✓ Initialized output file")

    # Use Python engine for more robust parsing (handles malformed CSVs better)
    print("Using Python engine for robust CSV parsing...")
    reader = pd.read_csv(
        input_path,
        chunksize=chunksize,
        quotechar='"',
        usecols=cols,
        on_bad_lines='skip',
        engine='python'
    )

    with Pool(n_workers) as pool:
        print(f"Starting parallel processing...")
        chunk_count = 0
        for i, processed_chunk in enumerate(pool.imap(process_chunk, reader, chunksize=1), 1):
            print(f"  Chunk {i} processed ({len(processed_chunk)} rows)")
            processed_chunk.to_csv(
                output_path,
                mode="a",
                header=False,
                index=False
            )
            chunk_count = i
    
    print(f"✓ Preprocessing complete! Processed {chunk_count} chunks")
    print(f"✓ Saved to {output_path}")

def main():
    """Main function using default paths for backward compatibility."""
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    preprocess_large_csv(input_path, output_path)

if __name__ == "__main__":
    main()
    print("Done")