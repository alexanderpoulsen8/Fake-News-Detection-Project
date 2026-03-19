"""
Memory-efficient preprocessing using DuckDB
Processes large CSV files without loading everything into memory
"""
import duckdb
import os
from pathlib import Path
from .preprocessing import preprocess_for_vectorizer
import pandas as pd
from multiprocessing import Pool, cpu_count


def _process_batch_data(batch_df):
    """Helper function to preprocess a batch (for multiprocessing)."""
    if 'content' in batch_df.columns:
        try:
            batch_df['content'] = preprocess_for_vectorizer(batch_df['content'])
        except Exception as e:
            print(f"  Warning: Error preprocessing batch: {e}")
    return batch_df


def preprocess_with_duckdb(input_path, output_path, batch_size=10000, n_workers=None):
    """
    Preprocess large CSV using DuckDB for memory-efficient processing.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save preprocessed CSV
        batch_size: Number of rows to process at a time
        n_workers: Number of parallel workers (default: CPU count - 1)
    """
    if n_workers is None:
        n_workers = max(cpu_count() - 1, 1)
    
    print(f"Preprocessing {input_path} using DuckDB...")
    print(f"Batch size: {batch_size} rows")
    print(f"Workers: {n_workers} CPU cores")
    
    # Connect to DuckDB (in-memory)
    con = duckdb.connect(':memory:')
    
    # Create a view of the CSV file (doesn't load into memory)
    print("Creating CSV view...")
    con.execute(f"""
        CREATE VIEW raw_data AS 
        SELECT * FROM read_csv_auto('{input_path}', 
            ignore_errors=true,
            max_line_size=1048576,
            strict_mode=false
        )
    """)
    
    # Get total row count
    total_rows = con.execute("SELECT COUNT(*) FROM raw_data").fetchone()[0]
    print(f"Total rows: {total_rows:,}")
    
    # Get column names
    columns = [col[0] for col in con.execute("DESCRIBE raw_data").fetchall()]
    print(f"Columns: {', '.join(columns)}")
    
    # Initialize output file
    pd.DataFrame(columns=columns).to_csv(output_path, index=False)
    print(f"✓ Initialized output file: {output_path}")
    
    # Process in batches with parallel processing
    num_batches = (total_rows + batch_size - 1) // batch_size
    print(f"\nProcessing {num_batches} batches with {n_workers} workers...")
    
    # Generator to fetch batches on-demand
    def batch_generator():
        for batch_num in range(num_batches):
            offset = batch_num * batch_size
            query = f"""
                SELECT * FROM raw_data 
                LIMIT {batch_size} OFFSET {offset}
            """
            batch_df = con.execute(query).df()
            if len(batch_df) > 0:
                yield batch_df
    
    # Process batches in parallel using imap
    with Pool(n_workers) as pool:
        for i, processed_batch in enumerate(pool.imap(_process_batch_data, batch_generator(), chunksize=1), 1):
            # Save to file incrementally
            processed_batch.to_csv(output_path, mode='a', header=False, index=False)
            
            # Progress update
            processed = min(i * batch_size, total_rows)
            progress = (processed / total_rows) * 100
            print(f"  Batch {i}/{num_batches} - {processed:,}/{total_rows:,} rows ({progress:.1f}%)")
    
    con.close()
    print(f"\n✓ Preprocessing complete!")
    print(f"✓ Saved to {output_path}")


def main():
    """Main function for command-line usage."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python preprocess_with_duckdb.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    preprocess_with_duckdb(input_path, output_path)


if __name__ == "__main__":
    main()
