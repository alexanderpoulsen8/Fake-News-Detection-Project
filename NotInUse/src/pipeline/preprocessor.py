"""
Data Preprocessor Module
Preprocesses raw CSV dataset with text cleaning and tokenization
"""
import os
import pandas as pd
from .preprocessing import preprocess


def preprocess_dataset(input_file, output_file, text_column="content", chunksize=50000):
    """
    Preprocess raw dataset with text cleaning and tokenization.
    
    Args:
        input_file: Path to raw CSV file
        output_file: Path to save preprocessed CSV
        text_column: Name of the text column to preprocess
        chunksize: Number of rows to process at a time (for large files)
        
    Returns:
        bool: True if preprocessing was performed, False if skipped
    """
    # Check if output already exists
    if os.path.exists(output_file):
        print(f"Preprocessed file already exists: {output_file}")
        print("Skipping preprocessing...")
        return False
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading raw data from {input_file}...")
    
    # Try multiple strategies to read the CSV
    df = None
    
    # Strategy 1: Standard C engine with error handling
    try:
        print("Attempting standard read with C engine...")
        df = pd.read_csv(
            input_file,
            low_memory=False,
            on_bad_lines='skip',
            encoding='utf-8'
        )
        print(f"✓ Successfully loaded {len(df)} rows")
    except Exception as e:
        print(f"  Failed: {str(e)[:100]}")
    
    # Strategy 2: Python engine (more robust, no low_memory option)
    if df is None:
        try:
            print("Attempting read with Python engine...")
            df = pd.read_csv(
                input_file,
                on_bad_lines='skip',
                engine='python',
                encoding='utf-8',
                quoting=1
            )
            print(f"✓ Successfully loaded {len(df)} rows")
        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")
    
    # Strategy 3: Chunked reading with C engine
    if df is None:
        try:
            print(f"Attempting chunked reading (chunks of {chunksize})...")
            chunks = []
            for i, chunk in enumerate(pd.read_csv(
                input_file,
                chunksize=chunksize,
                on_bad_lines='skip',
                encoding='utf-8'
            )):
                print(f"  Loaded chunk {i+1} ({len(chunk)} rows)...")
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            print(f"✓ Successfully loaded {len(df)} rows from {len(chunks)} chunks")
        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")
    
    # Strategy 4: Chunked reading with Python engine
    if df is None:
        try:
            print(f"Attempting chunked reading with Python engine...")
            chunks = []
            for i, chunk in enumerate(pd.read_csv(
                input_file,
                chunksize=chunksize,
                on_bad_lines='skip',
                engine='python',
                encoding='utf-8',
                quoting=1
            )):
                print(f"  Loaded chunk {i+1} ({len(chunk)} rows)...")
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            print(f"✓ Successfully loaded {len(df)} rows from {len(chunks)} chunks")
        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")
    
    # If all strategies failed
    if df is None:
        raise RuntimeError(
            "Could not read CSV file with any method. "
            "The file may be corrupted or in an unsupported format. "
            "Try opening it in Excel or a text editor to verify the structure."
        )
    
    print(f"Original dataset size: {len(df)} rows")
    
    if text_column not in df.columns:
        available_cols = ', '.join(df.columns[:10])
        raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {available_cols}...")
    
    print(f"Preprocessing '{text_column}' column...")
    print("  - Cleaning text (URLs, emails, dates, numbers)")
    print("  - Tokenizing")
    print("  - Removing stopwords")
    print("  - Stemming")
    
    # Preprocess the text column
    df[text_column] = preprocess(df[text_column], tokenize_dates=True)
    
    # Remove rows with empty content after preprocessing
    df = df[df[text_column].apply(lambda x: len(x) > 0)]
    
    print(f"After preprocessing: {len(df)} rows")
    
    # Save preprocessed data
    print(f"\nSaving preprocessed data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("✓ Preprocessing complete!")
    return True
