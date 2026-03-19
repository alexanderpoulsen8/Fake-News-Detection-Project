import pandas as pd
import sys
from pathlib import Path


def analyze_type_column(csv_file):
    """Analyze the 'type' column in a CSV file and print comprehensive statistics."""
    
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_file}")
        return
    
    print(f"Analyzing file: {csv_file}")
    print(f"File size: {csv_path.stat().st_size / (1024**3):.2f} GB")
    print("="*80)
    
    # Chunked reading for all files
    chunk_size = 100000
    type_counts = {}
    total_rows = 0
    null_count = 0
    unique_labels_set = set()
    
    print("\nReading file in chunks...")
    for i, chunk in enumerate(pd.read_csv(csv_file, usecols=['type'], chunksize=chunk_size, low_memory=False)):
        total_rows += len(chunk)
        null_count += chunk['type'].isna().sum()
        
        # Collect unique labels
        unique_labels_set.update(chunk['type'].dropna().unique())
        
        # Count occurrences
        for label, count in chunk['type'].value_counts(dropna=False).items():
            type_counts[label] = type_counts.get(label, 0) + count
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {total_rows:,} rows...")
    
    print(f"\nTotal rows processed: {total_rows:,}")
    print("\n" + "="*80)
    print("TYPE COLUMN ANALYSIS")
    print("="*80)
    
    # Sort counts by frequency
    sorted_counts = sorted(type_counts.items(), key=lambda x: x[1] if pd.notna(x[0]) else 0, reverse=True)
    
    print("\n1. VALUE COUNTS (Absolute):")
    print("-"*80)
    for label, count in sorted_counts:
        print(f"  {str(label):20s}: {count:>10,} rows")
    
    print("\n2. VALUE COUNTS (Percentage):")
    print("-"*80)
    for label, count in sorted_counts:
        pct = (count / total_rows) * 100
        print(f"  {str(label):20s}: {pct:>6.2f}%")
    
    print("\n3. MISSING VALUES:")
    print("-"*80)
    null_pct = (null_count / total_rows) * 100
    print(f"  Null/NaN values: {null_count:,} ({null_pct:.2f}%)")
    print(f"  Non-null values: {total_rows - null_count:,} ({100-null_pct:.2f}%)")
    
    print("\n4. UNIQUE VALUES:")
    print("-"*80)
    print(f"  Total unique labels (excluding NaN): {len(unique_labels_set)}")
    
    print("\n5. ALL UNIQUE LABELS:")
    print("-"*80)
    for label in sorted(unique_labels_set):
        print(f"  - {label}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ReadLabels.py <path_to_csv_file>")
        print("\nExample:")
        print("  python ReadLabels.py data/train.csv")
        print("  python ReadLabels.py \"C:/Users/jespe/GDS eksamen/Fake-News-Detection-Project/data/train.csv\"")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    analyze_type_column(csv_file)
