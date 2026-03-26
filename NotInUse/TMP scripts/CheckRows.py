import pandas as pd
import os
from pathlib import Path
#Script brugt til at læse csv filerne for at validere om de er splittet rigtigt
data_folder = r'C:\Users\jespe\GDS eksamen\Fake-News-Detection-Project\Data'

csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

print(f"Found {len(csv_files)} CSV files in {data_folder}\n")
print("="*80)

for csv_file in csv_files:
    file_path = os.path.join(data_folder, csv_file)
    try:
        df = pd.read_csv(file_path)
        rows, cols = df.shape
        print(f"File: {csv_file}")
        print(f"  Rows: {rows:,}")
        print(f"  Columns: {cols}")
        print(f"  Column names: {list(df.columns)}")
        print("-"*80)
    except Exception as e:
        print(f"File: {csv_file}")
        print(f"  Error: {e}")
        print("-"*80)
