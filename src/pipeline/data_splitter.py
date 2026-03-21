"""
Data Splitter Module
Splits preprocessed dataset into train/val/test (80/10/10)
"""
import os
import duckdb


def split_data(data_dir, preprocessed_file="full_preprocessed_dataset.csv", salt="v1"):
    """
    Split preprocessed data into train/val/test using DuckDB.

    Args:
        data_dir: Path to data directory
        preprocessed_file: Name of preprocessed CSV file
        salt: Salt for deterministic random splitting

    Returns:
        bool: True if split was performed, False if skipped
    """
    input_file = os.path.join(data_dir, preprocessed_file)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Preprocessed dataset not found: {input_file}")

    # Check if splits already exist
    train_file = os.path.join(data_dir, "train.csv")
    val_file = os.path.join(data_dir, "val.csv")
    test_file = os.path.join(data_dir, "test.csv")

    if all(os.path.exists(f) for f in [train_file, val_file, test_file]):
        print("Data splits already exist. Skipping...")
        return False

    print(f"Splitting {input_file}...")

    TRAIN, VAL = 0.80, 0.10
    DELIM = ","

    con = duckdb.connect()

    # Create view of source data
    con.execute(f"""
    CREATE OR REPLACE VIEW src AS
    SELECT *
    FROM read_csv_auto('{input_file}',
        delim='{DELIM}',
        header=true,
        sample_size=-1,
        ignore_errors=true
    );
    """)

    # Create deterministic random split
    con.execute(f"""
    CREATE OR REPLACE VIEW scored AS
    SELECT
      *,
      (hash('{salt}' || cast(id AS varchar))::DOUBLE / 18446744073709551616.0) AS u
    FROM src;
    """)

    # Export splits
    print("Exporting train split...")
    con.execute(f"""
    COPY (SELECT * EXCLUDE(u) FROM scored WHERE u < {TRAIN})
    TO '{train_file}'
    (HEADER, DELIMITER '{DELIM}');
    """)

    print("Exporting validation split...")
    con.execute(f"""
    COPY (SELECT * EXCLUDE(u) FROM scored WHERE u >= {TRAIN} AND u < {TRAIN + VAL})
    TO '{val_file}'
    (HEADER, DELIMITER '{DELIM}');
    """)

    print("Exporting test split...")
    con.execute(f"""
    COPY (SELECT * EXCLUDE(u) FROM scored WHERE u >= {TRAIN + VAL})
    TO '{test_file}'
    (HEADER, DELIMITER '{DELIM}');
    """)

    con.close()
    print("✓ Data split complete!")
    return True