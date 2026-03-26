#!/usr/bin/env python3

import csv
from pathlib import Path

input_dir = Path("data/liar/liar_dataset")
output_file = input_dir / "liar_dataset_combined.tsv"

input_files = [
    input_dir / "train.tsv",
    input_dir / "valid.tsv",
    input_dir / "test.tsv",
]

def main() -> None:
    for file in input_files:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    with output_file.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f, delimiter="\t")

        for input_file in input_files:
            with input_file.open("r", newline="", encoding="utf-8") as in_f:
                reader = csv.reader(in_f, delimiter="\t")

                for row in reader:
                    writer.writerow(row)

    print(f"Done. Combined TSV written to:\n{output_file}")

if __name__ == "__main__":
    main()