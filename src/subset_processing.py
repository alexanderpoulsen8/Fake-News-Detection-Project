import pandas as pd
from pipeline import preprocessing as preproc
import csv

file = '../../995,000_rows.csv'
fields = []
rows = []
with open(file, mode="r", encoding="utf-8", errors="replace") as f:
    csvreader = csv.reader(f)
    fields = next(csvreader)  # Read header
    i = 0
    for row in csvreader:
        if i >= 50:
            break
        rows.append(row)
        i += 1

print([row for row in rows])
