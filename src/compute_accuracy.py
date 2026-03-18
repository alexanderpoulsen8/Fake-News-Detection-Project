import pandas as pd
from pathlib import Path

StartPath = Path.cwd().parents[0]

_FILEPATH = StartPath / "data" / "predictions_by_vocab.csv"
df = pd.read_csv(_FILEPATH, usecols=['prediction', 'target'])
correct = (df['prediction'] == df['target'])
print(sum(correct)/len(df))
