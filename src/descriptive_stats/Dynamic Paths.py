
# Istedet for at hard code stier så brug det her i stedet.

from pathlib import Path

StartPath = Path.cwd()
_VOCAB_FILEPATH = StartPath / "data" / "vocabulary.csv"
_PROCESSED_FILEPATH = StartPath / "data" / "preprocessed_dataset.csv"
_OUTPUT_PATH = StartPath / "data" / "vocabulary_stats.csv"