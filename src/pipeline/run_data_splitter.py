from data_splitter import *
from pathlib import Path

StartPath = Path.cwd().parents[1]
data_dir = StartPath / 'data'

split_data(data_dir)