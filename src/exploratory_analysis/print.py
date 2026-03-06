import pandas as pd

_FILEPATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\preprocessed_dataset.csv"
_COLS = pd.read_csv(_FILEPATH, nrows=100)

# for keyword in _COLS['domain']:
#     print(keyword)
print(_COLS.columns)