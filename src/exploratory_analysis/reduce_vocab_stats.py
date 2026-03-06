import pandas as pd

_FILEPATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\vocabulary_stats.csv"
_OUTPUT_PATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\reduced_vocabulary_stats.csv"

df = pd.read_csv(_FILEPATH, low_memory=False)

df = df.mask(df['count'] < 5).dropna()
df['count'] = df['count'].astype(int)
df = df.sort_values(by=['count'], ascending=False)
df.to_csv(_OUTPUT_PATH,mode="w",header=True, index=False)