import pandas as pd
import exploratory_data_analysis as eda

FILEPATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\preprocessed_dataset.csv"
CHUNKSIZE = 500
OUTPUT_PATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\descriptive_stats.csv"


with pd.read_csv(
    FILEPATH,
    chunksize=CHUNKSIZE,
    quotechar='"',
    usecols=['content'],
    low_memory=False
) as reader:
    df = {'URLs': 0, 'dates': 0, 'numbers': 0}
    i = 0
    for chunk in reader:
        i += 1

        df['URLs'] += chunk['content'].str.findall('<url>').transform(len).sum()
        df['dates'] += chunk['content'].str.findall('<date>').transform(len).sum()
        df['numbers'] += chunk['content'].str.findall('<num>').transform(len).sum()
        if i % 10 == 0:
            break
pd.Series(df).to_csv(OUTPUT_PATH, mode='w', header=False, index=False)

print("LETS FUCKING GOOOOOOOOOOO")