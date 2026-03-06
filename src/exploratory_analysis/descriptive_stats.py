import pandas as pd
from multiprocessing import Pool, cpu_count

_FILEPATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\preprocessed_dataset.csv"
_CHUNKSIZE = 50000
_OUTPUT_PATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\descriptive_stats.csv"



with pd.read_csv(
    _FILEPATH,
    chunksize=_CHUNKSIZE,
    quotechar='"',
    usecols=['content'],
    low_memory=False
) as reader:
    df = pd.DataFrame({'URLs': 0, 'dates': 0, 'numbers': 0, 'rows analyzed': 0}, index=[0])
    i = 0
    for chunk in reader:
        i += 1
        print(i)
        df['URLs'] += chunk['content'].str.findall('<url>').transform(len).sum()
        df['dates'] += chunk['content'].str.findall('<date>').transform(len).sum()
        df['numbers'] += chunk['content'].str.findall('<num>').transform(len).sum()
        df['rows analyzed'] += len(chunk)
        df.to_csv(_OUTPUT_PATH, mode='w', header=True, index=False)


print("Finished counting")