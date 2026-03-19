import pandas as pd



_PROCESSED_FILEPATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\full_preprocessed_dataset.csv"
_CHUNKSIZE = 50000

with pd.read_csv(
    _PROCESSED_FILEPATH,
    chunksize=_CHUNKSIZE,
    usecols=['type']
) as reader:
    i = 0
    n_state_articles = 0
    for chunk in reader:
        i += 1
        print(f'articles: {i*_CHUNKSIZE}')
        print(chunk['type'].unique())
        n_state_articles += (chunk['type'] == 'state').sum()
        print(f'Number of state articles in chunk: {n_state_articles}')


print('done')