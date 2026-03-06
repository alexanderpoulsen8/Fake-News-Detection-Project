import pandas as pd
import preprocessing as pp

# columns
# Unnamed, id, domain, type, url, content, scraped_at, inserted_at, updated_at, title, authors, keywords, meta_keywords, meta_description, tags, summary, source,

FILEPATH = "C:/Users/45515/OneDrive/Desktop/Studie/2_Semester_KU/GDS/Exam/data/995,000_rows.csv"
CHUNKSIZE = 50000
OUTPUT_PATH = "preprocessed_dataset.csv"
cols = pd.read_csv(FILEPATH, nrows=0).columns
pd.DataFrame(columns=cols).to_csv(OUTPUT_PATH, index=False)

with pd.read_csv(
    FILEPATH,
    chunksize=CHUNKSIZE,
    quotechar='"',
    usecols=cols,
    low_memory=False
) as reader:
    i = 0
    for chunk in reader:
        i += 1
        print(i)
        preproc_chunk = pd.DataFrame(chunk)
        preproc_chunk['content'] = pp.preprocess(preproc_chunk['content'])
        preproc_chunk.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
print("LETS FUCKING GOOOOOOOOOOO")