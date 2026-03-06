import pandas as pd

FILEPATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\preprocessed_dataset.csv"
CHUNKSIZE = 500
OUTPUT_PATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\vocabulary.csv"


with pd.read_csv(
    FILEPATH,
    chunksize=CHUNKSIZE,
    quotechar='"',
    usecols=['content'],
    low_memory=False
) as reader:
    vocab = set()
    i = 0
    for chunk in reader:
        chunk_vocab = chunk['content'].str.lstrip("['").str.rstrip("']").str.split(r"', '").explode(ignore_index=True)
        i += 1
        if i == 1:
            pd.Series(chunk['content'][0]).to_csv(OUTPUT_PATH, mode='w', header=False, index=False)
        print(chunk_vocab)
        vocab.update(chunk_vocab)
        if i % 10 == 0:
            break
pd.Series(list(vocab)).to_csv(OUTPUT_PATH, mode='a', header=False, index=False)

print("LETS FUCKING GOOOOOOOOOOO")