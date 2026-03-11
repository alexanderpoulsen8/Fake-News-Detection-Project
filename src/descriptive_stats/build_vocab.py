import pandas as pd

_FILEPATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\preprocessed_dataset.csv"
_CHUNKSIZE = 50000
_OUTPUT_PATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\vocabulary.csv"

with pd.read_csv(
    _FILEPATH,
    chunksize=_CHUNKSIZE,
    quotechar='"',
    usecols=['content'],
    low_memory=False
) as reader:
    vocab = set()
    i = 0
    for chunk in reader:
        i += 1
        print(i)
        chunk_vocab = chunk['content'].str.lstrip("['").str.rstrip("']").str.split(r"', '").explode(ignore_index=True)

        vocab.update(chunk_vocab)
pd.Series(list(vocab)).to_csv(_OUTPUT_PATH, mode='w', header=False, index=False)

print("Vocabulary finished")
