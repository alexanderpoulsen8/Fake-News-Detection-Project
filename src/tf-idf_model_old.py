import numpy as np
import pandas as pd
from pathlib import Path

StartPath = Path.cwd().parents[0]
_VOCAB_RELIABLE_SCORE_PATH = StartPath / "data" / "vocabulary_reliable_frequency.csv"
_PROCESSED_FILEPATH = StartPath / "data" / "preprocessed_dataset.csv"
_OUTPUT_PATH = StartPath / "data" / "predictions_by_vocab.csv"
_CHUNKSIZE = 10000

FAKE_LABELS = {'unreliable','hate','junksci','fake','satire','conspiracy','bias','clickbait'}
TRUE_LABELS = {'reliable','political','state'}

vocab_df = pd.read_csv(_VOCAB_RELIABLE_SCORE_PATH, index_col='word')


# fast lookup dictionary
score_map = vocab_df['score'].to_dict()

pd.DataFrame(columns=['score','prediction','target']).to_csv(_OUTPUT_PATH, index=False)

with pd.read_csv(
    _PROCESSED_FILEPATH,
    chunksize=_CHUNKSIZE,
    usecols=['content','type']
) as reader:

    for i, chunk in enumerate(reader, 1):

        chunk = chunk.mask(chunk['type'] == 'unknown').dropna(subset=['type'])
        chunk['type'] = chunk['type'].map(lambda t: True if t in TRUE_LABELS else False)

        print(i * _CHUNKSIZE)

        tokens = (
            chunk['content']
            .str.lstrip("['")
            .str.rstrip("']")
            .str.split(r"', '")
        )

        # sum token scores using dictionary lookup
        score = tokens.apply(lambda art: sum(score_map.get(t, 0) for t in art))

        out = pd.DataFrame({
            'score': score,
            'prediction': score > 0,
            'target': chunk['type']
        })

        out.to_csv(_OUTPUT_PATH, mode='a', header=False, index=False)


print("Done")