import pandas as pd
from pathlib import Path
import numpy as np

StartPath = Path.cwd().parents[1]

_FILEPATH = StartPath / "data" / "reduced_vocabulary_stats.csv"

_OUTPUT_PATH = StartPath / "data" / "vocabulary_reliable_frequency.csv"

df = pd.read_csv(_FILEPATH, usecols=['word', 'count', 'types'], low_memory=False)
FAKE_LABELS = {'unreliable', 'hate', 'junksci', 'fake', 'satire', 'conspiracy', 'bias', 'clickbait'}
TRUE_LABELS = {'reliable', 'political', 'state'}
ALL_LABELS = {'unreliable', 'hate', 'junksci', 'fake', 'satire', 'conspiracy', 'bias', 'reliable', 'political', 'state', 'clickbait'}

df['types'] = df['types'].str.lstrip("{'").str.rstrip("}").str.split(r", '")
df['types'] = df['types'].apply(lambda row: {key: int(val) for key, val in [label.split("': ") for label in row]})

df['reliable_count'] = df['types'].apply(lambda w: sum([w.get(l, 0) for l in TRUE_LABELS]))
df['fake_count'] = df['types'].apply(lambda w: sum([w.get(l, 0) for l in FAKE_LABELS]))

df['score'] = np.log((df['reliable_count'] + 1) / (df['fake_count'] + 1))
df['score'] *= np.log1p(df['count'])

# df['reliable_freq'] = df['types'].apply(
#     lambda r:
#         sum([int(r.get(l, 0)) for l in TRUE_LABELS])/sum([int(r.get(l, 0)) for l in ALL_LABELS])
# )
# df['fake_freq'] = df['types'].apply(
#     lambda r:
#         sum([int(r.get(l, 0)) for l in FAKE_LABELS])/sum([int(r.get(l, 0)) for l in ALL_LABELS])
# )

df['normalizer'] = np.log1p(df['count'])

# df['reliable_normalized'] = df['reliable_freq']*df['normalizer']
# df['fake_normalized'] = df['fake_freq']*df['normalizer']

d_out = pd.DataFrame(
    data={
        'word': df['word'],
        'count': df['count'],
        'reliable_count': df['reliable_count'],
        'fake_count': df['fake_count'],
        # 'reliable_freq': df['reliable_freq'],
        'score': df['score'],
        'normalizer': df['normalizer']
        # 'reliable_normalized': df['reliable_normalized'],
        # 'fake_normalized': df['fake_normalized']
    }
)

d_out.to_csv(_OUTPUT_PATH, mode="w", header=True, index=False)