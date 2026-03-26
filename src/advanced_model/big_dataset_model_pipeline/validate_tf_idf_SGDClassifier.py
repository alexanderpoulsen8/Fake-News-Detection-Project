import tf_idf_vectorizer as tfidf
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import joblib

start_path = Path.cwd().parents[2]
data_dir = start_path / 'data' / 'LIAR'
_VAL_PATH = data_dir / 'valid.tsv'
_MODEL_PATH = data_dir / 'models' / 'SGDClassifier.joblib'
_OUTPUT_RESULTS_PATH = data_dir / 'results' / 'SGDClassifier_metrics.txt'

def main(
    val_path=_VAL_PATH,
    model_path=_MODEL_PATH,
    output_path=_OUTPUT_RESULTS_PATH
):
    X, y = tfidf.vectorize_articles(val_path, LIAR=True)
    clf = joblib.load(model_path)

    print('\nPredicting y values...')
    y_pred = clf.predict(X)

    print('\nComputing F1-score...')
    f1 = f1_score(y, y_pred, average='weighted')
    report = classification_report(y, y_pred)

    print(f"\nWeighted validation F1: {f1:.4f}")
    print(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Weighted validation F1: {f1:.4f}\n\n")
        f.write(f"Val path: {val_path}\n")
        f.write("TF-IDF config:\n")
        f.write("  Absolute min_df=1500\n")
        f.write("  Relative max_df=0.1\n")
        f.write("  ngram_range=(1,2)\n")
        f.write("  sublinear_tf=True\n")
        f.write("  smooth_idf=True\n")
        f.write("SGDClassifier config:\n")
        f.write(f"  loss='hinge'\n")
        f.write(f"  penalty='l2'\n")
        f.write(f"  alpha=1e-7\n")
        f.write(f"  learning_rate='constant'\n")
        f.write(f"  eta0=0.001\n")
        f.write(f"  fit_intercept=True\n")
        f.write(f"  max_iter=1\n")
        f.write(f"  tol=None\n")
        f.write(f"  shuffle=False\n")
        f.write(f"  average=True\n")
        f.write(f"  class_weight=class_weight_dict (class_weight_dict computed manually in train_SGDClassifier.py)\n\n")
        f.write(report)
    print(f"Saved results to {output_path}")


if __name__ == '__main__':
    main()



# import pandas as pd
# import numpy as np
# import tf_idf_vectorizer as Vec
# from sklearn.metrics import f1_score, classification_report
# from multiprocessing import Pool, cpu_count
# from pathlib import Path
# import joblib

# start_path = Path.cwd().parents[2]
# data_dir = start_path / 'data' / 'big_dataset'
# _VAL_PATH = data_dir / 'big_preprocessed_split' / 'val.csv'
# _MODEL_PATH = data_dir / 'models' / 'SGDClassifier.joblib'
# _OUTPUT_RESULTS_PATH = data_dir / 'results' / 'SGDClassifier_metrics.txt'

# _N_WORKERS = max(cpu_count() - 1, 1)
# _CHUNKSIZE = 20000

# def predict(
#     val_path=_VAL_PATH,
#     model_path=_MODEL_PATH,
#     output_path=_OUTPUT_RESULTS_PATH,
#     n_workers=_N_WORKERS,
#     chunksize=_CHUNKSIZE
# ):
#     clf = joblib.load(model_path)
#     print('\nProcessing articles in validation set...')
#     reader = pd.read_csv(
#         val_path,
#         usecols=['content', 'type'],
#         chunksize=chunksize,
#         low_memory=False
#     )
#     with Pool(n_workers) as pool:
#         total_rows = 0
#         y_full = []
#         y_pred_full = []
#         for i, (X, y) in enumerate(
#             pool.imap(Vec.vectorize_chunk, reader, chunksize=10),
#             1
#         ):
#             if X.shape[0] == 0:
#                 continue
#             total_rows += len(y)
#             # print(f'\nMean value of X: {X.mean()}')
#             y_pred = clf.predict(X)

#             y_full.append(y.to_numpy())
#             y_pred_full.append(y_pred)

#             print(f'Vectorized and predicted {i:,} chunks = {total_rows:,} rows')

#     print('\nFinished predicting all articles')
#     y_full = np.concat(y_full)
#     y_pred_full = np.concat(y_pred_full)

#     print('Computing F1-score')
#     print(y_full.shape, y_pred_full.shape)
#     assert y_full.shape == y_pred_full.shape
#     print(np.unique(y_full))
#     print(np.unique(y_pred_full))
#     print(y_full[:10])
#     print(y_pred_full[:10])
#     f1 = f1_score(y_full, y_pred_full)
#     report = classification_report(y_full, y_pred_full)

#     print(f"\nValidation F1: {f1:.4f}")
#     print(report)

#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(f"Validation F1: {f1:.4f}\n\n")
#         f.write(f"Val path: {val_path}\n")
#         f.write(f"Chunk size: {chunksize}\n")
#         f.write("TF-IDF config:\n")
#         f.write("  sublinear_tf=True\n")
#         f.write("  smooth_idf=True\n")
#         f.write("  stop_words=None\n")
#         f.write(report)
#     print(f"Saved results to {output_path}")


# if __name__ == '__main__':
#     predict()