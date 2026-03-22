from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, classification_report
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
# from multiprocessing import Pool, cpu_count
import joblib

StartPath = Path.cwd().parents[1]
data_dir = StartPath / 'data' / 'big_dataset' / 'big_preprocessed_split'

TRAIN_PATH = data_dir / "train.csv"
VAL_PATH = data_dir / "val.csv"
MODEL_PATH = data_dir / 'models' / 'advanced_model.joblib'
RESULTS_PATH = data_dir / '/results' / 'advanced_model_metrics.txt'

SAMPLE_SIZE = 6821441
MAX_FEATURES = 30000
MIN_DF = 5
NGRAM_RANGE = (1, 2)
C_VALUE = 0.5
CHUNK_SIZE = 10000
# N_WORKERS = max(cpu_count() - 1, 1)

FAKE_LABELS = {
    "fake", "conspiracy", "hate", "junksci", "unreliable", "bias", "satire", "political", "clickbait"
}

REAL_LABELS = {
    "reliable"
}

VECTORIZER=HashingVectorizer(
    n_features=2**20,   # tune this
    alternate_sign=False
)


def map_label(label):
    if pd.isna(label):
        return None
    label = str(label).strip().lower()
    if label in FAKE_LABELS:
        return 1
    if label in REAL_LABELS:
        return 0
    return None


def prepare_df(df):
    text_col = "content"
    label_col = "type"
    required_cols = [text_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_prepared = df.dropna(subset=[text_col, label_col])
    df_prepared["label"] = df_prepared[label_col].apply(map_label)
    df_prepared = df_prepared.dropna(subset=["label"])
    df_prepared["label"] = df_prepared["label"].astype(int)
    return df_prepared

def prepare_and_transform(df):
    df_prepared = prepare_df(df)
    y = df_prepared['label']
    X_vec = VECTORIZER.transform(df_prepared['content'].astype(str))
    return (X_vec, y)


def main():
    model = SGDClassifier(
        loss="hinge",        # same objective as LinearSVC
        penalty="l2",
        alpha=1e-4,          # tune this!
        max_iter=1,          # important for partial_fit loop
        warm_start=True
    )



    reader = pd.read_csv(
        TRAIN_PATH,
        usecols=['content', 'type'],
        chunksize=CHUNK_SIZE,
        low_memory=False
    )
    total_rows = 0
    print(f"Loading data from {TRAIN_PATH} in chunks and doing partial fits of SGDClassifier...")
    for i, chunk in enumerate(reader, 1):
        X_vec, y_chunk = prepare_and_transform(chunk)
        total_rows += len(y_chunk)
        model.partial_fit(X_vec, y_chunk, [0,1])

        # if sample_size and total_rows >= sample_size:
        #     print(f"Reached sample size limit: {total_rows} rows")
        #     break

        if i % 10 == 0:
            print(f"  Fitted {i:,} chunks and {total_rows:,} rows so far...")
    print("Model fit complete")

    # Validation set can be loaded normally if it's smaller
    # Or use chunked loading if val is also very large

    print(f"\nLoading validation data...")
    val_df = pd.read_csv(VAL_PATH, low_memory=False)
    val_df = prepare_df(val_df, text_col="content", label_col="type")
    X_val = val_df["content"].astype(str)
    y_val = val_df["label"]
    print(f"Validation set: {len(val_df):,} rows")

    # print("\nFitting model...")
    # start = 0
    # while start < len(train_df):
    #     chunk = train_df[start:start + CHUNK_SIZE]
    #     X_vec = vectorizer.transform(chunk['content'].astype(str))
    #     y_chunk = chunk['label']
    #     clf.partial_fit(X_vec, y_chunk, [0,1])
    #     start += CHUNK_SIZE
    #     print(f'Fitted on {start:,} rows')
    # model.fit(X_train, y_train)


    print("\nEvaluating on validation set...")
    y_val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)
    report = classification_report(y_val, y_val_pred)

    print(f"\nValidation F1: {val_f1:.4f}")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Validation F1: {val_f1:.4f}\n\n")
        f.write(f"Train path: {TRAIN_PATH}\n")
        f.write(f"Val path: {VAL_PATH}\n")
        f.write(f"Sample size: {SAMPLE_SIZE}\n")
        f.write(f"Chunk size: {CHUNK_SIZE}\n")
        f.write("TF-IDF config:\n")
        f.write(f"  max_features={MAX_FEATURES}\n")
        f.write(f"  min_df={MIN_DF}\n")
        f.write(f"  ngram_range={NGRAM_RANGE}\n")
        f.write("  sublinear_tf=True\n")
        f.write("  smooth_idf=True\n")
        f.write("  stop_words='english'\n")
        f.write("LinearSVC config:\n")
        f.write(f"  C={C_VALUE}\n\n")
        f.write(report)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()