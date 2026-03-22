from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from multiprocessing import Pool, cpu_count
import joblib
import pickle

StartPath = Path.cwd().parents[1]
data_dir = StartPath / 'data' / 'big_dataset'

TRAIN_PATH = data_dir / 'big_preprocessed_split' / "train.csv"
VAL_PATH = data_dir / 'big_preprocessed_split' / "val.csv"
MODEL_PATH = data_dir / 'models' / 'advanced_model.joblib'
RESULTS_PATH = data_dir / '/results' / 'advanced_model_metrics.txt'

SAMPLE_SIZE = 6821441
MAX_FEATURES = 30000
MIN_DF = 5
NGRAM_RANGE = (1, 2)
C_VALUE = 0.5
CHUNK_SIZE = 10000
N_WORKERS = max(cpu_count() - 1, 1)

FAKE_LABELS = {
    "fake", "conspiracy", "hate", "junksci", "unreliable", "bias", "satire", "political", "clickbait"
}

REAL_LABELS = {
    "reliable"
}


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

    df = df.dropna(subset=[text_col, label_col]).copy()
    df["label"] = df[label_col].apply(map_label)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df


def load_data_chunked(file_path, sample_size=None, text_col="content", label_col="type"):
    """Load CSV in chunks to avoid memory issues with large files."""
    print(f"Loading data from {file_path} in chunks...")

    chunks = []
    total_rows = 0

    reader = pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )
    with Pool(N_WORKERS) as pool:
        for i, chunk in enumerate(pool.imap(prepare_df, reader, chunksize=1), 1):
            chunks.append(chunk)
            total_rows += len(chunk)

            if sample_size and total_rows >= sample_size:
                print(f"Reached sample size limit: {total_rows} rows")
                break

            if len(chunks) % 10 == 0:
                print(f"  Loaded {total_rows:,} rows so far...")

    print(f"Concatenating {len(chunks)} chunks...")
    df = pd.concat(chunks, ignore_index=True)

    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size} rows from {len(df)} total...")
        df = df.sample(n=sample_size, random_state=42)

    print(f"Final dataset: {len(df):,} rows")
    return df


def main():
    # Load data in chunks to handle large files
    train_df = load_data_chunked(
        TRAIN_PATH,
        sample_size=SAMPLE_SIZE,
        text_col="content",
        label_col="type"
    )

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts())
    print(train_df["label"].value_counts(normalize=True))

    print("\nBuilding TF-IDF + LinearSVC pipeline...")
    # model = Pipeline([
    #     ("tfidf", TfidfVectorizer(
    #         max_features=MAX_FEATURES,
    #         min_df=MIN_DF,
    #         ngram_range=NGRAM_RANGE,
    #         sublinear_tf=True,
    #         smooth_idf=True
    #     )),
    #     ("clf", LinearSVC(C=C_VALUE, random_state=42, max_iter=2000, verbose=1))
    # ])
    tf_idf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        smooth_idf=True,
        stop_words='english'
    )
    clr = LinearSVC(C=C_VALUE, random_state=42, max_iter=2000, verbose=1)

    max_sample_size = 250000

    # Sampling to reduce memory overhead when fitting the tf-idf vectorizer
    # Ensure all labels are represented proportionally
    print('\nSampling training data frame to fit tf-idf vectorizer')
    train_df = train_df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(
            n=int(min(len(x), max_sample_size * len(x)/len(df))),
            random_state=42
        )
    )



    print(f"Sampled {len(train_df)} articles for TF-IDF fitting.")


    tf_idf.fit(train_df['content'])
    with open(data_dir / 'models' / 'tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(tf_idf, file)

    del sampled_df
    del sample_corpus

    print("\nFitting model...")
    X_train = tf_idf.transform(train_df["content"].astype(str))
    y_train = train_df['label']
    del train_df
    clr.fit(X_train, y_train)
    print("Model fit complete")

    del X_train
    del y_train


    # Validation set can be loaded normally if it's smaller
    # Or use chunked loading if val is also very large
    print(f"\nLoading validation data...")
    val_df = load_data_chunked(VAL_PATH, low_memory=False)
    print(f"Validation set: {len(val_df):,} rows")


    X_val = tf_idf.transform(val_df["content"].astype(str))
    y_val = val_df["label"]

    del val_df

    print("\nEvaluating on validation set...")
    y_val_pred = clr.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)
    report = classification_report(y_val, y_val_pred)

    print(f"\nValidation F1: {val_f1:.4f}")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clr, MODEL_PATH)
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