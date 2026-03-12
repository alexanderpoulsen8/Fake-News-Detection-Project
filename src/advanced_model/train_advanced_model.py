from pathlib import Path
import pandas as pd

from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib


TRAIN_PATH = Path("data/train.csv")
VAL_PATH = Path("data/val.csv")
MODEL_PATH = Path("data/models/advanced_model.joblib")
RESULTS_PATH = Path("data/results/advanced_model_metrics.txt")

SAMPLE_SIZE = 100000  # set to None to use all of train.csv

FAKE_LABELS = {
    "fake", "conspiracy", "hate", "junksci", "clickbait", "unreliable", "bias", "satire", "political"
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


def prepare_df(df, text_col="content", label_col="type"):
    required_cols = [text_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[text_col, label_col]).copy()
    df["label"] = df[label_col].apply(map_label)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df


def main():
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    val_df = pd.read_csv(VAL_PATH, low_memory=False)

    text_col = "content"
    label_col = "type"

    train_df = prepare_df(train_df, text_col=text_col, label_col=label_col)
    val_df = prepare_df(val_df, text_col=text_col, label_col=label_col)

    if SAMPLE_SIZE is not None:
        sample_n = min(SAMPLE_SIZE, len(train_df))
        train_df = train_df.sample(n=sample_n, random_state=42)
        print(f"Using sample of {sample_n} rows from training set")

    print("Training label distribution:")
    print(train_df["label"].value_counts())
    print(train_df["label"].value_counts(normalize=True))

    X_train = train_df[text_col].astype(str)
    y_train = train_df["label"]

    X_val = val_df[text_col].astype(str)
    y_val = val_df["label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            min_df=5,
            ngram_range=(1, 2),
            sublinear_tf=True,
            smooth_idf=True,
            stop_words="english"
        )),
        ("clf", LinearSVC(C=0.5, random_state=42))
    ])

    print("Fitting model...")
    model.fit(X_train, y_train)
    print("Model fit complete")

    y_val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)
    report = classification_report(y_val, y_val_pred)

    print(f"Validation F1: {val_f1:.4f}")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Validation F1: {val_f1:.4f}\n\n")
        f.write(f"Train path: {TRAIN_PATH}\n")
        f.write(f"Val path: {VAL_PATH}\n")
        f.write(f"Sample size: {SAMPLE_SIZE}\n")
        f.write("TF-IDF config:\n")
        f.write("  max_features=30000\n")
        f.write("  min_df=5\n")
        f.write("  ngram_range=(1, 2)\n")
        f.write("  sublinear_tf=True\n")
        f.write("  smooth_idf=True\n")
        f.write("  stop_words='english'\n")
        f.write("LinearSVC config:\n")
        f.write("  C=0.5\n\n")
        f.write(report)


if __name__ == "__main__":
    main()