from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib


TRAIN_PATH = Path("/Volumes/Lenovo PS6/Big Splits/train.csv")
VAL_PATH = Path("/Volumes/Lenovo PS6/Big Splits/val.csv")
MODEL_PATH = Path("data/models/advanced_model_chunked.joblib")
RESULTS_PATH = Path("data/results/advanced_model_chunked_metrics.txt")

SAMPLE_SIZE = 6_821_441
CHUNK_SIZE = 50_000

N_HASH_FEATURES = 2 ** 16
NGRAM_RANGE = (1, 1)
ALPHA = 1e-4
MAX_EPOCHS = 2

FAKE_LABELS = {
    "fake", "conspiracy", "hate", "junksci", "unreliable",
    "bias", "satire", "political", "clickbait"
}
REAL_LABELS = {"reliable"}


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

    df = df.dropna(subset=[text_col, label_col])
    df[label_col] = df[label_col].map(map_label)
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype("int8")

    return df[[text_col, label_col]].rename(columns={label_col: "label"})


def compute_training_class_weights():
    print("Computing class weights from training data...")

    y_all = []
    total_rows = 0

    for chunk in pd.read_csv(
        TRAIN_PATH,
        chunksize=CHUNK_SIZE,
        usecols=["type"],
        dtype={"type": "string"},
        low_memory=False,
    ):
        chunk["label"] = chunk["type"].map(map_label)
        chunk = chunk.dropna(subset=["label"])

        if chunk.empty:
            continue

        if SAMPLE_SIZE is not None:
            remaining = SAMPLE_SIZE - total_rows
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        y = chunk["label"].astype("int8").to_numpy()
        y_all.append(y)
        total_rows += len(y)

        if total_rows % (CHUNK_SIZE * 10) == 0:
            print(f"  Counted labels for {total_rows:,} rows")

        if SAMPLE_SIZE is not None and total_rows >= SAMPLE_SIZE:
            break

    y_all = np.concatenate(y_all)
    classes = np.array([0, 1], dtype=np.int8)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_all
    )

    class_weight_dict = {0: float(weights[0]), 1: float(weights[1])}
    print(f"Computed class weights: {class_weight_dict}")

    return class_weight_dict


def chunked_train():
    print(f"Streaming training from {TRAIN_PATH}")
    print(f"Sample size: {SAMPLE_SIZE:,}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print(f"Hash features: {N_HASH_FEATURES:,}")
    print(f"Ngram range: {NGRAM_RANGE}")
    print(f"Alpha: {ALPHA}")
    print(f"Epochs: {MAX_EPOCHS}")

    class_weight_dict = compute_training_class_weights()

    hasher = HashingVectorizer(
        n_features=N_HASH_FEATURES,
        ngram_range=NGRAM_RANGE,
        alternate_sign=False,
        stop_words="english",
        norm="l2",
    )

    clf = SGDClassifier(
        loss="log_loss",
        alpha=ALPHA,
        class_weight=class_weight_dict,
        random_state=42,
    )

    classes = np.array([0, 1], dtype=np.int8)
    total_rows = 0
    first_fit = True
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
        epoch_rows = 0

        for i, chunk in enumerate(
            pd.read_csv(
                TRAIN_PATH,
                chunksize=CHUNK_SIZE,
                usecols=["content", "type"],
                dtype={"content": "string", "type": "string"},
                low_memory=False,
            ),
            start=1,
        ):
            chunk = prepare_df(chunk)

            if chunk.empty:
                continue

            if SAMPLE_SIZE is not None:
                remaining = SAMPLE_SIZE - epoch_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]

            X_text = chunk["content"].astype(str).tolist()
            y = chunk["label"].to_numpy()

            X = hasher.transform(X_text).astype(np.float32)

            if first_fit:
                clf.partial_fit(X, y, classes=classes)
                first_fit = False
            else:
                clf.partial_fit(X, y)

            epoch_rows += len(y)
            total_rows += len(y)

            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {epoch_rows:,} rows this epoch in {elapsed:.1f}s")

            del chunk, X_text, y, X

            if SAMPLE_SIZE is not None and epoch_rows >= SAMPLE_SIZE:
                print(f"Reached sample size limit this epoch: {epoch_rows:,}")
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete")
    print(f"Total rows seen across all epochs: {total_rows:,}")
    print(f"Training time: {total_time:.1f}s")

    return hasher, clf, total_rows, total_time


def chunked_evaluate(hasher, clf):
    print(f"\nStreaming validation from {VAL_PATH}")

    y_true_all = []
    y_pred_all = []
    total_val = 0
    start_time = time.time()

    for i, chunk in enumerate(
        pd.read_csv(
            VAL_PATH,
            chunksize=CHUNK_SIZE,
            usecols=["content", "type"],
            dtype={"content": "string", "type": "string"},
            low_memory=False,
        ),
        start=1,
    ):
        chunk = prepare_df(chunk)

        if chunk.empty:
            continue

        X_text = chunk["content"].astype(str).tolist()
        y_true = chunk["label"].to_numpy()

        X = hasher.transform(X_text).astype(np.float32)
        y_pred = clf.predict(X)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        total_val += len(y_true)

        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Validated {total_val:,} rows in {elapsed:.1f}s")

        del chunk, X_text, y_true, X, y_pred

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    val_f1 = f1_score(y_true_all, y_pred_all)
    report = classification_report(y_true_all, y_pred_all, digits=4)

    print(f"\nValidation rows: {total_val:,}")
    print(f"Validation F1: {val_f1:.4f}")
    print(report)

    return val_f1, report, total_val


def save_outputs(hasher, clf, val_f1, report, total_train, total_val, train_time):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"vectorizer": hasher, "clf": clf}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Validation F1: {val_f1:.4f}\n\n")
        f.write(f"Train path: {TRAIN_PATH}\n")
        f.write(f"Val path: {VAL_PATH}\n")
        f.write(f"Rows used for training: {total_train}\n")
        f.write(f"Rows used for validation: {total_val}\n")
        f.write(f"Training time (seconds): {train_time:.1f}\n")
        f.write(f"Sample size: {SAMPLE_SIZE}\n")
        f.write(f"Chunk size: {CHUNK_SIZE}\n")
        f.write(f"Hashing features: {N_HASH_FEATURES}\n")
        f.write(f"Ngram range: {NGRAM_RANGE}\n")
        f.write(f"Alpha: {ALPHA}\n")
        f.write(f"Epochs: {MAX_EPOCHS}\n\n")
        f.write(report)

    print(f"Saved results to {RESULTS_PATH}")


def main():
    hasher, clf, total_train, train_time = chunked_train()
    val_f1, report, total_val = chunked_evaluate(hasher, clf)
    save_outputs(hasher, clf, val_f1, report, total_train, total_val, train_time)


if __name__ == "__main__":
    main()