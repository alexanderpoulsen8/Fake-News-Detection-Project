from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)

TRAIN_PATH = Path("data/train.csv")
VAL_PATH = Path("data/val.csv")
TEST_PATH = Path("data/test.csv")
MODEL_PATH = Path("data/models/SGDClassifier.joblib")
RESULTS_PATH = Path("data/results/advanced_model_full_evaluation.txt")

FAKE_LABELS = {
    "fake", "conspiracy", "hate", "junksci", "clickbait",
    "unreliable", "bias", "satire", "political", "unknown"
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


def evaluate_split(model, df, split_name, text_col="content"):
    X = df[text_col].astype(str)
    y = df["label"]

    y_pred = model.predict(X)

    metrics = {
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred),
        "report": classification_report(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "support": len(df),
    }
    return metrics


def diagnose_fit(train_f1, val_f1, test_f1):
    diagnosis = []

    train_val_gap = train_f1 - val_f1
    train_test_gap = train_f1 - test_f1

    if train_f1 < 0.85 and val_f1 < 0.85 and test_f1 < 0.85:
        diagnosis.append(
            "Possible underfitting: train, validation, and test F1 are all relatively low."
        )

    if train_val_gap > 0.03 or train_test_gap > 0.03:
        diagnosis.append(
            "Possible overfitting: training F1 is noticeably higher than validation/test F1."
        )

    if abs(train_f1 - val_f1) <= 0.02 and abs(train_f1 - test_f1) <= 0.02:
        diagnosis.append(
            "No strong sign of classic overfitting: train, validation, and test F1 are fairly close."
        )

    if not diagnosis:
        diagnosis.append(
            "No obvious simple diagnosis from F1 gaps alone. Check class-wise precision/recall and confusion matrices."
        )

    return diagnosis


def format_split_output(split_name, metrics):
    cm = metrics["confusion_matrix"]
    return (
        f"{split_name} RESULTS\n"
        f"{'-' * 60}\n"
        f"Rows: {metrics['support']}\n"
        f"F1: {metrics['f1']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n\n"
        f"Classification report:\n{metrics['report']}\n"
        f"Confusion matrix:\n{cm}\n\n"
    )


def main():
    print("Loading data...")
    train_df = prepare_df(pd.read_csv(TRAIN_PATH, low_memory=False))
    val_df = prepare_df(pd.read_csv(VAL_PATH, low_memory=False))
    test_df = prepare_df(pd.read_csv(TEST_PATH, low_memory=False))

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Evaluating TRAIN...")
    train_metrics = evaluate_split(model, train_df, "TRAIN")

    print("Evaluating VAL...")
    val_metrics = evaluate_split(model, val_df, "VAL")

    print("Evaluating TEST...")
    test_metrics = evaluate_split(model, test_df, "TEST")

    diagnosis = diagnose_fit(
        train_metrics["f1"],
        val_metrics["f1"],
        test_metrics["f1"],
    )

    print("\n" + format_split_output("TRAIN", train_metrics))
    print(format_split_output("VAL", val_metrics))
    print(format_split_output("TEST", test_metrics))

    print("FIT DIAGNOSIS")
    print("-" * 60)
    for line in diagnosis:
        print(f"- {line}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("ADVANCED MODEL FULL EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        f.write(f"Train path: {TRAIN_PATH}\n")
        f.write(f"Val path: {VAL_PATH}\n")
        f.write(f"Test path: {TEST_PATH}\n\n")

        f.write(format_split_output("TRAIN", train_metrics))
        f.write(format_split_output("VAL", val_metrics))
        f.write(format_split_output("TEST", test_metrics))

        f.write("FIT DIAGNOSIS\n")
        f.write("-" * 60 + "\n")
        for line in diagnosis:
            f.write(f"- {line}\n")

    print(f"\nSaved full evaluation to {RESULTS_PATH}")


if __name__ == "__main__":
    main()