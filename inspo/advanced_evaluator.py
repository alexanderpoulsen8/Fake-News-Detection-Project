"""
Advanced Model Evaluator Module
Evaluates trained model on train/val/test sets
"""
import os
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


FAKE_LABELS = {
    "fake", "conspiracy", "hate", "junksci", "clickbait",
    "unreliable", "bias", "satire", "political"
}

REAL_LABELS = {
    "reliable"
}


def map_label(label):
    """Map article type to binary label (0=real, 1=fake)."""
    if pd.isna(label):
        return None
    label = str(label).strip().lower()
    if label in FAKE_LABELS:
        return 1
    if label in REAL_LABELS:
        return 0
    return None


def prepare_df(df, text_col="content", label_col="type"):
    """Prepare dataframe with binary labels."""
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
    """Evaluate model on a data split."""
    X = df[text_col].astype(str)
    y = df["label"]

    y_pred = model.predict(X)

    metrics = {
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred),
        "report": classification_report(y, y_pred, target_names=['REAL', 'FAKE']),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "support": len(df),
    }
    return metrics


def diagnose_fit(train_f1, val_f1, test_f1):
    """Diagnose potential overfitting/underfitting."""
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
            "No strong sign of overfitting: train, validation, and test F1 are fairly close."
        )

    if not diagnosis:
        diagnosis.append(
            "No obvious diagnosis from F1 gaps. Check class-wise precision/recall and confusion matrices."
        )

    return diagnosis


def format_split_output(split_name, metrics):
    """Format metrics output for a split."""
    cm = metrics["confusion_matrix"]
    return (
        f"{split_name} RESULTS\n"
        f"{'-' * 60}\n"
        f"Samples: {metrics['support']}\n"
        f"F1:        {metrics['f1']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall:    {metrics['recall']:.4f}\n"
        f"Accuracy:  {metrics['accuracy']:.4f}\n\n"
        f"Classification Report:\n{metrics['report']}\n"
        f"Confusion Matrix:\n{cm}\n\n"
    )


def evaluate_advanced_model(data_dir, results_dir, model_name="advanced_model"):
    """
    Evaluate trained advanced model on all splits.
    
    Args:
        data_dir: Path to data directory containing train/val/test.csv
        results_dir: Path to results directory containing trained model
        model_name: Name of the model to evaluate
        
    Returns:
        dict: Evaluation metrics for all splits
    """
    # Load data
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    print("Loading data...")
    train_df = prepare_df(pd.read_csv(train_path, low_memory=False))
    val_df = prepare_df(pd.read_csv(val_path, low_memory=False))
    test_df = prepare_df(pd.read_csv(test_path, low_memory=False))

    # Load model
    model_path = os.path.join(results_dir, f"{model_name}.joblib")
    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)

    # Evaluate on all splits
    print("\nEvaluating on TRAIN split...")
    train_metrics = evaluate_split(model, train_df, "TRAIN")

    print("Evaluating on VAL split...")
    val_metrics = evaluate_split(model, val_df, "VAL")

    print("Evaluating on TEST split...")
    test_metrics = evaluate_split(model, test_df, "TEST")

    # Diagnose fit
    diagnosis = diagnose_fit(
        train_metrics["f1"],
        val_metrics["f1"],
        test_metrics["f1"],
    )

    # Print results
    print("\n" + "="*60)
    print(format_split_output("TRAIN", train_metrics))
    print(format_split_output("VAL", val_metrics))
    print(format_split_output("TEST", test_metrics))

    print("FIT DIAGNOSIS")
    print("-" * 60)
    for line in diagnosis:
        print(f"- {line}")
    print("="*60)

    # Save full evaluation
    eval_path = os.path.join(results_dir, f"{model_name}_full_evaluation.txt")
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write("ADVANCED MODEL FULL EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Train path: {train_path}\n")
        f.write(f"Val path: {val_path}\n")
        f.write(f"Test path: {test_path}\n\n")

        f.write(format_split_output("TRAIN", train_metrics))
        f.write(format_split_output("VAL", val_metrics))
        f.write(format_split_output("TEST", test_metrics))

        f.write("FIT DIAGNOSIS\n")
        f.write("-" * 60 + "\n")
        for line in diagnosis:
            f.write(f"- {line}\n")

    print(f"\n✓ Full evaluation saved to {eval_path}")
    
    # Save CSV summary
    csv_path = os.path.join(results_dir, f"{model_name}_evaluation.csv")
    eval_df = pd.DataFrame({
        'split': ['train', 'val', 'test'],
        'f1': [train_metrics['f1'], val_metrics['f1'], test_metrics['f1']],
        'precision': [train_metrics['precision'], val_metrics['precision'], test_metrics['precision']],
        'recall': [train_metrics['recall'], val_metrics['recall'], test_metrics['recall']],
        'accuracy': [train_metrics['accuracy'], val_metrics['accuracy'], test_metrics['accuracy']],
        'samples': [train_metrics['support'], val_metrics['support'], test_metrics['support']]
    })
    eval_df.to_csv(csv_path, index=False)
    print(f"✓ Evaluation CSV saved to {csv_path}")

    return {
        'train_f1': train_metrics['f1'],
        'val_f1': val_metrics['f1'],
        'test_f1': test_metrics['f1'],
        'test_accuracy': test_metrics['accuracy'],
        'diagnosis': diagnosis
    }
