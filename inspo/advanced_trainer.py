"""
Advanced Model Trainer Module
Trains TF-IDF + LinearSVC model
"""
import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib


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


def train_advanced_model(
    data_dir,
    results_dir,
    model_name="advanced_model",
    sample_size=None,
    max_features=30000,
    min_df=5,
    ngram_range=(1, 2),
    c_value=0.5
):
    """
    Train advanced TF-IDF + LinearSVC model.
    
    Args:
        data_dir: Path to data directory containing train.csv and val.csv
        results_dir: Path to results directory for saving model
        model_name: Name for saved model files
        sample_size: Optional sample size for training (None = use all)
        max_features: Maximum number of features for TF-IDF
        min_df: Minimum document frequency for TF-IDF
        ngram_range: N-gram range for TF-IDF
        c_value: Regularization parameter for LinearSVC
        
    Returns:
        dict: Training results with validation F1 score
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path, low_memory=False)
    
    print(f"Loading validation data from {val_path}...")
    val_df = pd.read_csv(val_path, low_memory=False)

    text_col = "content"
    label_col = "type"

    print("Preparing data...")
    train_df = prepare_df(train_df, text_col=text_col, label_col=label_col)
    val_df = prepare_df(val_df, text_col=text_col, label_col=label_col)

    # Optional sampling
    if sample_size is not None:
        sample_n = min(sample_size, len(train_df))
        train_df = train_df.sample(n=sample_n, random_state=42)
        print(f"Using sample of {sample_n} rows from training set")

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts())
    print(train_df["label"].value_counts(normalize=True))

    X_train = train_df[text_col].astype(str)
    y_train = train_df["label"]

    X_val = val_df[text_col].astype(str)
    y_val = val_df["label"]

    # Build model pipeline
    print("\nBuilding model pipeline...")
    print(f"  TF-IDF: max_features={max_features}, min_df={min_df}, ngram_range={ngram_range}")
    print(f"  LinearSVC: C={c_value}")
    
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
            sublinear_tf=True,
            smooth_idf=True,
            stop_words="english"
        )),
        ("clf", LinearSVC(C=c_value, random_state=42, max_iter=2000))
    ])

    print("\nFitting model...")
    model.fit(X_train, y_train)
    print("✓ Model fit complete")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)
    report = classification_report(y_val, y_val_pred, target_names=['REAL', 'FAKE'])

    print(f"\nValidation F1: {val_f1:.4f}")
    print("\nClassification Report:")
    print(report)

    # Save model
    model_path = os.path.join(results_dir, f"{model_name}.joblib")
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    print("✓ Model saved successfully")

    # Save training results
    results_path = os.path.join(results_dir, f"{model_name}_training_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Advanced Model Training Results\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Validation F1: {val_f1:.4f}\n\n")
        f.write(f"Data:\n")
        f.write(f"  Train path: {train_path}\n")
        f.write(f"  Val path: {val_path}\n")
        f.write(f"  Train samples: {len(train_df)}\n")
        f.write(f"  Val samples: {len(val_df)}\n")
        if sample_size:
            f.write(f"  Sample size: {sample_size}\n")
        f.write(f"\nTF-IDF Configuration:\n")
        f.write(f"  max_features: {max_features}\n")
        f.write(f"  min_df: {min_df}\n")
        f.write(f"  ngram_range: {ngram_range}\n")
        f.write(f"  sublinear_tf: True\n")
        f.write(f"  smooth_idf: True\n")
        f.write(f"  stop_words: 'english'\n")
        f.write(f"\nLinearSVC Configuration:\n")
        f.write(f"  C: {c_value}\n")
        f.write(f"  random_state: 42\n\n")
        f.write(f"Validation Classification Report:\n")
        f.write(report)

    print(f"✓ Training results saved to {results_path}")

    return {
        'val_f1': val_f1,
        'model_path': model_path,
        'results_path': results_path
    }
