"""
Model Trainer Module
Trains logistic regression model and saves results
"""
import os
import pickle
import pandas as pd
import numpy as np
import ast
from scipy.sparse import lil_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from collections import Counter


# Label groupings
FAKE_LABELS = {'unreliable', 'hate', 'junksci', 'fake', 'satire', 'conspiracy', 'bias'}
TRUE_LABELS = {'reliable', 'political', 'state', 'clickbait'}


def load_data(data_dir, split):
    """Load train/val/test split and create binary labels."""
    print(f"Loading {split} data...")
    file_path = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(file_path, usecols=['content', 'type'])
    df = df.dropna(subset=['content', 'type'])
    
    df['label'] = df['type'].apply(
        lambda x: 0 if x in FAKE_LABELS else (1 if x in TRUE_LABELS else -1)
    )
    df = df[df['label'] != -1]
    
    print(f"  Loaded {len(df)} samples")
    return df[['content', 'label']]


def create_features(df, vocab):
    """Create bag-of-words feature matrix using sparse representation."""
    print(f"Creating features for {len(df)} documents...")
    vocab_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
    X = lil_matrix((len(df), len(vocab)), dtype=np.int32)
    
    for i, content in enumerate(df['content']):
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i}/{len(df)} documents...")
        try:
            tokens = ast.literal_eval(content) if isinstance(content, str) else []
            for word, count in Counter(tokens).items():
                if word in vocab_to_idx:
                    X[i, vocab_to_idx[word]] = count
        except:
            continue
    
    print("  Converting to CSR format...")
    return X.tocsr()


def train_model(data_dir, results_dir, top_k, model_name="logistic_model"):
    """
    Train logistic regression model and save to results directory.
    
    Args:
        data_dir: Path to data directory
        results_dir: Path to results directory
        top_k: Vocabulary size
        model_name: Name for saved model files
        
    Returns:
        dict: Results containing F1 scores and accuracy
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load vocabulary
    vocab_file = os.path.join(data_dir, f"top_{top_k}_vocab.pkl")
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary not found: {vocab_file}")
    
    print(f"Loading vocabulary from {vocab_file}...")
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    print(f"  Loaded {len(vocab)} words")
    
    # Load data
    train_df = load_data(data_dir, 'train')
    val_df = load_data(data_dir, 'val')
    test_df = load_data(data_dir, 'test')
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    
    # Create features
    print("\nCreating feature matrices...")
    X_train = create_features(train_df, vocab)
    y_train = train_df['label'].values
    
    X_val = create_features(val_df, vocab)
    y_val = val_df['label'].values
    
    X_test = create_features(test_df, vocab)
    y_test = test_df['label'].values
    
    print(f"\nFeature matrix shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    # Train model
    print("\nTraining logistic regression...")
    print(f"  Hyperparameters:")
    print(f"    - max_iter: 5000")
    print(f"    - solver: lbfgs")
    print(f"    - random_state: 42")
    
    model = LogisticRegression(max_iter=5000, random_state=42, verbose=1)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("\n" + "-"*60)
    print("VALIDATION RESULTS")
    print("-"*60)
    y_val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"Accuracy: {val_acc:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    print("\nClassification Report:")
    val_report = classification_report(y_val, y_val_pred, target_names=['FAKE', 'TRUE'])
    print(val_report)
    
    # Evaluate on test set
    print("\n" + "-"*60)
    print("TEST RESULTS")
    print("-"*60)
    y_test_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print("\nClassification Report:")
    test_report = classification_report(y_test, y_test_pred, target_names=['FAKE', 'TRUE'])
    print(test_report)
    
    # Save model to results directory
    model_file = os.path.join(results_dir, f"{model_name}.pkl")
    vocab_save_file = os.path.join(results_dir, f"{model_name}_vocab.pkl")
    
    print(f"\nSaving model to {model_file}...")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vocab_save_file, 'wb') as f:
        pickle.dump(vocab, f)
    
    print("✓ Model saved successfully!")
    
    # Save detailed results to results directory
    results_file = os.path.join(results_dir, f"{model_name}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Simple Logistic Regression Model Results\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  - Vocabulary size: {top_k}\n")
        f.write(f"  - Max iterations: 5000\n")
        f.write(f"  - Solver: lbfgs\n")
        f.write(f"  - Random state: 42\n\n")
        f.write(f"Validation Results:\n")
        f.write(f"  - Accuracy: {val_acc:.4f}\n")
        f.write(f"  - F1 Score: {val_f1:.4f}\n\n")
        f.write(f"Validation Classification Report:\n")
        f.write(val_report)
        f.write(f"\n\nTest Results:\n")
        f.write(f"  - Accuracy: {test_acc:.4f}\n")
        f.write(f"  - F1 Score: {test_f1:.4f}\n\n")
        f.write(f"Test Classification Report:\n")
        f.write(test_report)
    
    print(f"✓ Results saved to {results_file}")
    
    # Save results as CSV for easy analysis
    results_csv = os.path.join(results_dir, f"{model_name}_results.csv")
    results_df = pd.DataFrame({
        'metric': ['val_accuracy', 'val_f1', 'test_accuracy', 'test_f1'],
        'value': [val_acc, val_f1, test_acc, test_f1]
    })
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Results CSV saved to {results_csv}")
    
    return {
        'val_f1': val_f1,
        'val_acc': val_acc,
        'test_f1': test_f1,
        'test_acc': test_acc
    }
