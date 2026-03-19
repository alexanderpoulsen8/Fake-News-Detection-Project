"""
Advanced Model Pipeline - Clean Orchestrator

This script orchestrates the entire pipeline for training an advanced TF-IDF + LinearSVC model:
1. Preprocess raw CSV dataset (optional - skip if already done)
2. Split preprocessed data into train/val/test
3. Train advanced model (TF-IDF + LinearSVC)
4. Evaluate model on all splits

Usage:
    Edit the configuration variables at the top of main() and run:
    python Advanced_Main.py
"""

import os
import sys

# Import pipeline modules
from pipeline.data_splitter import split_data
from pipeline.preprocess_with_duckdb import preprocess_with_duckdb

# Import advanced model scripts
from advanced_model import train_advanced_model
from advanced_model import evaluate_advanced_model


def main():
    # ============================================================
    # CONFIGURATION - Edit these values
    # ============================================================
    DATA_DIR = r"C:\Users\jespe\GDS eksamen\Fake-News-Detection-Project\data"
    RESULTS_DIR = r"C:\Users\jespe\GDS eksamen\Fake-News-Detection-Project\results"
    MODEL_NAME = "advanced_model"
    
    # Optional: Set to path of raw CSV if you need to preprocess
    # Set to None to skip preprocessing step
    RAW_CSV_PATH = None  # e.g., r"D:\GDS\Fake-News-Detection-Project\data\raw_dataset.csv"
    
    # Model hyperparameters
    SAMPLE_SIZE = None  # Set to integer to use subset for faster training, None = use all data
    MAX_FEATURES = 30000  # Maximum number of TF-IDF features
    MIN_DF = 5  # Minimum document frequency
    NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
    C_VALUE = 0.5  # SVM regularization parameter
    
    print("\n" + "="*60)
    print("ADVANCED MODEL PIPELINE")
    print("="*60)
    print(f"Data directory:    {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Model name:        {MODEL_NAME}")
    print(f"Sample size:       {SAMPLE_SIZE if SAMPLE_SIZE else 'All data'}")
    print(f"Max features:      {MAX_FEATURES}")
    print("="*60)
    
    # Verify data directory exists
    if not os.path.exists(DATA_DIR):
        print(f" Error: Data directory not found: {DATA_DIR}")
        sys.exit(1)
    
    try:
        # Step 1: Preprocess raw CSV (optional)
        if RAW_CSV_PATH:
            print("\n" + "="*60)
            print("STEP 1: Preprocessing Raw Dataset")
            print("="*60)
            print("Using parallel preprocessing script...")
            
            # Check if preprocessed file already exists
            preprocessed_file = os.path.join(DATA_DIR, "preprocessed_dataset.csv")
            if os.path.exists(preprocessed_file):
                print(f"Preprocessed file already exists: {preprocessed_file}")
                print("Skipping preprocessing...")
            else:
                # Run DuckDB-based preprocessing (memory efficient)
                preprocess_with_duckdb(
                    input_path=RAW_CSV_PATH,
                    output_path=preprocessed_file,
                    batch_size=50000  # Smaller batches for better memory management
                )
        else:
            print("\n" + "="*60)
            print("STEP 1: Preprocessing (SKIPPED)")
            print("="*60)
            print("Using existing preprocessed_dataset.csv")
            
            # Verify preprocessed file exists
            preprocessed_file = os.path.join(DATA_DIR, "preprocessed_dataset.csv")
            if not os.path.exists(preprocessed_file):
                print(f"❌ Error: Preprocessed dataset not found: {preprocessed_file}")
                print("Please set RAW_CSV_PATH to preprocess the data first!")
                sys.exit(1)
        
        # Step 2: Split data into train/val/test
        print("\n" + "="*60)
        print("STEP 2: Data Splitting")
        print("="*60)
        split_data(DATA_DIR)
        
        # Step 3: Train advanced model
        print("\n" + "="*60)
        print("STEP 3: Training Advanced Model")
        print("="*60)
        train_advanced_model.main()
        
        # Step 4: Evaluate model on all splits
        print("\n" + "="*60)
        print("STEP 4: Evaluating Model")
        print("="*60)
        evaluate_advanced_model.main()
        
        # Final summary
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {DATA_DIR}")
        print(f"  - models/{MODEL_NAME}.joblib")
        print(f"  - results/{MODEL_NAME}_metrics.txt")
        print(f"  - results/{MODEL_NAME}_full_evaluation.txt")
        print("="*60)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
