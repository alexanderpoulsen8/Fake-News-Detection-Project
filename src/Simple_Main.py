"""
Simple Model Pipeline - Clean Orchestrator

This script orchestrates the entire pipeline for training a simple logistic regression model:
1. Split preprocessed data into train/val/test
2. Build vocabulary from training data
3. Train model and save results to results folder

Usage:
    Edit the configuration variables at the top of main() and run:
    python Simple_Main.py
"""

import os
import sys
from pathlib import Path

# Import pipeline modules
from pipeline.data_splitter import split_data
from pipeline.vocab_builder import build_vocabulary
from pipeline.model_trainer import train_model


def main():
    # ============================================================
    # CONFIGURATION - Edit these values
    # ============================================================
    DATA_DIR = Path.cwd().parents[0] / 'data' / 'LIAR'
    RESULTS_DIR = DATA_DIR / "results"
    TOP_K_WORDS = 10000
    MODEL_NAME = "logistic_model"

    print("\n" + "="*60)
    print("SIMPLE MODEL PIPELINE")
    print("="*60)
    print(f"Data directory:    {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Top K words:       {TOP_K_WORDS}")
    print(f"Model name:        {MODEL_NAME}")
    print("="*60)

    # Verify data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    # Check if preprocessed data exists
    preprocessed_file = os.path.join(DATA_DIR, "preprocessed_dataset.csv")
    if not os.path.exists(preprocessed_file):
        print(f"❌ Error: Preprocessed dataset not found: {preprocessed_file}")
        print("Please run preprocessing first!")
        sys.exit(1)

    try:
        # Step 1: Split data (if not already split)
        print("\n" + "="*60)
        print("STEP 1: Data Splitting")
        print("="*60)
        split_data(DATA_DIR)

        # Step 2: Build vocabulary (if not already built)
        print("\n" + "="*60)
        print(f"STEP 2: Building Vocabulary (top {TOP_K_WORDS} words)")
        print("="*60)
        build_vocabulary(DATA_DIR, TOP_K_WORDS)

        # Step 3: Train model and save to results folder
        print("\n" + "="*60)
        print("STEP 3: Training Model")
        print("="*60)
        results = train_model(DATA_DIR, RESULTS_DIR, TOP_K_WORDS, MODEL_NAME)

        # Final summary
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETE!")
        print("="*60)
        print(f"Validation F1: {results['val_f1']:.4f}")
        print(f"Test F1:       {results['test_f1']:.4f}")
        print(f"Test Accuracy: {results['test_acc']:.4f}")
        print(f"\nResults saved to: {RESULTS_DIR}")
        print(f"  - {MODEL_NAME}.pkl")
        print(f"  - {MODEL_NAME}_vocab.pkl")
        print(f"  - {MODEL_NAME}_results.txt")
        print(f"  - {MODEL_NAME}_results.csv")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
