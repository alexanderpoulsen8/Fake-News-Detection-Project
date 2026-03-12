"""
Vocabulary Builder Module
Builds vocabulary of top K most frequent words from training data
"""
import os
import pickle
import pandas as pd
import ast
from collections import Counter


def build_vocabulary(data_dir, top_k):
    """
    Build vocabulary of top K most frequent words from training data.
    
    Args:
        data_dir: Path to data directory
        top_k: Number of most frequent words to include
        
    Returns:
        bool: True if vocab was built, False if skipped
    """
    vocab_file = os.path.join(data_dir, f"top_{top_k}_vocab.pkl")
    
    if os.path.exists(vocab_file):
        print(f"Vocabulary already exists: {vocab_file}")
        print("Skipping vocabulary building...")
        return False
    
    train_file = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")
    
    print(f"Reading training data from {train_file}...")
    train_df = pd.read_csv(train_file, usecols=['content'])
    train_df = train_df.dropna(subset=['content'])
    
    print(f"Counting word frequencies from {len(train_df)} documents...")
    word_counter = Counter()
    
    for i, content in enumerate(train_df['content'], 1):
        if i % 50000 == 0:
            print(f"  Processed {i}/{len(train_df)} documents...")
        try:
            tokens = ast.literal_eval(content) if isinstance(content, str) else []
            word_counter.update(tokens)
        except:
            continue
    
    print(f"\nTotal unique words: {len(word_counter):,}")
    
    # Get top K words
    top_words = [word for word, count in word_counter.most_common(top_k)]
    vocab = set(top_words)
    
    print(f"Selected top {len(vocab)} words")
    print(f"\nMost frequent words:")
    for word, count in word_counter.most_common(10):
        print(f"  {word}: {count:,}")
    
    # Save vocabulary
    print(f"\nSaving vocabulary to {vocab_file}...")
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    
    print("✓ Vocabulary built successfully!")
    return True
