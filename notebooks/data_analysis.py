"""
Data analysis for hate speech detection dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import HateSpeechPreprocessor
from configs.config import CATEGORIES

def analyze_dataset():
    """Analyze the multi-label hate speech dataset"""
    
    # Load dataset
    data_path = Path(__file__).parent.parent / "Data" / "Ethos_Dataset_Multi_Label.csv"
    df = pd.read_csv(data_path, sep=';')
    
    print("=== Dataset Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    # Rename columns
    df.columns = ['comment', 'violence', 'directed_vs_generalized', 'gender', 
                 'race', 'national_origin', 'disability', 'religion', 'sexual_orientation']
    
    print("\n=== Label Distribution ===")
    label_columns = ['violence', 'directed_vs_generalized', 'gender', 'race', 
                    'national_origin', 'disability', 'religion', 'sexual_orientation']
    
    for col in label_columns:
        positive_count = (df[col] > 0.5).sum()
        percentage = (positive_count / len(df)) * 100
        print(f"{col}: {positive_count} ({percentage:.1f}%)")
    
    # Convert to binary labels
    for col in label_columns:
        df[col] = (df[col] > 0.5).astype(int)
    
    print("\n=== Binary Label Distribution ===")
    label_counts = df[label_columns].sum()
    print(label_counts)
    
    # Text length analysis
    print("\n=== Text Length Analysis ===")
    df['text_length'] = df['comment'].str.len()
    print(f"Mean length: {df['text_length'].mean():.1f}")
    print(f"Median length: {df['text_length'].median():.1f}")
    print(f"Min length: {df['text_length'].min()}")
    print(f"Max length: {df['text_length'].max()}")
    
    # Multi-label analysis
    print("\n=== Multi-label Analysis ===")
    df['num_labels'] = df[label_columns].sum(axis=1)
    print(f"Average labels per sample: {df['num_labels'].mean():.2f}")
    print(f"Max labels per sample: {df['num_labels'].max()}")
    print(f"Samples with 0 labels: {(df['num_labels'] == 0).sum()}")
    print(f"Samples with 1+ labels: {(df['num_labels'] > 0).sum()}")
    
    # Label co-occurrence
    print("\n=== Label Co-occurrence ===")
    cooccurrence = df[label_columns].T.dot(df[label_columns])
    print(cooccurrence)
    
    # Sample texts by category
    print("\n=== Sample Texts by Category ===")
    for col in label_columns:
        positive_samples = df[df[col] == 1]['comment'].head(3)
        print(f"\n{col.upper()}:")
        for i, text in enumerate(positive_samples, 1):
            print(f"{i}. {text[:100]}...")
    
    # Class imbalance analysis
    print("\n=== Class Imbalance Analysis ===")
    total_samples = len(df)
    for col in label_columns:
        positive = df[col].sum()
        negative = total_samples - positive
        imbalance_ratio = max(positive, negative) / min(positive, negative)
        print(f"{col}: {positive} positive, {negative} negative (ratio: {imbalance_ratio:.2f})")
    
    return df

def test_preprocessing():
    """Test the preprocessing pipeline"""
    print("\n=== Testing Preprocessing Pipeline ===")
    
    config = {'categories': CATEGORIES}
    preprocessor = HateSpeechPreprocessor(config)
    
    # Test samples
    test_samples = [
        "You should know women's sports are a joke @user #hashtag",
        "RT @user: This is so stupid!",
        "I hate all muslims and jews equally",
        "Women deserve to be abused, I guess.",
        "Why don't you go back to your country?"
    ]
    
    for i, sample in enumerate(test_samples, 1):
        cleaned = preprocessor.preprocess_text(sample)
        print(f"\nSample {i}:")
        print(f"Original: {sample}")
        print(f"Cleaned:  {cleaned}")

def test_augmentation():
    """Test the augmentation pipeline"""
    print("\n=== Testing Augmentation Pipeline ===")
    
    from src.data.augmentation import HateSpeechAugmenter
    
    config = {'categories': CATEGORIES}
    augmenter = HateSpeechAugmenter(config)
    
    # Test sample
    text = "Women are terrible at sports"
    labels = [0, 1, 0, 0, 0, 0, 0, 0]  # gender = 1
    
    augmented_samples = augmenter.augment_sample(text, labels)
    
    print(f"Original: {text}")
    for i, (aug_text, aug_labels) in enumerate(augmented_samples):
        print(f"Augmented {i}: {aug_text}")

if __name__ == "__main__":
    # Run analysis
    df = analyze_dataset()
    
    # Test preprocessing
    test_preprocessing()
    
    # Test augmentation
    test_augmentation()
    
    print("\n=== Analysis Complete ===")
