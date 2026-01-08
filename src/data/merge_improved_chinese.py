import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Ensure project root is on path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_improved_chinese_dataset():
    """
    Merge improved Chinese dataset with existing expanded dataset.
    Replace old Chinese samples with improved ones.
    """
    logger.info("=" * 80)
    logger.info("MERGING IMPROVED CHINESE DATASET")
    logger.info("=" * 80)
    
    # Load improved Chinese dataset
    improved_chinese_path = 'Data/improved_chinese_hate_speech.csv'
    if not os.path.exists(improved_chinese_path):
        logger.error(f"Improved Chinese dataset not found: {improved_chinese_path}")
        return None
    
    improved_chinese = pd.read_csv(improved_chinese_path)
    logger.info(f"Loaded improved Chinese dataset: {len(improved_chinese)} samples")
    logger.info(f"  Hate: {improved_chinese['isHate'].sum()} ({improved_chinese['isHate'].mean()*100:.1f}%)")
    logger.info(f"  Neutral: {(improved_chinese['isHate'] == 0).sum()} ({(improved_chinese['isHate'] == 0).mean()*100:.1f}%)")
    
    # Load existing expanded dataset
    expanded_path = 'Data/expanded_multilingual_hate_speech.csv'
    if not os.path.exists(expanded_path):
        logger.error(f"Expanded dataset not found: {expanded_path}")
        return None
    
    expanded_df = pd.read_csv(expanded_path)
    logger.info(f"\nLoaded expanded dataset: {len(expanded_df)} samples")
    
    # Show original Chinese statistics
    original_chinese = expanded_df[expanded_df['original_language'] == 'cmn']
    logger.info(f"Original Chinese samples: {len(original_chinese)}")
    logger.info(f"  Hate: {original_chinese['isHate'].sum()} ({original_chinese['isHate'].mean()*100:.1f}%)")
    logger.info(f"  Neutral: {(original_chinese['isHate'] == 0).sum()} ({(original_chinese['isHate'] == 0).mean()*100:.1f}%)")
    
    # Remove old Chinese samples
    expanded_df_no_chinese = expanded_df[expanded_df['original_language'] != 'cmn'].copy()
    logger.info(f"\nAfter removing old Chinese: {len(expanded_df_no_chinese)} samples")
    
    # Ensure improved Chinese has same columns
    required_columns = ['text', 'isHate', 'original_language']
    if 'source' not in improved_chinese.columns:
        improved_chinese['source'] = 'improved_chinese'
    if 'confidence' not in improved_chinese.columns:
        improved_chinese['confidence'] = 1.0
    
    # Select only required columns from improved Chinese
    improved_chinese_clean = improved_chinese[required_columns + ['source', 'confidence']].copy()
    
    # Merge datasets
    merged_df = pd.concat([expanded_df_no_chinese, improved_chinese_clean], ignore_index=True)
    
    # Remove duplicates based on text
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['text'], keep='first')
    removed_count = initial_count - len(merged_df)
    logger.info(f"\nRemoved {removed_count} duplicate samples")
    
    # Shuffle
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Final statistics
    logger.info("\n" + "=" * 80)
    logger.info("MERGED DATASET STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total samples: {len(merged_df)}")
    logger.info(f"Hate speech: {merged_df['isHate'].sum()} ({merged_df['isHate'].mean()*100:.1f}%)")
    logger.info(f"Neutral: {(merged_df['isHate'] == 0).sum()} ({(merged_df['isHate'] == 0).mean()*100:.1f}%)")
    
    logger.info("\nLanguage distribution:")
    lang_dist = merged_df['original_language'].value_counts()
    for lang, count in lang_dist.items():
        lang_hate = merged_df[merged_df['original_language'] == lang]['isHate'].sum()
        lang_hate_pct = merged_df[merged_df['original_language'] == lang]['isHate'].mean() * 100
        logger.info(f"  {lang}: {count} samples, {lang_hate} hate ({lang_hate_pct:.1f}%)")
    
    # Save merged dataset
    output_path = 'Data/expanded_multilingual_hate_speech_improved_chinese.csv'
    merged_df.to_csv(output_path, index=False)
    logger.info(f"\nMerged dataset saved to: {output_path}")
    
    # Save statistics
    stats = {
        'total_samples': len(merged_df),
        'hate_speech_ratio': float(merged_df['isHate'].mean()),
        'languages': merged_df['original_language'].value_counts().to_dict(),
        'language_hate_ratios': {
            lang: float(merged_df[merged_df['original_language'] == lang]['isHate'].mean())
            for lang in merged_df['original_language'].unique()
        }
    }
    
    import json
    stats_path = 'Data/improved_chinese_merge_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to: {stats_path}")
    
    return merged_df


if __name__ == "__main__":
    merged_df = merge_improved_chinese_dataset()
    if merged_df is not None:
        logger.info("\n" + "=" * 80)
        logger.info("MERGE COMPLETE - Ready for retraining")
        logger.info("=" * 80)

