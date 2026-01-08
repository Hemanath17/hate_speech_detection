"""
ETHOS Dataset Loader for Hate Speech Detection
Handles the larger ETHOS binary dataset and converts to multi-label format
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class EthosDataLoader:
    """Load and process ETHOS dataset"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.binary_path = config['ethos_binary_path']
        self.multilabel_path = config['ethos_multilabel_path']
        
    def load_binary_dataset(self) -> pd.DataFrame:
        """Load the binary ETHOS dataset (998 samples)"""
        logger.info(f"Loading binary ETHOS dataset from {self.binary_path}")
        
        try:
            df = pd.read_csv(self.binary_path, sep=';')
            logger.info(f"Loaded {len(df)} samples from binary dataset")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Check data distribution
            hate_count = df['isHate'].sum()
            non_hate_count = len(df) - hate_count
            logger.info(f"Hate speech samples: {hate_count}")
            logger.info(f"Non-hate speech samples: {non_hate_count}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading binary dataset: {e}")
            raise
    
    def load_multilabel_dataset(self) -> pd.DataFrame:
        """Load the multi-label ETHOS dataset (433 samples)"""
        logger.info(f"Loading multi-label ETHOS dataset from {self.multilabel_path}")
        
        try:
            df = pd.read_csv(self.multilabel_path, sep=';')
            logger.info(f"Loaded {len(df)} samples from multi-label dataset")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading multi-label dataset: {e}")
            raise
    
    def create_combined_dataset(self) -> Tuple[pd.DataFrame, List[str]]:
        """Create a combined dataset using both binary and multi-label data"""
        logger.info("Creating combined ETHOS dataset...")
        
        # Load both datasets
        binary_df = self.load_binary_dataset()
        multilabel_df = self.load_multilabel_dataset()
        
        # Create a mapping from comment to multi-label annotations
        multilabel_dict = {}
        for _, row in multilabel_df.iterrows():
            comment = row['comment']
            labels = {
                'violence': row['violence'],
                'directed_vs_generalized': row['directed_vs_generalized'],
                'gender': row['gender'],
                'race': row['race'],
                'national_origin': row['national_origin'],
                'disability': row['disability'],
                'religion': row['religion'],
                'sexual_orientation': row['sexual_orientation']
            }
            multilabel_dict[comment] = labels
        
        # Create combined dataset
        combined_data = []
        categories = ['violence', 'directed_vs_generalized', 'gender', 'race', 
                     'national_origin', 'disability', 'religion', 'sexual_orientation']
        
        for _, row in binary_df.iterrows():
            comment = row['comment']
            is_hate = row['isHate']
            
            # Initialize all labels as 0
            label_row = {cat: 0.0 for cat in categories}
            label_row['comment'] = comment
            label_row['isHate'] = is_hate
            
            # If it's hate speech and we have multi-label annotations, use them
            if is_hate == 1.0 and comment in multilabel_dict:
                multilabel_annotations = multilabel_dict[comment]
                for cat in categories:
                    label_row[cat] = multilabel_annotations[cat]
            
            combined_data.append(label_row)
        
        combined_df = pd.DataFrame(combined_data)
        logger.info(f"Created combined dataset with {len(combined_df)} samples")
        
        # Log label distribution
        for cat in categories:
            count = (combined_df[cat] > 0).sum()
            logger.info(f"{cat}: {count} samples")
        
        return combined_df, categories
    
    def create_enhanced_multilabel_dataset(self) -> Tuple[pd.DataFrame, List[str]]:
        """Create an enhanced multi-label dataset using all 998 samples"""
        logger.info("Creating enhanced multi-label dataset...")
        
        # Load binary dataset
        binary_df = self.load_binary_dataset()
        
        # Load multi-label dataset for reference
        multilabel_df = self.load_multilabel_dataset()
        
        # Create mapping for multi-label annotations
        multilabel_dict = {}
        for _, row in multilabel_df.iterrows():
            comment = row['comment']
            labels = {
                'violence': row['violence'],
                'directed_vs_generalized': row['directed_vs_generalized'],
                'gender': row['gender'],
                'race': row['race'],
                'national_origin': row['national_origin'],
                'disability': row['disability'],
                'religion': row['religion'],
                'sexual_orientation': row['sexual_orientation']
            }
            multilabel_dict[comment] = labels
        
        # Create enhanced dataset
        enhanced_data = []
        categories = ['violence', 'directed_vs_generalized', 'gender', 'race', 
                     'national_origin', 'disability', 'religion', 'sexual_orientation']
        
        for _, row in binary_df.iterrows():
            comment = row['comment']
            is_hate = row['isHate']
            
            # Initialize all labels as 0
            label_row = {cat: 0.0 for cat in categories}
            label_row['comment'] = comment
            label_row['isHate'] = is_hate
            
            # If it's hate speech and we have multi-label annotations, use them
            if is_hate == 1.0 and comment in multilabel_dict:
                multilabel_annotations = multilabel_dict[comment]
                for cat in categories:
                    label_row[cat] = multilabel_annotations[cat]
            elif is_hate == 1.0:
                # For hate speech without multi-label annotations, create generic labels
                # This is a heuristic approach - you might want to manually annotate these
                label_row['directed_vs_generalized'] = 0.5  # Default to directed
                # You could add more sophisticated heuristics here
            
            enhanced_data.append(label_row)
        
        enhanced_df = pd.DataFrame(enhanced_data)
        logger.info(f"Created enhanced dataset with {len(enhanced_df)} samples")
        
        # Log label distribution
        for cat in categories:
            count = (enhanced_df[cat] > 0).sum()
            logger.info(f"{cat}: {count} samples")
        
        return enhanced_df, categories
    
    def get_dataset_stats(self, df: pd.DataFrame, categories: List[str]) -> Dict:
        """Get comprehensive dataset statistics"""
        stats = {
            'total_samples': len(df),
            'hate_samples': (df['isHate'] == 1.0).sum(),
            'non_hate_samples': (df['isHate'] == 0.0).sum(),
            'category_distribution': {},
            'avg_text_length': df['comment'].str.len().mean(),
            'max_text_length': df['comment'].str.len().max(),
            'min_text_length': df['comment'].str.len().min()
        }
        
        for cat in categories:
            stats['category_distribution'][cat] = (df[cat] > 0).sum()
        
        return stats
