import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Ensure project root is on path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

from src.data.data_collector import HateSpeechDataCollector
from src.data.data_augmenter import AdvancedDataAugmenter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MasterDataPipeline:
    """
    Master pipeline that combines data collection and augmentation
    to create a comprehensive dataset for reaching 0.89 accuracy target.
    """
    
    def __init__(self, target_samples_per_language: int = 5000):
        self.target_samples_per_language = target_samples_per_language
        self.data_dir = Path("Data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.collector = HateSpeechDataCollector()
        self.augmenter = AdvancedDataAugmenter(target_samples_per_language)
    
    def run_complete_pipeline(self) -> pd.DataFrame:
        """
        Run the complete data expansion pipeline.
        
        Returns:
            Final expanded dataset
        """
        logger.info("=" * 80)
        logger.info("MASTER DATA EXPANSION PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Load existing data
        logger.info("Step 1: Loading existing data...")
        existing_data_path = self.data_dir / "improved_multilingual_hate_speech_v2.csv"
        
        if existing_data_path.exists():
            existing_df = pd.read_csv(existing_data_path)
            logger.info(f"Loaded existing data: {len(existing_df)} samples")
        else:
            logger.warning("No existing data found, starting from scratch")
            existing_df = pd.DataFrame()
        
        # Step 2: Collect additional datasets
        logger.info("Step 2: Collecting additional datasets...")
        try:
            collected_df = self.collector.collect_all_datasets()
            if len(collected_df) > 0:
                self.collector.save_collected_data(collected_df)
                logger.info(f"Collected additional data: {len(collected_df)} samples")
            else:
                logger.warning("No additional data collected")
                collected_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            collected_df = pd.DataFrame()
        
        # Step 3: Combine existing and collected data
        logger.info("Step 3: Combining datasets...")
        if len(existing_df) > 0 and len(collected_df) > 0:
            combined_df = pd.concat([existing_df, collected_df], ignore_index=True)
        elif len(existing_df) > 0:
            combined_df = existing_df
        elif len(collected_df) > 0:
            combined_df = collected_df
        else:
            logger.error("No data available for processing")
            return pd.DataFrame()
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['text'])
        logger.info(f"Combined dataset: {len(combined_df)} samples")
        
        # Step 4: Augment data
        logger.info("Step 4: Augmenting data...")
        try:
            final_df = self.augmenter.augment_existing_data(combined_df)
            logger.info(f"Augmented dataset: {len(final_df)} samples")
        except Exception as e:
            logger.error(f"Data augmentation failed: {e}")
            final_df = combined_df
        
        # Step 5: Save final dataset
        logger.info("Step 5: Saving final dataset...")
        final_path = self.data_dir / "expanded_multilingual_hate_speech.csv"
        final_df.to_csv(final_path, index=False)
        
        # Save statistics
        self._save_pipeline_stats(final_df)
        
        logger.info("=" * 80)
        logger.info("MASTER PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Final dataset: {len(final_df)} samples")
        logger.info(f"Hate speech ratio: {final_df['isHate'].mean():.2%}")
        logger.info(f"Languages: {final_df['original_language'].value_counts().to_dict()}")
        logger.info(f"Sources: {final_df['source'].value_counts().to_dict()}")
        
        return final_df
    
    def _save_pipeline_stats(self, df: pd.DataFrame):
        """Save pipeline statistics"""
        stats = {
            'total_samples': len(df),
            'hate_speech_ratio': float(df['isHate'].mean()),
            'languages': df['original_language'].value_counts().to_dict(),
            'sources': df['source'].value_counts().to_dict(),
            'target_samples_per_language': self.target_samples_per_language
        }
        
        stats_path = self.data_dir / "pipeline_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Pipeline statistics saved to {stats_path}")
    
    def analyze_dataset_balance(self, df: pd.DataFrame) -> Dict:
        """
        Analyze dataset balance across languages and classes.
        
        Args:
            df: Dataset to analyze
            
        Returns:
            Analysis results
        """
        logger.info("Analyzing dataset balance...")
        
        analysis = {}
        
        for language in ['eng', 'tam', 'hin', 'spa', 'cmn']:
            lang_data = df[df['original_language'] == language]
            if len(lang_data) > 0:
                hate_count = lang_data['isHate'].sum()
                neutral_count = len(lang_data) - hate_count
                hate_ratio = hate_count / len(lang_data)
                
                analysis[language] = {
                    'total_samples': len(lang_data),
                    'hate_samples': hate_count,
                    'neutral_samples': neutral_count,
                    'hate_ratio': hate_ratio,
                    'meets_target': len(lang_data) >= self.target_samples_per_language
                }
                
                logger.info(f"{language}: {len(lang_data)} samples, "
                           f"{hate_count} hate ({hate_ratio:.1%}), "
                           f"Target met: {len(lang_data) >= self.target_samples_per_language}")
        
        return analysis


def main():
    """Main function to run the master pipeline"""
    # Initialize pipeline
    pipeline = MasterDataPipeline(target_samples_per_language=5000)
    
    # Run complete pipeline
    final_dataset = pipeline.run_complete_pipeline()
    
    if len(final_dataset) > 0:
        # Analyze results
        analysis = pipeline.analyze_dataset_balance(final_dataset)
        
        # Show sample data
        logger.info("\nSample of final dataset:")
        print(final_dataset.head(10))
        
        logger.info("\nDataset analysis complete!")
    else:
        logger.error("Pipeline failed to produce any data")


if __name__ == "__main__":
    main()
