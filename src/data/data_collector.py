import os
import sys
import json
import logging
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from urllib.parse import urljoin
import zipfile
import gzip
from pathlib import Path

# Ensure project root is on path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HateSpeechDataCollector:
    """
    Comprehensive data collector for hate speech datasets from multiple sources.
    """
    
    def __init__(self, data_dir: str = "Data/collected_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset sources and their configurations
        self.dataset_sources = {
            'hatexplain': {
                'url': 'https://github.com/deepaknlp/HateXplain/raw/master/dataset/dataset.json',
                'format': 'json',
                'description': 'Multi-label hate speech dataset with explanations'
            },
            'olid': {
                'url': 'https://raw.githubusercontent.com/uds-lsv/offensive-language-identification/master/data/olid-training-v1.0.tsv',
                'format': 'tsv',
                'description': 'Offensive Language Identification Dataset'
            },
            'davidson': {
                'url': 'https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv',
                'format': 'csv',
                'description': 'Hate speech and offensive language dataset'
            }
        }
        
        # Language mapping for collected data
        self.language_mapping = {
            'en': 'eng', 'english': 'eng',
            'ta': 'tam', 'tamil': 'tam',
            'hi': 'hin', 'hindi': 'hin',
            'es': 'spa', 'spanish': 'spa',
            'zh': 'cmn', 'chinese': 'cmn'
        }
    
    def download_dataset(self, dataset_name: str, url: str, format: str) -> str:
        """
        Download a dataset from URL.
        
        Args:
            dataset_name: Name of the dataset
            url: URL to download from
            format: File format (json, csv, tsv)
            
        Returns:
            Path to downloaded file
        """
        logger.info(f"Downloading {dataset_name} from {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            file_path = self.data_dir / f"{dataset_name}.{format}"
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded {dataset_name} to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return None
    
    def process_hatexplain(self, file_path: str) -> pd.DataFrame:
        """
        Process HateXplain dataset.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Processed DataFrame
        """
        logger.info("Processing HateXplain dataset...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        
        for item in data:
            text = item['post_tokens']
            if isinstance(text, list):
                text = ' '.join(text)
            
            # Get labels
            labels = item['annotators']['label']
            hate_count = labels.count('hate')
            offensive_count = labels.count('offensive')
            normal_count = labels.count('normal')
            
            # Determine if it's hate speech (majority vote)
            if hate_count > offensive_count and hate_count > normal_count:
                is_hate = 1
            elif offensive_count > normal_count:
                is_hate = 1  # Treat offensive as hate for our purposes
            else:
                is_hate = 0
            
            processed_data.append({
                'text': text,
                'isHate': is_hate,
                'original_language': 'eng',
                'source': 'hatexplain'
            })
        
        df = pd.DataFrame(processed_data)
        logger.info(f"Processed HateXplain: {len(df)} samples, {df['isHate'].sum()} hate speech")
        return df
    
    def process_olid(self, file_path: str) -> pd.DataFrame:
        """
        Process OLID dataset.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Processed DataFrame
        """
        logger.info("Processing OLID dataset...")
        
        df = pd.read_csv(file_path, sep='\t')
        
        # Map labels: OFF -> 1 (hate/offensive), NOT -> 0 (normal)
        df['isHate'] = df['subtask_a'].map({'OFF': 1, 'NOT': 0})
        df['text'] = df['tweet']
        df['original_language'] = 'eng'
        df['source'] = 'olid'
        
        processed_df = df[['text', 'isHate', 'original_language', 'source']].copy()
        processed_df = processed_df.dropna()
        
        logger.info(f"Processed OLID: {len(processed_df)} samples, {processed_df['isHate'].sum()} hate speech")
        return processed_df
    
    def process_davidson(self, file_path: str) -> pd.DataFrame:
        """
        Process Davidson dataset.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Processed DataFrame
        """
        logger.info("Processing Davidson dataset...")
        
        df = pd.read_csv(file_path)
        
        # Map labels: 0 -> 0 (hate), 1 -> 1 (offensive), 2 -> 0 (neither)
        df['isHate'] = df['class'].map({0: 1, 1: 1, 2: 0})  # Both hate and offensive as hate
        df['text'] = df['tweet']
        df['original_language'] = 'eng'
        df['source'] = 'davidson'
        
        processed_df = df[['text', 'isHate', 'original_language', 'source']].copy()
        processed_df = processed_df.dropna()
        
        logger.info(f"Processed Davidson: {len(processed_df)} samples, {processed_df['isHate'].sum()} hate speech")
        return processed_df
    
    def collect_all_datasets(self) -> pd.DataFrame:
        """
        Collect and process all available datasets.
        
        Returns:
            Combined DataFrame with all collected data
        """
        logger.info("Starting comprehensive data collection...")
        
        all_datasets = []
        
        for dataset_name, config in self.dataset_sources.items():
            logger.info(f"Processing {dataset_name}...")
            
            # Download dataset
            file_path = self.download_dataset(dataset_name, config['url'], config['format'])
            if not file_path:
                logger.warning(f"Skipping {dataset_name} due to download failure")
                continue
            
            # Process dataset based on type
            try:
                if dataset_name == 'hatexplain':
                    df = self.process_hatexplain(file_path)
                elif dataset_name == 'olid':
                    df = self.process_olid(file_path)
                elif dataset_name == 'davidson':
                    df = self.process_davidson(file_path)
                else:
                    logger.warning(f"Unknown dataset type: {dataset_name}")
                    continue
                
                all_datasets.append(df)
                logger.info(f"Successfully processed {dataset_name}: {len(df)} samples")
                
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {e}")
                continue
        
        if not all_datasets:
            logger.error("No datasets were successfully processed")
            return pd.DataFrame()
        
        # Combine all datasets
        combined_df = pd.concat(all_datasets, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['text'])
        
        # Shuffle the data
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Combined dataset: {len(combined_df)} samples")
        logger.info(f"Hate speech ratio: {combined_df['isHate'].mean():.2%}")
        logger.info(f"Sources: {combined_df['source'].value_counts().to_dict()}")
        
        return combined_df
    
    def save_collected_data(self, df: pd.DataFrame, filename: str = "collected_hate_speech_data.csv"):
        """
        Save collected data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        file_path = self.data_dir / filename
        df.to_csv(file_path, index=False)
        logger.info(f"Saved collected data to {file_path}")
        
        # Also save statistics
        stats_path = self.data_dir / "collection_stats.json"
        stats = {
            'total_samples': len(df),
            'hate_speech_ratio': float(df['isHate'].mean()),
            'sources': df['source'].value_counts().to_dict(),
            'languages': df['original_language'].value_counts().to_dict()
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved collection statistics to {stats_path}")


def main():
    """Main function to run data collection"""
    logger.info("=" * 80)
    logger.info("HATE SPEECH DATA COLLECTION PIPELINE")
    logger.info("=" * 80)
    
    # Initialize collector
    collector = HateSpeechDataCollector()
    
    # Collect all datasets
    collected_data = collector.collect_all_datasets()
    
    if len(collected_data) > 0:
        # Save collected data
        collector.save_collected_data(collected_data)
        
        logger.info("=" * 80)
        logger.info("DATA COLLECTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total samples collected: {len(collected_data)}")
        logger.info(f"Hate speech ratio: {collected_data['isHate'].mean():.2%}")
        
        # Show sample data
        logger.info("\nSample collected data:")
        print(collected_data.head())
        
    else:
        logger.error("No data was collected successfully")


if __name__ == "__main__":
    main()
