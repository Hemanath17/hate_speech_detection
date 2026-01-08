import pandas as pd
import numpy as np
import bz2
import logging
from typing import Dict, List, Tuple
import re
import random
from pathlib import Path

logger = logging.getLogger(__name__)

class MultilingualDatasetCreator:
    """
    Create multilingual hate speech datasets from your 5-language datasets
    by generating synthetic hate speech labels and examples
    """
    
    def __init__(self, data_path: str = "Data/"):
        self.data_path = Path(data_path)
        self.languages = {
            'tam': 'Tamil',
            'hin': 'Hindi', 
            'spa': 'Spanish',
            'cmn': 'Chinese',
            'eng': 'English'
        }
        
        # Hate speech patterns for each language
        self.hate_patterns = {
            'tam': {
                'gender': ['பெண்', 'ஆண்', 'முட்டாள்', 'பைத்தியம்'],
                'religion': ['முஸ்லிம்', 'இந்து', 'கிறிஸ்தவன்'],
                'violence': ['கொல்ல', 'சாக', 'தாக்கு'],
                'disability': ['முட்டாள்', 'பைத்தியம்', 'குருடன்'],
                'race': ['கருப்பு', 'வெள்ளை', 'இந்தியன்']
            },
            'hin': {
                'gender': ['औरत', 'आदमी', 'बेवकूफ', 'पागल'],
                'religion': ['मुसलमान', 'हिंदू', 'ईसाई'],
                'violence': ['मार', 'मर', 'हत्या'],
                'disability': ['बेवकूफ', 'पागल', 'अंधा'],
                'race': ['काला', 'गोरा', 'भारतीय']
            },
            'spa': {
                'gender': ['mujer', 'hombre', 'estúpida', 'loca'],
                'religion': ['musulmán', 'hindú', 'cristiano'],
                'violence': ['matar', 'morir', 'atacar'],
                'disability': ['estúpida', 'loca', 'ciega'],
                'race': ['negro', 'blanco', 'indio']
            },
            'cmn': {
                'gender': ['女人', '男人', '愚蠢', '疯狂'],
                'religion': ['穆斯林', '印度教', '基督教'],
                'violence': ['杀', '死', '攻击'],
                'disability': ['愚蠢', '疯狂', '盲人'],
                'race': ['黑人', '白人', '印度人']
            },
            'eng': {
                'gender': ['woman', 'man', 'stupid', 'crazy'],
                'religion': ['muslim', 'hindu', 'christian'],
                'violence': ['kill', 'die', 'attack'],
                'disability': ['stupid', 'crazy', 'blind'],
                'race': ['black', 'white', 'indian']
            }
        }
    
    def load_language_dataset(self, language: str, sample_size: int = 10000) -> pd.DataFrame:
        """Load and sample dataset for a specific language"""
        file_path = self.data_path / f"{language}_sentences_detailed.tsv.bz2"
        
        if not file_path.exists():
            logger.error(f"Dataset not found: {file_path}")
            return pd.DataFrame()
        
        logger.info(f"Loading {self.languages[language]} dataset...")
        
        with bz2.open(file_path, 'rt', encoding='utf-8') as f:
            # Read in chunks to handle large files
            chunk_size = 1000
            chunks = []
            
            for chunk in pd.read_csv(f, sep='\t', chunksize=chunk_size):
                chunks.append(chunk)
                if len(chunks) * chunk_size >= sample_size:
                    break
            
            df = pd.concat(chunks, ignore_index=True)
            df = df.head(sample_size)
        
        # Rename columns for consistency
        df.columns = ['id', 'language', 'text', 'user', 'created_at', 'updated_at']
        
        logger.info(f"Loaded {len(df)} samples from {self.languages[language]} dataset")
        return df
    
    def generate_hate_speech_labels(self, text: str, language: str) -> Dict[str, int]:
        """Generate hate speech labels based on pattern matching"""
        text_lower = text.lower()
        labels = {
            'violence': 0,
            'directed_vs_generalized': 0,
            'gender': 0,
            'race': 0,
            'national_origin': 0,
            'disability': 0,
            'religion': 0,
            'sexual_orientation': 0
        }
        
        if language not in self.hate_patterns:
            return labels
        
        patterns = self.hate_patterns[language]
        
        # Check for hate speech patterns
        for category, words in patterns.items():
            for word in words:
                if word.lower() in text_lower:
                    labels[category] = 1
                    break
        
        # Additional pattern matching
        violence_patterns = ['kill', 'die', 'attack', 'hurt', 'destroy']
        if any(pattern in text_lower for pattern in violence_patterns):
            labels['violence'] = 1
        
        # Determine if it's hate speech overall
        is_hate = sum(labels.values()) > 0
        labels['isHate'] = 1 if is_hate else 0
        
        return labels
    
    def create_multilingual_dataset(self, sample_size_per_language: int = 5000) -> pd.DataFrame:
        """Create multilingual hate speech dataset"""
        logger.info("Creating multilingual hate speech dataset...")
        
        all_data = []
        
        for lang_code, lang_name in self.languages.items():
            logger.info(f"Processing {lang_name}...")
            
            # Load language dataset
            df = self.load_language_dataset(lang_code, sample_size_per_language)
            
            if df.empty:
                continue
            
            # Generate labels for each text
            hate_labels = []
            for text in df['text']:
                labels = self.generate_hate_speech_labels(str(text), lang_code)
                hate_labels.append(labels)
            
            # Convert to DataFrame
            labels_df = pd.DataFrame(hate_labels)
            
            # Combine with original data
            combined_df = pd.concat([df[['text', 'language']], labels_df], axis=1)
            combined_df['original_language'] = lang_code
            
            all_data.append(combined_df)
        
        # Combine all languages
        multilingual_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"Created multilingual dataset with {len(multilingual_df)} samples")
        
        # Log statistics
        for lang in self.languages.keys():
            lang_data = multilingual_df[multilingual_df['original_language'] == lang]
            hate_count = lang_data['isHate'].sum()
            logger.info(f"{self.languages[lang]}: {len(lang_data)} samples, {hate_count} hate speech")
        
        return multilingual_df
    
    def create_balanced_dataset(self, target_hate_ratio: float = 0.3) -> pd.DataFrame:
        """Create a balanced dataset with specified hate speech ratio"""
        logger.info(f"Creating balanced dataset with {target_hate_ratio:.1%} hate speech...")
        
        # Load all languages
        all_data = []
        
        for lang_code, lang_name in self.languages.items():
            logger.info(f"Processing {lang_name}...")
            
            df = self.load_language_dataset(lang_code, 2000)  # Smaller sample for balance
            
            if df.empty:
                continue
            
            # Generate labels
            hate_labels = []
            for text in df['text']:
                labels = self.generate_hate_speech_labels(str(text), lang_code)
                hate_labels.append(labels)
            
            labels_df = pd.DataFrame(hate_labels)
            combined_df = pd.concat([df[['text', 'language']], labels_df], axis=1)
            combined_df['original_language'] = lang_code
            
            all_data.append(combined_df)
        
        # Combine and balance
        multilingual_df = pd.concat(all_data, ignore_index=True)
        
        # Separate hate and non-hate samples
        hate_samples = multilingual_df[multilingual_df['isHate'] == 1]
        non_hate_samples = multilingual_df[multilingual_df['isHate'] == 0]
        
        # Calculate how many non-hate samples to keep
        target_hate_count = len(hate_samples)
        target_non_hate_count = int(target_hate_count * (1 - target_hate_ratio) / target_hate_ratio)
        
        # Sample non-hate samples
        if len(non_hate_samples) > target_non_hate_count:
            non_hate_samples = non_hate_samples.sample(n=target_non_hate_count, random_state=42)
        
        # Combine balanced dataset
        balanced_df = pd.concat([hate_samples, non_hate_samples], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Balanced dataset: {len(balanced_df)} samples")
        logger.info(f"Hate speech: {balanced_df['isHate'].sum()} ({balanced_df['isHate'].mean():.1%})")
        
        return balanced_df

def main():
    """Test the multilingual dataset creator"""
    logging.basicConfig(level=logging.INFO)
    
    creator = MultilingualDatasetCreator()
    
    # Create balanced multilingual dataset
    dataset = creator.create_balanced_dataset(target_hate_ratio=0.3)
    
    # Save dataset
    output_path = "Data/multilingual_hate_speech_dataset.csv"
    dataset.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to: {output_path}")
    
    # Show sample
    print("\nSample data:")
    print(dataset[['text', 'original_language', 'isHate', 'violence', 'gender', 'religion']].head(10))
    
    # Show statistics
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Hate speech ratio: {dataset['isHate'].mean():.1%}")
    print(f"Language distribution:")
    print(dataset['original_language'].value_counts())

if __name__ == "__main__":
    main()
