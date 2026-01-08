"""
Data preprocessing pipeline for hate speech detection
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class HateSpeechPreprocessor:
    """Preprocessing pipeline for hate speech detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.mlb = MultiLabelBinarizer()
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            config.get('distilbert_model', 'distilbert-base-uncased')
        )
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Remove extra punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenize and lemmatize text"""
        if not text:
            return ""
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_text(self, text: str) -> str:
        """Complete text preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.tokenize_and_lemmatize(text)
        return text
    
    def load_ethos_dataset(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess Ethos dataset"""
        logger.info(f"Loading dataset from {file_path}")
        
        # Load the dataset
        df = pd.read_csv(file_path, sep=';')
        
        # Rename columns for consistency
        df.columns = ['comment', 'violence', 'directed_vs_generalized', 'gender', 
                     'race', 'national_origin', 'disability', 'religion', 'sexual_orientation']
        
        # Clean text
        df['comment'] = df['comment'].apply(self.preprocess_text)
        
        # Remove empty comments
        df = df[df['comment'].str.len() > 0]
        
        # Convert labels to binary (threshold 0.5)
        label_columns = ['violence', 'directed_vs_generalized', 'gender', 'race', 
                        'national_origin', 'disability', 'religion', 'sexual_orientation']
        
        for col in label_columns:
            df[col] = (df[col] > 0.5).astype(int)
        
        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Label distribution:\n{df[label_columns].sum()}")
        
        return df
    
    def create_multilabel_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create multi-label data for training"""
        label_columns = ['violence', 'directed_vs_generalized', 'gender', 'race', 
                        'national_origin', 'disability', 'religion', 'sexual_orientation']
        
        # Get texts and labels
        texts = df['comment'].values
        
        # Ensure all label columns exist and convert to proper types
        for col in label_columns:
            if col in df.columns:
                df[col] = df[col].astype(float).fillna(0).astype(int)
            else:
                df[col] = 0
        
        labels = df[label_columns].values
        
        # Convert to binary multi-label format (each sample has 8 binary labels)
        multilabel_data = []
        for i in range(len(labels)):
            sample_labels = []
            for j, col in enumerate(label_columns):
                # Ensure each label is an integer
                label_val = labels[i, j]
                if isinstance(label_val, str):
                    label_val = 1 if label_val.lower() in ['true', '1', 'yes'] else 0
                else:
                    label_val = int(float(label_val))
                sample_labels.append(label_val)
            multilabel_data.append(sample_labels)
        
        return texts, np.array(multilabel_data, dtype=np.int32)
    
    def split_data(self, texts: np.ndarray, labels: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """Split data into train, validation, and test sets"""
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=None
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=None
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for handling imbalance"""
        # Count positive samples for each class
        class_counts = {}
        for i in range(len(self.config['categories'])):
            count = sum(1 for labels in y_train if i in labels)
            class_counts[i] = count
        
        # Calculate weights (inverse frequency)
        total_samples = len(y_train)
        class_weights = {}
        for i, count in class_counts.items():
            if count > 0:
                class_weights[i] = total_samples / (len(self.config['categories']) * count)
            else:
                class_weights[i] = 1.0
        
        logger.info(f"Class weights: {class_weights}")
        return class_weights

def main():
    """Test the preprocessing pipeline"""
    from configs.config import CATEGORIES
    
    config = {'categories': CATEGORIES}
    preprocessor = HateSpeechPreprocessor(config)
    
    # Test with sample data
    sample_text = "You should know women's sports are a joke @user #hashtag"
    cleaned = preprocessor.preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")

if __name__ == "__main__":
    main()
