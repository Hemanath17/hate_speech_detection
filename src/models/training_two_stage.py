import torch
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

from configs.config import MODEL_CONFIG, DATA_PATHS, CATEGORIES, PERFORMANCE_TARGETS
from src.data.preprocessing import HateSpeechPreprocessor
from src.data.augmentation import HateSpeechAugmenter
from src.data.ethos_loader import EthosDataLoader
from src.models.distilbert_model import MultiLabelDistilBERT, HateSpeechDataset, HateSpeechTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwoStageHateSpeechDetector:
    """
    Two-stage hate speech detection:
    Stage 1: Binary classification (hate vs non-hate)
    Stage 2: Multi-label classification (8 categories) for hate samples only
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.categories = config['categories']
        self.binary_model = None
        self.multilabel_model = None
        self.binary_tokenizer = None
        self.multilabel_tokenizer = None
        
    def load_binary_model(self, model_path: str, tokenizer_path: str):
        """Load the pre-trained binary classification model"""
        logger.info(f"Loading binary model from {model_path}")
        
        # Create a simple binary model that matches the saved architecture
        class BinaryDistilBERT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                from transformers import DistilBertModel
                self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
                self.classifier = torch.nn.Linear(768, 2)  # 2 classes for binary classification
                self.dropout = torch.nn.Dropout(0.1)
            
            def forward(self, input_ids, attention_mask):
                outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                return logits
        
        # Load binary model
        self.binary_model = BinaryDistilBERT()
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location='cpu')
        self.binary_model.load_state_dict(checkpoint)
        self.binary_model.eval()
        
        # Load tokenizer
        self.binary_tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        
        logger.info("Binary model loaded successfully")
    
    def train_multilabel_model(self, hate_samples_df: pd.DataFrame):
        """Train multi-label model on hate samples only"""
        logger.info("Training multi-label model on hate samples...")
        
        # Initialize preprocessor
        preprocessor = HateSpeechPreprocessor({
            'categories': self.categories,
            **MODEL_CONFIG
        })
        
        # Create multi-label data from hate samples only
        texts, labels = preprocessor.create_multilabel_data(hate_samples_df)
        
        logger.info(f"Multi-label training data: {len(texts)} samples")
        
        # Split data
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = preprocessor.split_data(
            texts, labels, test_size=0.2, val_size=0.1
        )
        
        logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Data augmentation for multi-label
        logger.info("Applying data augmentation...")
        augmenter = HateSpeechAugmenter({'categories': self.categories})
        augmented_texts, augmented_labels = augmenter.augment_dataset(
            train_texts, train_labels, target_multiplier=3.0  # More augmentation for smaller dataset
        )
        
        logger.info(f"Augmented data: {len(augmented_texts)} samples")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            augmented_texts, augmented_labels,
            val_texts, val_labels,
            test_texts, test_labels,
            preprocessor.tokenizer, MODEL_CONFIG
        )
        
        # Initialize multi-label model
        multilabel_config = MODEL_CONFIG.copy()
        multilabel_config['categories'] = self.categories
        
        self.multilabel_model = MultiLabelDistilBERT(multilabel_config)
        self.multilabel_tokenizer = preprocessor.tokenizer
        
        # Train multi-label model
        trainer = HateSpeechTrainer(self.multilabel_model, multilabel_config)
        test_metrics = trainer.train(train_loader, val_loader, test_loader)
        
        logger.info("Multi-label model training completed")
        return test_metrics
    
    def create_data_loaders(self, train_texts, train_labels, val_texts, val_labels, 
                           test_texts, test_labels, tokenizer, config):
        """Create data loaders for multi-label training"""
        train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, config['max_length'])
        val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, config['max_length'])
        test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer, config['max_length'])
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def predict(self, text: str) -> dict:
        """Two-stage prediction"""
        if self.binary_model is None or self.multilabel_model is None:
            raise ValueError("Models not loaded. Please load binary model and train multilabel model first.")
        
        # Stage 1: Binary classification
        binary_prediction = self.predict_binary(text)
        
        result = {
            'text': text,
            'is_hate_speech': binary_prediction['is_hate_speech'],
            'binary_confidence': binary_prediction['confidence'],
            'categories': {}
        }
        
        # Stage 2: Multi-label classification (only if hate speech detected)
        if binary_prediction['is_hate_speech']:
            multilabel_prediction = self.predict_multilabel(text)
            result['categories'] = multilabel_prediction
        else:
            # If not hate speech, all categories are 0
            for category in self.categories:
                result['categories'][category] = {
                    'predicted': False,
                    'confidence': 0.0
                }
        
        return result
    
    def predict_binary(self, text: str) -> dict:
        """Binary classification prediction"""
        # Tokenize text
        encoding = self.binary_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        # Get prediction
        with torch.no_grad():
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            logits = self.binary_model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            hate_probability = probabilities[0][1].item()  # Probability of class 1 (hate)
        
        return {
            'is_hate_speech': hate_probability > 0.5,
            'confidence': hate_probability
        }
    
    def predict_multilabel(self, text: str) -> dict:
        """Multi-label classification prediction"""
        # Tokenize text
        encoding = self.multilabel_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        # Get prediction
        with torch.no_grad():
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            logits = self.multilabel_model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits).squeeze().numpy()
        
        # Format results
        result = {}
        for i, category in enumerate(self.categories):
            result[category] = {
                'predicted': probabilities[i] > 0.5,
                'confidence': float(probabilities[i])
            }
        
        return result

def main():
    logger.info("==================================================")
    logger.info("TWO-STAGE HATE SPEECH DETECTION")
    logger.info("==================================================")
    
    # Load enhanced ETHOS dataset
    logger.info("Loading enhanced ETHOS dataset...")
    ethos_loader = EthosDataLoader({
        'ethos_binary_path': DATA_PATHS['ethos_binary'],
        'ethos_multilabel_path': DATA_PATHS['ethos_multilabel_full'],
        'categories': CATEGORIES
    })
    df, categories = ethos_loader.create_enhanced_multilabel_dataset()
    
    # Filter hate samples for multi-label training
    hate_samples = df[df['isHate'] == 1].copy()
    logger.info(f"Found {len(hate_samples)} hate speech samples for multi-label training")
    
    # Initialize two-stage detector
    detector = TwoStageHateSpeechDetector({'categories': CATEGORIES})
    
    # Load binary model (use the one we trained earlier)
    binary_model_path = "models/hate_speech_binary_20250918_111759_model.pth"
    binary_tokenizer_path = "models/hate_speech_binary_20250918_111759_tokenizer"
    
    if os.path.exists(binary_model_path) and os.path.exists(binary_tokenizer_path):
        detector.load_binary_model(binary_model_path, binary_tokenizer_path)
    else:
        logger.error("Binary model not found. Please train binary model first.")
        return
    
    # Train multi-label model on hate samples only
    multilabel_metrics = detector.train_multilabel_model(hate_samples)
    
    # Test the two-stage approach
    logger.info("Testing two-stage approach...")
    
    # Test samples
    test_samples = [
        "You are a stupid woman and should die",  # Gender + Violence
        "All Muslims are terrorists",  # Religion + Violence
        "This is a great day!",  # Non-hate
        "People with disabilities are useless",  # Disability
        "I hate gay people",  # Sexual orientation
    ]
    
    for sample in test_samples:
        result = detector.predict(sample)
        logger.info(f"\nText: '{sample}'")
        logger.info(f"Is Hate Speech: {result['is_hate_speech']} (confidence: {result['binary_confidence']:.3f})")
        if result['is_hate_speech']:
            logger.info("Categories:")
            for category, pred in result['categories'].items():
                if pred['predicted']:
                    logger.info(f"  - {category}: {pred['confidence']:.3f}")
    
    # Save the two-stage model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"models/two_stage_hate_detector_{timestamp}"
    
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save binary model
    torch.save(detector.binary_model.state_dict(), f"{model_save_path}/binary_model.pth")
    detector.binary_tokenizer.save_pretrained(f"{model_save_path}/binary_tokenizer")
    
    # Save multi-label model
    torch.save(detector.multilabel_model.state_dict(), f"{model_save_path}/multilabel_model.pth")
    detector.multilabel_tokenizer.save_pretrained(f"{model_save_path}/multilabel_tokenizer")
    
    # Save results
    results = {
        'timestamp': timestamp,
        'binary_model_path': binary_model_path,
        'multilabel_metrics': multilabel_metrics,
        'categories': CATEGORIES,
        'model_save_path': model_save_path
    }
    
    with open(f"{model_save_path}/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Two-stage model saved to: {model_save_path}")
    logger.info("==================================================")
    logger.info("TWO-STAGE TRAINING COMPLETE")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
