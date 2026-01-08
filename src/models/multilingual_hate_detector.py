import torch
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from transformers import DistilBertTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import os
import sys
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

from configs.config import MODEL_CONFIG, DATA_PATHS, CATEGORIES, PERFORMANCE_TARGETS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultilingualHateSpeechDetector:
    """
    Complete multilingual hate speech detection system:
    1. Language detection
    2. Translation to English (mBART)
    3. Binary hate speech detection (tuned threshold)
    4. Multi-label categorization
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.categories = config['categories']
        self.languages = {
            'en': 'English',
            'ta': 'Tamil', 
            'hi': 'Hindi',
            'es': 'Spanish',
            'zh': 'Chinese'
        }
        
        # Models
        self.binary_model = None
        self.multilabel_model = None
        self.binary_tokenizer = None
        self.multilabel_tokenizer = None
        self.translation_model = None
        self.translation_tokenizer = None
        
        # Thresholds
        self.binary_threshold = 0.5  # Will be tuned
        self.category_threshold = 0.5
        
    def load_models(self, model_path: str):
        """Load all pre-trained models"""
        logger.info("Loading all models...")
        
        # Load binary model
        self._load_binary_model(f"{model_path}/binary_model.pth", f"{model_path}/binary_tokenizer")
        
        # Load multi-label model
        self._load_multilabel_model(f"{model_path}/multilabel_model.pth", f"{model_path}/multilabel_tokenizer")
        
        # Load translation model (mBART)
        self._load_translation_model()
        
        logger.info("All models loaded successfully")
    
    def _load_binary_model(self, model_path: str, tokenizer_path: str):
        """Load binary classification model"""
        class BinaryDistilBERT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                from transformers import DistilBertModel
                self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
                self.classifier = torch.nn.Linear(768, 2)
                self.dropout = torch.nn.Dropout(0.1)
            
            def forward(self, input_ids, attention_mask):
                outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0]
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                return logits
        
        self.binary_model = BinaryDistilBERT()
        checkpoint = torch.load(model_path, map_location='cpu')
        self.binary_model.load_state_dict(checkpoint)
        self.binary_model.eval()
        self.binary_tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    
    def _load_multilabel_model(self, model_path: str, tokenizer_path: str):
        """Load multi-label classification model"""
        from src.models.distilbert_model import MultiLabelDistilBERT
        
        multilabel_config = MODEL_CONFIG.copy()
        multilabel_config['categories'] = self.categories
        
        self.multilabel_model = MultiLabelDistilBERT(multilabel_config)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.multilabel_model.load_state_dict(checkpoint)
        self.multilabel_model.eval()
        self.multilabel_tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    
    def _load_translation_model(self):
        """Load mBART translation model"""
        logger.info("Loading mBART translation model...")
        self.translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.translation_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        logger.info("mBART model loaded successfully")
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        # Simple heuristic-based language detection
        text_lower = text.lower()
        
        # Tamil detection
        if any(char in text for char in 'தமிழ்'):
            return 'ta'
        
        # Hindi detection  
        if any(char in text for char in 'हिन्दी'):
            return 'hi'
        
        # Chinese detection
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh'
        
        # Spanish detection (common words)
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'más', 'pero', 'sus', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'desde', 'está', 'mi', 'porque', 'sólo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde', 'bien', 'tiempo', 'mismo', 'ese', 'ahora', 'cada', 'e', 'vida', 'otro', 'después', 'te', 'otros', 'aunque', 'esa', 'esos', 'estas', 'estos', 'cual', 'poco', 'tan', 'tanto', 'todo', 'toda', 'todos', 'todas', 'muy', 'más', 'menos', 'mucho', 'muchos', 'muchas', 'poco', 'pocos', 'pocas', 'algo', 'nada', 'todo', 'nada', 'alguien', 'nadie', 'algo', 'nada', 'alguien', 'nadie', 'algo', 'nada', 'alguien', 'nadie']
        if any(word in text_lower for word in spanish_words):
            return 'es'
        
        # Default to English
        return 'en'
    
    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English using mBART"""
        if source_lang == 'en':
            return text
        
        try:
            # Set source language
            self.translation_tokenizer.src_lang = source_lang
            
            # Tokenize
            inputs = self.translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Translate
            with torch.no_grad():
                generated_tokens = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=self.translation_tokenizer.lang_code_to_id["en_XX"]
                )
            
            # Decode
            translated_text = self.translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translated_text
            
        except Exception as e:
            logger.warning(f"Translation failed: {e}. Using original text.")
            return text
    
    def tune_binary_threshold(self, validation_data: List[Tuple[str, int]]) -> float:
        """Tune binary classification threshold for optimal performance"""
        logger.info("Tuning binary classification threshold...")
        
        best_threshold = 0.5
        best_f1 = 0.0
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        
        for threshold in thresholds:
            predictions = []
            true_labels = []
            
            for text, true_label in validation_data:
                # Get prediction
                pred = self._predict_binary_raw(text)
                predictions.append(1 if pred > threshold else 0)
                true_labels.append(true_label)
            
            # Calculate F1 score
            f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        self.binary_threshold = best_threshold
        return best_threshold
    
    def _predict_binary_raw(self, text: str) -> float:
        """Get raw binary prediction probability"""
        encoding = self.binary_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            logits = self.binary_model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            hate_probability = probabilities[0][1].item()
        
        return hate_probability
    
    def predict(self, text: str, source_lang: Optional[str] = None) -> Dict:
        """Complete multilingual hate speech detection pipeline"""
        # Detect language if not provided
        if source_lang is None:
            source_lang = self.detect_language(text)
        
        # Translate to English if needed
        if source_lang != 'en':
            translated_text = self.translate_to_english(text, source_lang)
        else:
            translated_text = text
        
        # Binary classification
        hate_probability = self._predict_binary_raw(translated_text)
        is_hate_speech = hate_probability > self.binary_threshold
        
        result = {
            'original_text': text,
            'translated_text': translated_text,
            'detected_language': source_lang,
            'language_name': self.languages.get(source_lang, 'Unknown'),
            'is_hate_speech': is_hate_speech,
            'hate_confidence': hate_probability,
            'categories': {}
        }
        
        # Multi-label classification if hate speech detected
        if is_hate_speech:
            categories = self._predict_multilabel(translated_text)
            result['categories'] = categories
        else:
            # If not hate speech, all categories are 0
            for category in self.categories:
                result['categories'][category] = {
                    'predicted': False,
                    'confidence': 0.0
                }
        
        return result
    
    def _predict_multilabel(self, text: str) -> Dict:
        """Multi-label classification prediction"""
        encoding = self.multilabel_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            logits = self.multilabel_model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits).squeeze().numpy()
        
        result = {}
        for i, category in enumerate(self.categories):
            result[category] = {
                'predicted': probabilities[i] > self.category_threshold,
                'confidence': float(probabilities[i])
            }
        
        return result

def main():
    logger.info("==================================================")
    logger.info("MULTILINGUAL HATE SPEECH DETECTION SYSTEM")
    logger.info("==================================================")
    
    # Initialize detector
    detector = MultilingualHateSpeechDetector({'categories': CATEGORIES})
    
    # Load models
    model_path = "models/two_stage_hate_detector_20250918_143110"
    if os.path.exists(model_path):
        detector.load_models(model_path)
    else:
        logger.error(f"Model path not found: {model_path}")
        return
    
    # Create validation data for threshold tuning
    validation_samples = [
        ("You are a stupid woman and should die", 1),
        ("All Muslims are terrorists", 1), 
        ("This is a great day!", 0),
        ("People with disabilities are useless", 1),
        ("I hate gay people", 1),
        ("Have a wonderful day", 0),
        ("You're an idiot", 1),
        ("Thanks for your help", 0)
    ]
    
    # Tune binary threshold
    detector.tune_binary_threshold(validation_samples)
    
    # Test multilingual detection
    test_samples = [
        # English
        ("You are a stupid woman and should die", "en"),
        ("All Muslims are terrorists", "en"),
        ("This is a great day!", "en"),
        
        # Spanish
        ("Eres una mujer estúpida y deberías morir", "es"),
        ("Todos los musulmanes son terroristas", "es"),
        ("¡Qué día tan maravilloso!", "es"),
        
        # Hindi (transliterated)
        ("tum ek bekaar aurat ho aur mar jaana chahiye", "hi"),
        ("saare musalmaan aatankavaadi hain", "hi"),
        ("aaj ka din bahut achha hai", "hi"),
        
        # Tamil (transliterated)
        ("neenga oru thevdiya penn irukkeenga", "ta"),
        ("ellam muslimgal terrorgal", "ta"),
        ("indha naal romba nalla naal", "ta"),
        
        # Chinese (transliterated)
        ("ni shi ge ben nü ren ying gai si", "zh"),
        ("suo you de mu si lin dou shi kong bu fen zi", "zh"),
        ("jin tian shi ge hao ri zi", "zh")
    ]
    
    logger.info("Testing multilingual hate speech detection...")
    logger.info("=" * 60)
    
    for text, expected_lang in test_samples:
        result = detector.predict(text, expected_lang)
        
        logger.info(f"\nOriginal: '{text}'")
        logger.info(f"Language: {result['language_name']} ({result['detected_language']})")
        logger.info(f"Translated: '{result['translated_text']}'")
        logger.info(f"Hate Speech: {result['is_hate_speech']} (confidence: {result['hate_confidence']:.3f})")
        
        if result['is_hate_speech']:
            logger.info("Categories:")
            for category, pred in result['categories'].items():
                if pred['predicted']:
                    logger.info(f"  - {category}: {pred['confidence']:.3f}")
    
    logger.info("=" * 60)
    logger.info("MULTILINGUAL SYSTEM TESTING COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
