import os
import sys
import json
import time
import logging
from typing import List, Dict, Union, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, DetectorFactory

# Ensure project root is on path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0


class OptimizedHateDetector:
    """
    Optimized multilingual hate speech detector using tuned thresholds
    for maximum performance across all 5 languages.
    """
    
    def __init__(self, model_path: str = "models/mbert_improved_multilingual", 
                 thresholds_path: str = "models/optimized_thresholds.json"):
        """
        Initialize the optimized hate speech detector.
        
        Args:
            model_path: Path to the trained mBERT model
            thresholds_path: Path to the optimized thresholds file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.thresholds_path = thresholds_path
        
        # Language mapping
        self.language_mapping = {
            'en': 'eng', 'english': 'eng',
            'ta': 'tam', 'tamil': 'tam',
            'hi': 'hin', 'hindi': 'hin',
            'es': 'spa', 'spanish': 'spa',
            'zh': 'cmn', 'chinese': 'cmn', 'zh-cn': 'cmn'
        }
        
        # Default threshold (fallback)
        self.default_threshold = 0.5
        
        # Load model and thresholds
        self._load_model()
        self._load_thresholds()
        
        logger.info("OptimizedHateDetector initialized successfully")
        logger.info(f"Device: {self.device}")
        logger.info(f"Loaded thresholds: {self.thresholds}")
    
    def _load_model(self):
        """Load the trained mBERT model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _load_thresholds(self):
        """Load optimized thresholds for each language"""
        logger.info(f"Loading thresholds from {self.thresholds_path}")
        
        if not os.path.exists(self.thresholds_path):
            logger.warning(f"Thresholds file not found at {self.thresholds_path}")
            logger.warning("Using default threshold (0.5) for all languages")
            self.thresholds = {lang: self.default_threshold for lang in ['eng', 'tam', 'hin', 'spa', 'cmn']}
            return
        
        with open(self.thresholds_path, 'r') as f:
            self.thresholds = json.load(f)
        
        logger.info(f"Loaded thresholds: {self.thresholds}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language code (eng, tam, hin, spa, cmn)
        """
        try:
            # Clean text for language detection
            clean_text = text.strip()
            if len(clean_text) < 3:
                return 'eng'  # Default to English for very short text
            
            # Detect language
            detected_lang = detect(clean_text)
            
            # Map to our language codes
            mapped_lang = self.language_mapping.get(detected_lang, 'eng')
            
            return mapped_lang
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}. Defaulting to English.")
            return 'eng'
    
    def _preprocess_text(self, text: str, language: str) -> str:
        """
        Preprocess text for model input.
        
        Args:
            text: Input text
            language: Detected language code
            
        Returns:
            Preprocessed text
        """
        # Add language token for better multilingual understanding
        if language != 'eng':
            return f"[{language}] {text}"
        return text
    
    def _get_hate_probability(self, text: str, language: str, max_length: int = 128) -> float:
        """
        Get hate speech probability for a given text.
        
        Args:
            text: Input text
            language: Language code
            max_length: Maximum token length
            
        Returns:
            Hate speech probability (0.0 to 1.0)
        """
        # Preprocess text
        processed_text = self._preprocess_text(text, language)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            hate_probability = probabilities[0][1].item()  # Probability of hate class
        
        return hate_probability
    
    def predict(self, text: str, language: Optional[str] = None, 
                max_length: int = 128) -> Dict:
        """
        Predict hate speech for a single text.
        
        Args:
            text: Input text to analyze
            language: Optional language code (auto-detected if not provided)
            max_length: Maximum token length
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Detect language if not provided
        if language is None:
            language = self.detect_language(text)
        
        # Get hate probability
        hate_probability = self._get_hate_probability(text, language, max_length)
        
        # Get language-specific threshold
        threshold = self.thresholds.get(language, self.default_threshold)
        
        # Make prediction
        is_hate_speech = hate_probability >= threshold
        confidence = hate_probability if is_hate_speech else (1.0 - hate_probability)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        return {
            'text': text,
            'language': language,
            'is_hate_speech': is_hate_speech,
            'hate_probability': round(hate_probability, 4),
            'confidence': round(confidence, 4),
            'threshold_used': threshold,
            'inference_time_ms': round(inference_time * 1000, 2)
        }
    
    def predict_batch(self, texts: List[str], languages: Optional[List[str]] = None,
                     max_length: int = 128) -> List[Dict]:
        """
        Predict hate speech for multiple texts.
        
        Args:
            texts: List of input texts
            languages: Optional list of language codes
            max_length: Maximum token length
            
        Returns:
            List of prediction dictionaries
        """
        if languages is None:
            languages = [self.detect_language(text) for text in texts]
        
        if len(languages) != len(texts):
            raise ValueError("Number of languages must match number of texts")
        
        results = []
        for text, language in zip(texts, languages):
            result = self.predict(text, language, max_length)
            results.append(result)
        
        return results
    
    def get_language_stats(self) -> Dict:
        """
        Get statistics about the loaded thresholds and model.
        
        Returns:
            Dictionary with language statistics
        """
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'thresholds': self.thresholds,
            'supported_languages': list(self.thresholds.keys()),
            'language_mapping': self.language_mapping
        }
    
    def test_performance(self, test_texts: List[str], expected_labels: List[bool]) -> Dict:
        """
        Test model performance on a set of texts.
        
        Args:
            test_texts: List of test texts
            expected_labels: List of expected hate speech labels
            
        Returns:
            Performance metrics dictionary
        """
        if len(test_texts) != len(expected_labels):
            raise ValueError("Number of texts must match number of expected labels")
        
        predictions = self.predict_batch(test_texts)
        predicted_labels = [pred['is_hate_speech'] for pred in predictions]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(expected_labels, predicted_labels)
        precision = precision_score(expected_labels, predicted_labels, zero_division=0)
        recall = recall_score(expected_labels, predicted_labels, zero_division=0)
        f1 = f1_score(expected_labels, predicted_labels, zero_division=0)
        
        # Calculate average inference time
        avg_inference_time = np.mean([pred['inference_time_ms'] for pred in predictions])
        
        return {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'avg_inference_time_ms': round(avg_inference_time, 2),
            'total_samples': len(test_texts)
        }


def main():
    """Demo script showing how to use the OptimizedHateDetector"""
    logger.info("=" * 80)
    logger.info("OPTIMIZED MULTILINGUAL HATE SPEECH DETECTOR DEMO")
    logger.info("=" * 80)
    
    # Initialize detector
    try:
        detector = OptimizedHateDetector()
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return
    
    # Show language stats
    stats = detector.get_language_stats()
    logger.info(f"Model loaded from: {stats['model_path']}")
    logger.info(f"Device: {stats['device']}")
    logger.info(f"Supported languages: {stats['supported_languages']}")
    logger.info(f"Thresholds: {stats['thresholds']}")
    
    # Test cases
    test_cases = [
        # English
        ("You are a stupid woman and should die", True),
        ("This is a great day!", False),
        ("All Muslims are terrorists", True),
        ("I love my family", False),
        
        # Tamil
        ("நீ ஒரு வேசி பெண்", True),  # You are a whore woman
        ("இன்று நல்ல நாள்", False),  # Today is a good day
        
        # Hindi
        ("तुम एक बेकार औरत हो", True),  # You are a useless woman
        ("आज का दिन बहुत अच्छा है", False),  # Today is a very good day
        
        # Spanish
        ("Eres una mujer estúpida y deberías morir", True),
        ("¡Qué día tan maravilloso!", False),
        
        # Chinese
        ("你是个笨女人应该死", True),  # You are a stupid woman and should die
        ("今天是个好日子", False),  # Today is a good day
    ]
    
    logger.info("\n" + "="*50)
    logger.info("TESTING OPTIMIZED DETECTOR")
    logger.info("="*50)
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for text, expected in test_cases:
        result = detector.predict(text)
        
        status = "✅ CORRECT" if result['is_hate_speech'] == expected else "❌ WRONG"
        if result['is_hate_speech'] == expected:
            correct_predictions += 1
        
        logger.info(f"{status} | {text[:30]}... | "
                   f"Predicted: {result['is_hate_speech']} | "
                   f"Expected: {expected} | "
                   f"Lang: {result['language']} | "
                   f"Conf: {result['confidence']} | "
                   f"Time: {result['inference_time_ms']}ms")
    
    accuracy = correct_predictions / total_predictions
    logger.info(f"\nOverall Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    
    # Performance test
    logger.info("\n" + "="*50)
    logger.info("PERFORMANCE TEST")
    logger.info("="*50)
    
    test_texts = [case[0] for case in test_cases]
    expected_labels = [case[1] for case in test_cases]
    
    perf_results = detector.test_performance(test_texts, expected_labels)
    logger.info(f"Accuracy: {perf_results['accuracy']}")
    logger.info(f"Precision: {perf_results['precision']}")
    logger.info(f"Recall: {perf_results['recall']}")
    logger.info(f"F1 Score: {perf_results['f1_score']}")
    logger.info(f"Avg Inference Time: {perf_results['avg_inference_time_ms']}ms")
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZED DETECTOR DEMO COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
