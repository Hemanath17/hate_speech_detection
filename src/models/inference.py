"""
Inference module for hate speech detection
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import DistilBertTokenizer
from typing import Dict, List, Optional
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class HateSpeechInference:
    """Inference class for hate speech detection"""
    
    def __init__(self, model_path: str, tokenizer_path: str, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Hate speech detection model loaded successfully")
    
    def _load_model(self, model_path: str):
        """Load the trained model"""
        from src.models.distilbert_model import MultiLabelDistilBERT
        
        # Create model instance
        model = MultiLabelDistilBERT({'categories': self.config['categories'], **self.config})
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        
        return model
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for inference"""
        # Basic cleaning
        text = text.strip()
        
        # Truncate if too long
        if len(text) > self.config['max_length']:
            text = text[:self.config['max_length']]
        
        return text
    
    def predict_single(self, text: str) -> Dict[str, any]:
        """Predict hate speech for a single text"""
        start_time = time.time()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['max_length']
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Convert to binary predictions
        predictions = (probabilities > 0.5).astype(int)
        
        # Create results
        results = {
            'text': text,
            'processed_text': processed_text,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'categories': self.config['categories'],
            'processing_time': time.time() - start_time
        }
        
        # Add category-wise results
        category_results = {}
        for i, category in enumerate(self.config['categories']):
            category_results[category] = {
                'predicted': bool(predictions[i]),
                'probability': float(probabilities[i]),
                'confidence': 'high' if probabilities[i] > 0.8 else 'medium' if probabilities[i] > 0.6 else 'low'
            }
        
        results['category_results'] = category_results
        
        # Overall prediction
        has_hate = any(predictions)
        max_prob = float(np.max(probabilities))
        
        results['overall'] = {
            'has_hate_speech': has_hate,
            'max_probability': max_prob,
            'confidence': 'high' if max_prob > 0.8 else 'medium' if max_prob > 0.6 else 'low'
        }
        
        return results
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """Predict hate speech for multiple texts"""
        results = []
        
        for text in texts:
            try:
                result = self.predict_single(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text '{text[:50]}...': {e}")
                results.append({
                    'text': text,
                    'error': str(e),
                    'predictions': [0] * len(self.config['categories']),
                    'probabilities': [0.0] * len(self.config['categories'])
                })
        
        return results
    
    def get_explanation(self, result: Dict[str, any]) -> str:
        """Generate explanation for the prediction"""
        if not result['overall']['has_hate_speech']:
            return "No hate speech detected in this text."
        
        # Find categories with hate speech
        hate_categories = []
        for category, cat_result in result['category_results'].items():
            if cat_result['predicted']:
                confidence = cat_result['confidence']
                prob = cat_result['probability']
                hate_categories.append(f"{category} ({confidence} confidence: {prob:.2f})")
        
        if hate_categories:
            return f"Hate speech detected in categories: {', '.join(hate_categories)}"
        else:
            return "Hate speech detected but specific categories unclear."

def create_inference_pipeline(model_path: str, tokenizer_path: str, config: Dict):
    """Create inference pipeline"""
    return HateSpeechInference(model_path, tokenizer_path, config)

def test_inference():
    """Test inference functionality"""
    from configs.config import MODEL_CONFIG, CATEGORIES
    
    # Test with sample model (this would normally load a trained model)
    config = {'categories': CATEGORIES, **MODEL_CONFIG}
    
    # Sample texts for testing
    test_texts = [
        "You should know women's sports are a joke",
        "I love this beautiful day!",
        "Women deserve to be abused, I guess.",
        "This is a normal conversation about technology.",
        "I hate all muslims and jews equally"
    ]
    
    print("Testing Hate Speech Detection:")
    print("=" * 50)
    
    # Note: This would require a trained model to work properly
    # For now, just show the structure
    for i, text in enumerate(test_texts, 1):
        print(f"\nSample {i}: {text}")
        print("(Would run inference here with trained model)")

if __name__ == "__main__":
    test_inference()
