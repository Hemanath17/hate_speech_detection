"""
Interactive Hate Speech Detection Test Script
Takes text input from terminal and predicts if it's hate speech
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class HateSpeechDetector:
    """Hate speech detection using mBERT model"""
    
    def __init__(self, model_path: str = 'models/mbert_improved_chinese_multilingual'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        self.load_thresholds()
        
        # Language detection patterns
        self.language_patterns = {
            'tam': re.compile(r'[\u0B80-\u0BFF]+'),  # Tamil script
            'hin': re.compile(r'[\u0900-\u097F]+'),  # Devanagari script
            'cmn': re.compile(r'[\u4e00-\u9fff]+'),  # Chinese characters
            'spa': re.compile(r'[áéíóúüñÁÉÍÓÚÜÑ]'),  # Spanish accented characters
            'eng': re.compile(r'^[a-zA-Z\s\.,!?;:\'"-]+$')  # English (fallback)
        }
        
        # Strong hate keywords for override
        self.strong_hate_keywords = [
            'kill', 'death', 'murder', 'hate you', 'i hate'
        ]
    
    def load_model(self):
        """Load the mBERT model and tokenizer"""
        print(f"Loading model from {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def load_thresholds(self):
        """Load language-specific thresholds"""
        thresholds_path = 'models/optimized_thresholds.json'
        if os.path.exists(thresholds_path):
            try:
                with open(thresholds_path, 'r') as f:
                    self.thresholds = json.load(f)
                # Lower all thresholds for better detection
                self.thresholds['eng'] = min(self.thresholds.get('eng', 0.65), 0.30)
                self.thresholds['tam'] = min(self.thresholds.get('tam', 0.1), 0.1)
                self.thresholds['hin'] = min(self.thresholds.get('hin', 0.50), 0.35)
                self.thresholds['spa'] = min(self.thresholds.get('spa', 0.60), 0.40)
                self.thresholds['cmn'] = min(self.thresholds.get('cmn', 0.45), 0.35)
                print(f"✓ Loaded thresholds: {self.thresholds}")
            except:
                self.thresholds = {
                    'eng': 0.30,
                    'tam': 0.1,
                    'hin': 0.35,
                    'spa': 0.40,
                    'cmn': 0.35
                }
        else:
            self.thresholds = {
                'eng': 0.30,
                'tam': 0.1,
                'hin': 0.35,
                'spa': 0.40,
                'cmn': 0.35
            }
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        text_clean = text.strip()
        
        # Check for Chinese characters
        if self.language_patterns['cmn'].search(text_clean):
            chinese_chars = len(self.language_patterns['cmn'].findall(text_clean))
            total_chars = len(re.sub(r'\s', '', text_clean))
            if total_chars > 0 and chinese_chars / total_chars > 0.3:
                return 'cmn'
        
        # Check for Tamil script
        if self.language_patterns['tam'].search(text_clean):
            return 'tam'
        
        # Check for Hindi/Devanagari script
        if self.language_patterns['hin'].search(text_clean):
            return 'hin'
        
        # Check for Spanish accented characters
        if self.language_patterns['spa'].search(text_clean):
            return 'spa'
        
        # Default to English
        return 'eng'
    
    def has_strong_hate_keywords(self, text: str) -> bool:
        """Check for strong hate keywords"""
        text_lower = text.lower()
        return any(word in text_lower for word in self.strong_hate_keywords)
    
    def predict(self, text: str, language: str = None) -> dict:
        """Predict if text is hate speech"""
        if not text or not text.strip():
            return {
                'is_hate': False,
                'confidence': 0.0,
                'hate_probability': 0.0,
                'neutral_probability': 1.0,
                'language': 'unknown',
                'error': 'Empty text'
            }
        
        # Auto-detect language if not provided
        if language is None:
            language = self.detect_language(text)
        
        # Prepare text with language token
        if language != 'eng':
            processed_text = f"[{language}] {text}"
        else:
            processed_text = text
        
        # Tokenize and predict
        try:
            encodings = self.tokenizer(
                processed_text,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                
                hate_probability = probabilities[0][1].item()
                neutral_probability = probabilities[0][0].item()
            
            # Get threshold
            threshold = self.thresholds.get(language, 0.5)
            
            # Check for strong keywords
            has_strong_keywords = self.has_strong_hate_keywords(text)
            
            # Decision logic
            if has_strong_keywords and hate_probability > 0.20:
                is_hate = True
                confidence = max(hate_probability, 0.6)
            else:
                # Let model decide with lower threshold
                is_hate = hate_probability >= threshold
                confidence = hate_probability if is_hate else neutral_probability
            
            return {
                'is_hate': is_hate,
                'confidence': confidence,
                'hate_probability': hate_probability,
                'neutral_probability': neutral_probability,
                'language': language,
                'threshold_used': threshold,
                'keyword_override': has_strong_keywords
            }
            
        except Exception as e:
            return {
                'is_hate': False,
                'confidence': 0.0,
                'hate_probability': 0.0,
                'neutral_probability': 1.0,
                'language': language,
                'error': str(e)
            }


def print_result(text: str, result: dict):
    """Print formatted prediction result"""
    print("\n" + "="*70)
    print(f"Input Text: {text}")
    print("="*70)
    
    if result.get('error'):
        print(f"✗ Error: {result['error']}")
        return
    
    # Language mapping for display
    lang_names = {
        'eng': 'English',
        'tam': 'Tamil',
        'hin': 'Hindi',
        'spa': 'Spanish',
        'cmn': 'Chinese'
    }
    
    language = lang_names.get(result['language'], result['language'].upper())
    
    # Prediction
    if result['is_hate']:
        print("🔴 RESULT: HATE SPEECH DETECTED")
    else:
        print("🟢 RESULT: NOT HATE SPEECH")
    
    print(f"\nLanguage Detected: {language}")
    print(f"Threshold Used: {result['threshold_used']*100:.1f}%")
    print(f"\nProbabilities:")
    print(f"  Hate Speech:     {result['hate_probability']*100:.2f}%")
    print(f"  Neutral/Safe:    {result['neutral_probability']*100:.2f}%")
    print(f"  Confidence:      {result['confidence']*100:.2f}%")
    
    if result.get('keyword_override'):
        print(f"\n⚠️  Keyword override applied (strong hate indicators detected)")
    
    print("="*70 + "\n")


def main():
    """Main interactive loop"""
    print("\n" + "="*70)
    print("Hate Speech Detection - Interactive Test")
    print("="*70)
    print("Supports: English, Tamil, Hindi, Chinese, Spanish")
    print("Type 'quit' or 'exit' to stop")
    print("="*70 + "\n")
    
    # Initialize detector
    try:
        detector = HateSpeechDetector()
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        print("\nMake sure:")
        print("1. Model files exist in models/mbert_improved_chinese_multilingual/")
        print("2. You're running from the project root directory")
        return
    
    # Interactive loop
    while True:
        try:
            # Get input
            text = input("Enter text to test: ").strip()
            
            # Check for exit commands
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not text:
                print("Please enter some text.\n")
                continue
            
            # Predict
            result = detector.predict(text)
            
            # Print result
            print_result(text, result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


if __name__ == '__main__':
    main()

