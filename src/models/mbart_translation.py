"""
mBART integration for multilingual hate speech detection
"""

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from langdetect import detect, DetectorFactory
import logging
from typing import Dict, List, Tuple, Optional
import time

logger = logging.getLogger(__name__)

class MultilingualTranslator:
    """Multilingual translation using mBART"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed for consistent language detection
        DetectorFactory.seed = 0
        
        # Language mappings
        self.language_codes = {
            'en': 'en_XX',      # English
            'ta': 'ta_IN',      # Tamil
            'hi': 'hi_IN',      # Hindi
            'es': 'es_XX',      # Spanish
            'zh': 'zh_CN'       # Mandarin
        }
        
        # Initialize mBART model and tokenizer
        self._load_models()
        
    def _load_models(self):
        """Load mBART model and tokenizer"""
        logger.info("Loading mBART model and tokenizer...")
        
        try:
            self.tokenizer = MBart50TokenizerFast.from_pretrained(
                self.config['mbart_model']
            )
            self.model = MBartForConditionalGeneration.from_pretrained(
                self.config['mbart_model']
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("mBART model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load mBART model: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            # Clean text for better detection
            clean_text = text.strip()[:1000]  # Limit length for detection
            
            if len(clean_text) < 3:
                return 'en'  # Default to English for very short text
            
            detected_lang = detect(clean_text)
            
            # Map to supported languages
            if detected_lang in self.language_codes:
                return detected_lang
            else:
                logger.warning(f"Unsupported language detected: {detected_lang}, defaulting to English")
                return 'en'
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to English")
            return 'en'
    
    def translate_to_english(self, text: str, source_lang: str) -> Tuple[str, float]:
        """Translate text to English using mBART"""
        try:
            if source_lang == 'en':
                return text, 1.0  # No translation needed
            
            # Get source language code
            src_lang_code = self.language_codes.get(source_lang, 'en_XX')
            
            # Tokenize
            self.tokenizer.src_lang = src_lang_code
            encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move to device
            input_ids = encoded.input_ids.to(self.device)
            attention_mask = encoded.attention_mask.to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"],
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode translation
            translated_text = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]
            
            # Calculate confidence (simplified)
            confidence = min(1.0, len(translated_text) / max(len(text), 1))
            
            return translated_text, confidence
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text, 0.0  # Return original text with low confidence
    
    def translate_text(self, text: str) -> Dict[str, any]:
        """Complete translation pipeline"""
        start_time = time.time()
        
        # Detect language
        detected_lang = self.detect_language(text)
        
        # Translate if needed
        if detected_lang == 'en':
            translated_text = text
            translation_confidence = 1.0
        else:
            translated_text, translation_confidence = self.translate_to_english(text, detected_lang)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            'original_text': text,
            'translated_text': translated_text,
            'detected_language': detected_lang,
            'translation_confidence': translation_confidence,
            'processing_time': processing_time,
            'needs_translation': detected_lang != 'en'
        }

class MultilingualHateDetector:
    """Multilingual hate speech detector combining translation and detection"""
    
    def __init__(self, translator, hate_detector, config: Dict):
        self.translator = translator
        self.hate_detector = hate_detector
        self.config = config
        
    def detect_hate_speech(self, text: str) -> Dict[str, any]:
        """Detect hate speech in any supported language"""
        start_time = time.time()
        
        # Translate text
        translation_result = self.translator.translate_text(text)
        
        # Detect hate speech on translated text
        hate_detection_result = self.hate_detector.predict(translation_result['translated_text'])
        
        # Combine results
        total_time = time.time() - start_time
        
        return {
            'original_text': text,
            'translated_text': translation_result['translated_text'],
            'detected_language': translation_result['detected_language'],
            'translation_confidence': translation_result['translation_confidence'],
            'hate_detection': hate_detection_result,
            'total_processing_time': total_time,
            'translation_time': translation_result['processing_time'],
            'detection_time': hate_detection_result.get('processing_time', 0)
        }

def test_translation():
    """Test translation functionality"""
    from configs.config import MODEL_CONFIG
    
    # Initialize translator
    translator = MultilingualTranslator(MODEL_CONFIG)
    
    # Test samples in different languages
    test_samples = [
        "You should know women's sports are a joke",  # English
        "Las mujeres son terribles en los deportes",  # Spanish
        "महिलाएं खेल में बहुत बुरी हैं",  # Hindi
        "女性在体育方面很糟糕",  # Mandarin
        "பெண்கள் விளையாட்டில் மிகவும் மோசமானவர்கள்"  # Tamil
    ]
    
    print("Testing mBART Translation:")
    print("=" * 50)
    
    for i, text in enumerate(test_samples, 1):
        result = translator.translate_text(text)
        print(f"\nSample {i}:")
        print(f"Original: {text}")
        print(f"Detected Language: {result['detected_language']}")
        print(f"Translated: {result['translated_text']}")
        print(f"Confidence: {result['translation_confidence']:.3f}")
        print(f"Time: {result['processing_time']:.3f}s")

if __name__ == "__main__":
    test_translation()
