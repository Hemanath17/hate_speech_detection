"""
Flask API server for hate speech detection
Serves the mBERT model for browser extension
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import json

# Ensure project root is on path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for browser extension with proper headers
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    },
    r"/health": {
        "origins": "*",
        "methods": ["GET", "OPTIONS"]
    }
})

# Global model and tokenizer
model = None
tokenizer = None
device = None
thresholds = None


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
            'stupid', 'worthless', 'idiot', 'moron', 'fool', 'dumb',
            'hate', 'kill', 'die', 'death', 'murder',
            'useless', 'trash', 'garbage', 'scum', 'filth',
            'retard', 'retarded', 'imbecile'
        ]
    
    def load_model(self):
        """Load the mBERT model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✓ Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_thresholds(self):
        """Load language-specific thresholds"""
        thresholds_path = 'models/optimized_thresholds.json'
        if os.path.exists(thresholds_path):
            try:
                with open(thresholds_path, 'r') as f:
                    self.thresholds = json.load(f)
                # Lower English threshold for better detection
                self.thresholds['eng'] = min(self.thresholds.get('eng', 0.65), 0.40)
                logger.info(f"✓ Loaded thresholds: {self.thresholds}")
            except:
                self.thresholds = {
                    'eng': 0.40,
                    'tam': 0.1,
                    'hin': 0.50,
                    'spa': 0.60,
                    'cmn': 0.45
                }
        else:
            self.thresholds = {
                'eng': 0.40,
                'tam': 0.1,
                'hin': 0.50,
                'spa': 0.60,
                'cmn': 0.45
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
            if has_strong_keywords and hate_probability > 0.25:
                is_hate = True
                confidence = max(hate_probability, 0.5)
            else:
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
            logger.error(f"Prediction error: {e}")
            return {
                'is_hate': False,
                'confidence': 0.0,
                'hate_probability': 0.0,
                'neutral_probability': 1.0,
                'language': language,
                'error': str(e)
            }


# Initialize detector
detector = None


@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'device': str(detector.device) if detector else None
    })


@app.route('/api/detect', methods=['POST', 'OPTIONS'])
def detect():
    """Detect hate speech in text"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided',
                'is_hate': False
            }), 400
        
        text = data.get('text', '').strip()
        language = data.get('language', None)  # Optional language hint
        
        if not text:
            return jsonify({
                'error': 'Text is required',
                'is_hate': False
            }), 400
        
        # Predict
        result = detector.predict(text, language)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({
            'error': str(e),
            'is_hate': False
        }), 500


@app.route('/api/detect-batch', methods=['POST'])
def detect_batch():
    """Detect hate speech in multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({
                'error': 'Texts array is required',
                'results': []
            }), 400
        
        results = []
        for text in texts:
            result = detector.predict(text)
            results.append(result)
        
        return jsonify({
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        return jsonify({
            'error': str(e),
            'results': []
        }), 500


def initialize_model():
    """Initialize the model on startup"""
    global detector
    try:
        detector = HateSpeechDetector()
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'false')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response


if __name__ == '__main__':
    # Initialize model
    initialize_model()
    
    # Run server
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}...")
    logger.info(f"API will be available at http://localhost:{port}")
    logger.info(f"Health check: http://localhost:{port}/health")
    logger.info(f"Detection endpoint: http://localhost:{port}/api/detect")
    app.run(host='0.0.0.0', port=port, debug=False)

