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
# Enable CORS for browser extension with proper headers - allow all origins
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept", "Authorization"]
    },
    r"/health": {
        "origins": "*",
        "methods": ["GET", "OPTIONS", "POST"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})
# Also set CORS headers globally as fallback
@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'false')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

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
        
        # Strong hate keywords for override and word-level detection
        self.strong_hate_keywords = [
            'kill', 'death', 'murder', 'hate you', 'i hate', 'stupid', 'worthless',
            'idiot', 'moron', 'fool', 'dumb', 'useless', 'trash', 'garbage', 'scum',
            'fuck', 'fucking', 'fuck you', 'fuck off', 'bitch', 'bastard', 'asshole',
            'shit', 'crap', 'retard', 'retarded', 'imbecile', 'damn'
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
        """Load language-specific thresholds from file or use defaults"""
        thresholds_path = 'models/optimized_thresholds.json'
        default_thresholds = {
            'eng': 0.30,
            'tam': 0.1,
            'hin': 0.35,
            'spa': 0.40,
            'cmn': 0.35
        }
        
        if os.path.exists(thresholds_path):
            try:
                with open(thresholds_path, 'r') as f:
                    loaded_thresholds = json.load(f)
                # Use loaded thresholds, but ensure all languages have values
                self.thresholds = {**default_thresholds, **loaded_thresholds}
                logger.info(f"✓ Loaded thresholds: {self.thresholds}")
            except Exception as e:
                logger.warn(f"Failed to load thresholds from file: {e}, using defaults")
                self.thresholds = default_thresholds
        else:
            self.thresholds = default_thresholds
            logger.info(f"Using default thresholds: {self.thresholds}")
    
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
    
    def _identify_hate_words(self, text: str, hate_probability: float) -> list:
        """Identify specific words/phrases that are hateful"""
        import re
        hate_words = []
        text_lower = text.lower()
        
        # Check for strong hate keywords (use word boundaries to avoid false positives)
        for keyword in self.strong_hate_keywords:
            # Use word boundaries for single words, but allow phrases
            if len(keyword.split()) == 1:
                # Single word - use word boundary
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            else:
                # Phrase - match as is
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            
            matches = pattern.finditer(text)
            for match in matches:
                # Double check it's not part of a larger word
                matched_word = match.group()
                start = match.start()
                end = match.end()
                
                # Verify it's actually the keyword (not part of another word)
                if len(keyword.split()) == 1:
                    # Check character before and after
                    if (start > 0 and text[start-1].isalnum()) or \
                       (end < len(text) and text[end].isalnum()):
                        continue  # Skip if part of larger word
                
                hate_words.append({
                    'word': matched_word,
                    'start': start,
                    'end': end,
                    'type': 'strong_keyword',
                    'confidence': 'high'
                })
        
        # If hate probability is high, analyze sentence structure
        if hate_probability > 0.5:
            # Look for insult patterns
            insult_patterns = [
                r'\b(you|u)\s+(are|r)\s+(stupid|dumb|idiot|worthless|useless)\b',
                r'\b(you|u)\s+(should|need to)\s+(die|kill yourself|go away)\b',
                r'\b(i|I)\s+hate\s+(you|u|your)\b',
                r'\b(fuck|fucking)\s+(you|off|yourself)\b',
            ]
            
            for pattern in insult_patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    # Find original case version
                    orig_match = re.search(re.escape(match.group()), text, re.IGNORECASE)
                    if orig_match:
                        hate_words.append({
                            'word': orig_match.group(),
                            'start': orig_match.start(),
                            'end': orig_match.end(),
                            'type': 'insult_pattern',
                            'confidence': 'high' if hate_probability > 0.7 else 'medium'
                        })
        
        # Remove duplicates and overlapping words (keep longer phrases)
        # Sort by length (longest first) then by position
        sorted_words = sorted(hate_words, key=lambda x: (x['end'] - x['start'], -x['start']), reverse=True)
        
        unique_words = []
        seen_positions = set()
        
        for word_info in sorted_words:
            # Check if this word overlaps with any already added word
            overlaps = False
            for start, end in seen_positions:
                # Check if word_info overlaps with (start, end)
                if not (word_info['end'] <= start or word_info['start'] >= end):
                    overlaps = True
                    break
            
            if not overlaps:
                unique_words.append(word_info)
                seen_positions.add((word_info['start'], word_info['end']))
        
        # Sort by position for final output
        unique_words.sort(key=lambda x: x['start'])
        
        return unique_words
    
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
            
            # Check for strong keywords (minimal, only extreme cases)
            has_strong_keywords = self.has_strong_hate_keywords(text)
            
            # Decision logic - rely primarily on model, use lower threshold
            # Only override for extreme cases (kill, murder, etc.)
            if has_strong_keywords and hate_probability > 0.20:
                is_hate = True
                confidence = max(hate_probability, 0.6)
            else:
                # Let model decide with lower threshold
                is_hate = hate_probability >= threshold
                confidence = hate_probability if is_hate else neutral_probability
            
            # Identify specific hate words
            hate_words = []
            if is_hate:
                hate_words = self._identify_hate_words(text, hate_probability)
            
            return {
                'is_hate': is_hate,
                'confidence': confidence,
                'hate_probability': hate_probability,
                'neutral_probability': neutral_probability,
                'language': language,
                'threshold_used': threshold,
                'keyword_override': has_strong_keywords,
                'hate_words': hate_words
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


if __name__ == '__main__':
    # Initialize model
    initialize_model()
    
    # Run server - try port 5000, if busy try 5001
    port = int(os.environ.get('PORT', 5000))
    
    # Check if port is available, if not try 5001
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    
    if result == 0:
        # Port is in use, try 5001
        logger.warn(f"Port {port} is in use, trying port 5001...")
        port = 5001
        # Update API URL in config if needed
        os.environ['PORT'] = str(port)
    
    logger.info(f"Starting server on port {port}...")
    logger.info(f"API will be available at http://localhost:{port}")
    logger.info(f"Health check: http://localhost:{port}/health")
    logger.info(f"Detection endpoint: http://localhost:{port}/api/detect")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is still in use. Please stop the process using it or use a different port.")
            logger.error("On macOS, you may need to disable AirPlay Receiver in System Settings.")
            logger.error("Or kill the process: kill -9 $(lsof -ti:{port})")
        raise

