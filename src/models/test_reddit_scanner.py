"""
Reddit Comment Scanner and Hate Speech Detector
Scrapes Reddit comments from a URL and identifies exact hate speech words/phrases
"""

import os
import sys
import re
import json
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from urllib.parse import urljoin, urlparse
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class RedditHateSpeechScanner:
    """Scan Reddit comments and detect hate speech with word-level identification"""
    
    def __init__(self, model_path: str = 'models/mbert_improved_chinese_multilingual'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        self.load_thresholds()
        
        # Headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Strong hate keywords for word-level detection
        self.strong_hate_keywords = [
            'kill', 'death', 'murder', 'hate you', 'i hate', 'stupid', 'worthless',
            'idiot', 'moron', 'fool', 'dumb', 'useless', 'trash', 'garbage', 'scum',
            'fuck', 'fucking', 'fuck you', 'fuck off', 'bitch', 'bastard', 'asshole',
            'shit', 'crap', 'retard', 'retarded', 'imbecile', 'damn'
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
        if re.search(r'[\u4e00-\u9fff]+', text_clean):
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]+', text_clean))
            total_chars = len(re.sub(r'\s', '', text_clean))
            if total_chars > 0 and chinese_chars / total_chars > 0.3:
                return 'cmn'
        
        # Check for Tamil script
        if re.search(r'[\u0B80-\u0BFF]+', text_clean):
            return 'tam'
        
        # Check for Hindi/Devanagari script
        if re.search(r'[\u0900-\u097F]+', text_clean):
            return 'hin'
        
        # Check for Spanish accented characters
        if re.search(r'[áéíóúüñÁÉÍÓÚÜÑ]', text_clean):
            return 'spa'
        
        # Default to English
        return 'eng'
    
    def identify_hate_words(self, text: str, hate_probability: float) -> list:
        """Identify specific words/phrases that are hateful"""
        hate_words = []
        text_lower = text.lower()
        
        # Check for strong hate keywords
        for keyword in self.strong_hate_keywords:
            if keyword in text_lower:
                # Find all occurrences
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                matches = pattern.finditer(text)
                for match in matches:
                    hate_words.append({
                        'word': match.group(),
                        'start': match.start(),
                        'end': match.end(),
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
        
        # Remove duplicates and sort by position
        seen = set()
        unique_words = []
        for word_info in sorted(hate_words, key=lambda x: x['start']):
            key = (word_info['start'], word_info['end'])
            if key not in seen:
                seen.add(key)
                unique_words.append(word_info)
        
        return unique_words
    
    def predict(self, text: str, language: str = None) -> dict:
        """Predict if text is hate speech and identify hate words"""
        if not text or not text.strip():
            return {
                'is_hate': False,
                'confidence': 0.0,
                'hate_probability': 0.0,
                'neutral_probability': 1.0,
                'language': 'unknown',
                'hate_words': []
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
            has_strong_keywords = any(kw in text.lower() for kw in self.strong_hate_keywords)
            
            # Decision logic
            if has_strong_keywords and hate_probability > 0.20:
                is_hate = True
                confidence = max(hate_probability, 0.6)
            else:
                is_hate = hate_probability >= threshold
                confidence = hate_probability if is_hate else neutral_probability
            
            # Identify specific hate words
            hate_words = []
            if is_hate:
                hate_words = self.identify_hate_words(text, hate_probability)
            
            return {
                'is_hate': is_hate,
                'confidence': confidence,
                'hate_probability': hate_probability,
                'neutral_probability': neutral_probability,
                'language': language,
                'threshold_used': threshold,
                'hate_words': hate_words
            }
            
        except Exception as e:
            return {
                'is_hate': False,
                'confidence': 0.0,
                'hate_probability': 0.0,
                'neutral_probability': 1.0,
                'language': language,
                'error': str(e),
                'hate_words': []
            }
    
    def scrape_reddit_comments(self, url: str) -> list:
        """Scrape Reddit comments from a URL"""
        print(f"\n🔍 Scraping Reddit comments from: {url}")
        
        # Convert Reddit URL to JSON API format
        if not url.endswith('.json'):
            if url.endswith('/'):
                url = url[:-1]
            url = url + '.json'
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            comments = []
            
            # Reddit JSON structure: [0] is post, [1] is comments
            if len(data) > 1 and 'data' in data[1]:
                comment_tree = data[1]['data']['children']
                comments = self._extract_comments(comment_tree)
            
            print(f"✓ Found {len(comments)} comments")
            return comments
            
        except Exception as e:
            print(f"✗ Error scraping Reddit: {e}")
            print("Trying alternative method with BeautifulSoup...")
            return self._scrape_with_beautifulsoup(url.replace('.json', ''))
    
    def _extract_comments(self, comment_tree, depth=0, max_depth=3) -> list:
        """Recursively extract comments from Reddit JSON structure"""
        comments = []
        
        for item in comment_tree:
            if item.get('kind') == 't1':  # Comment
                comment_data = item.get('data', {})
                
                # Skip deleted/removed comments
                if comment_data.get('body') in ['[deleted]', '[removed]']:
                    continue
                
                body = comment_data.get('body', '').strip()
                if body and len(body) >= 3:
                    comments.append({
                        'text': body,
                        'author': comment_data.get('author', 'unknown'),
                        'score': comment_data.get('score', 0),
                        'depth': depth,
                        'permalink': comment_data.get('permalink', '')
                    })
                
                # Recursively get replies
                if depth < max_depth and 'replies' in comment_data:
                    replies_data = comment_data['replies']
                    if isinstance(replies_data, dict) and 'data' in replies_data:
                        replies = replies_data['data'].get('children', [])
                        comments.extend(self._extract_comments(replies, depth + 1, max_depth))
        
        return comments
    
    def _scrape_with_beautifulsoup(self, url: str) -> list:
        """Fallback: scrape with BeautifulSoup if JSON API fails"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            comments = []
            # Try to find comment elements (this is less reliable)
            comment_elements = soup.find_all(['div', 'p'], class_=re.compile(r'comment|md'))
            
            for elem in comment_elements:
                text = elem.get_text(strip=True)
                if text and len(text) >= 10 and text not in ['[deleted]', '[removed]']:
                    comments.append({
                        'text': text,
                        'author': 'unknown',
                        'score': 0,
                        'depth': 0,
                        'permalink': ''
                    })
            
            return comments[:50]  # Limit to 50 comments
            
        except Exception as e:
            print(f"✗ BeautifulSoup scraping also failed: {e}")
            return []
    
    def scan_reddit_url(self, url: str) -> dict:
        """Main function: scan Reddit URL for hate speech"""
        print("\n" + "="*70)
        print("Reddit Hate Speech Scanner")
        print("="*70)
        
        # Scrape comments
        comments = self.scrape_reddit_comments(url)
        
        if not comments:
            print("✗ No comments found. The URL might be invalid or comments are disabled.")
            return {
                'url': url,
                'total_comments': 0,
                'hate_comments': [],
                'summary': {}
            }
        
        print(f"\n📊 Analyzing {len(comments)} comments for hate speech...")
        
        hate_comments = []
        total_scanned = 0
        
        for i, comment in enumerate(comments, 1):
            text = comment['text']
            result = self.predict(text)
            total_scanned += 1
            
            if result['is_hate']:
                hate_comments.append({
                    'comment_number': i,
                    'text': text,
                    'author': comment['author'],
                    'score': comment['score'],
                    'hate_probability': result['hate_probability'],
                    'confidence': result['confidence'],
                    'language': result['language'],
                    'hate_words': result['hate_words']
                })
                
                print(f"\n🔴 HATE SPEECH DETECTED (#{i})")
                print(f"   Author: {comment['author']}")
                print(f"   Score: {comment['score']}")
                print(f"   Confidence: {result['confidence']*100:.1f}%")
                print(f"   Text: {text[:200]}...")
                
                if result['hate_words']:
                    print(f"   🎯 Hate Words/Phrases Found:")
                    for word_info in result['hate_words']:
                        print(f"      - '{word_info['word']}' ({word_info['type']}, {word_info['confidence']} confidence)")
                        # Show context
                        start = max(0, word_info['start'] - 20)
                        end = min(len(text), word_info['end'] + 20)
                        context = text[start:end]
                        print(f"        Context: ...{context}...")
            
            # Progress indicator
            if i % 10 == 0:
                print(f"   Scanned {i}/{len(comments)} comments...")
        
        # Summary
        summary = {
            'total_comments': len(comments),
            'hate_comments_count': len(hate_comments),
            'hate_percentage': (len(hate_comments) / len(comments) * 100) if comments else 0,
            'total_scanned': total_scanned
        }
        
        print("\n" + "="*70)
        print("SCAN SUMMARY")
        print("="*70)
        print(f"Total Comments: {summary['total_comments']}")
        print(f"Hate Speech Detected: {summary['hate_comments_count']}")
        print(f"Hate Speech Percentage: {summary['hate_percentage']:.1f}%")
        print("="*70 + "\n")
        
        return {
            'url': url,
            'total_comments': summary['total_comments'],
            'hate_comments': hate_comments,
            'summary': summary
        }


def main():
    """Main function"""
    # Default test URL
    test_url = "https://www.reddit.com/r/PublicFreakout/comments/uk5kcj/racist_girl_gets_taught_a_lesson/"
    
    print("\n" + "="*70)
    print("Reddit Hate Speech Scanner - Test Tool")
    print("="*70)
    print(f"Test URL: {test_url}")
    print("="*70)
    
    # Initialize scanner
    try:
        scanner = RedditHateSpeechScanner()
    except Exception as e:
        print(f"Failed to initialize scanner: {e}")
        print("\nMake sure:")
        print("1. Model files exist in models/mbert_improved_chinese_multilingual/")
        print("2. You're running from the project root directory")
        return
    
    # Scan the URL
    results = scanner.scan_reddit_url(test_url)
    
    # Save results to file
    output_file = 'reddit_scan_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print detailed results
    if results['hate_comments']:
        print("\n📋 DETAILED HATE SPEECH RESULTS:")
        print("="*70)
        for i, hate_comment in enumerate(results['hate_comments'], 1):
            print(f"\n{i}. Comment #{hate_comment['comment_number']}")
            print(f"   Author: {hate_comment['author']}")
            print(f"   Hate Probability: {hate_comment['hate_probability']*100:.1f}%")
            print(f"   Full Text: {hate_comment['text']}")
            print(f"   Hate Words/Phrases:")
            for word_info in hate_comment['hate_words']:
                print(f"      • '{word_info['word']}' (position {word_info['start']}-{word_info['end']})")
    else:
        print("\n✓ No hate speech detected in comments!")


if __name__ == '__main__':
    main()

