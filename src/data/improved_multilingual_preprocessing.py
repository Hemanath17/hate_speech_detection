import os
import sys
import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Tuple
import bz2
from collections import Counter

# Ensure project root is on path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedMultilingualPreprocessor:
    def __init__(self):
        self.language_patterns = {
            'tamil': {
                'hate_keywords': [
                    # Tamil script hate words
                    'வேசி', 'பெண்', 'பரத்தை', 'விலைமாது', 'காமக்கிழத்தி',
                    'முஸ்லிம்', 'தீவிரவாதி', 'குண்டு', 'கொலை', 'வெட்டு',
                    'கே', 'லெஸ்பியன்', 'திருநங்கை', 'திருநங்கை',
                    'பெண்கள்', 'சமையலறை', 'வீடு', 'வீட்டுக்கு',
                    'ஊனமுற்ற', 'பயனற்ற', 'வெறுப்பு', 'வெறுக்கத்தக்க',
                    'மதம்', 'இந்து', 'கிறிஸ்தவ', 'இஸ்லாம்',
                    'சாதி', 'தலித்', 'பிராமண', 'தீண்டாத'
                ],
                'neutral_keywords': [
                    # Tamil script neutral words
                    'நல்ல', 'அழகான', 'நன்றாக', 'அற்புதமான',
                    'குடும்பம்', 'காதல்', 'நன்றி', 'வரவேற்பு', 'உதவி',
                    'வானிலை', 'நாள்', 'மாலை', 'காலை', 'இரவு',
                    'உணவு', 'நீர்', 'வீடு', 'பள்ளி', 'வேலை',
                    'நண்பன்', 'மகிழ்ச்சி', 'புன்னகை', 'சிரிப்பு', 'மகிழ்ச்சி'
                ]
            },
            'hindi': {
                'hate_keywords': [
                    # Hindi/Devanagari script hate words
                    'बेकार', 'अनुपयोगी', 'बेमूल्य', 'मूर्ख', 'बेवकूफ',
                    'आतंकवादी', 'बम', 'मारना', 'हत्या', 'काटना',
                    'गे', 'लेस्बियन', 'समलैंगिक', 'ट्रांसजेंडर',
                    'औरत', 'महिला', 'रसोई', 'घर', 'घर में',
                    'अपंग', 'अक्षम', 'अनुपयोगी', 'घृणित',
                    'धर्म', 'हिंदू', 'ईसाई', 'इस्लाम', 'मुस्लिम',
                    'जाति', 'दलित', 'ब्राह्मण', 'अछूत'
                ],
                'neutral_keywords': [
                    # Hindi/Devanagari script neutral words
                    'अच्छा', 'सुंदर', 'अच्छा', 'अद्भुत',
                    'परिवार', 'प्यार', 'धन्यवाद', 'स्वागत', 'मदद',
                    'मौसम', 'दिन', 'शाम', 'सुबह', 'रात',
                    'भोजन', 'पानी', 'घर', 'स्कूल', 'काम',
                    'दोस्त', 'खुश', 'मुस्कान', 'हंसी', 'आनंद'
                ]
            },
            'spanish': {
                'hate_keywords': [
                    'estúpida', 'estupida', 'stupid', 'idiot', 'imbecil',
                    'terrorista', 'terrorist', 'bomb', 'kill', 'matar',
                    'gay', 'lesbiana', 'homosexual', 'transgender',
                    'mujer', 'woman', 'cocina', 'kitchen', 'casa', 'home',
                    'discapacitado', 'handicap', 'inútil', 'useless',
                    'religión', 'religion', 'cristiano', 'musulmán'
                ],
                'neutral_keywords': [
                    'bueno', 'good', 'hermoso', 'beautiful', 'nice', 'wonderful',
                    'familia', 'family', 'amor', 'love', 'gracias', 'thank',
                    'bienvenido', 'welcome', 'ayuda', 'help',
                    'clima', 'weather', 'día', 'day', 'tarde', 'evening',
                    'comida', 'food', 'agua', 'water', 'casa', 'house',
                    'amigo', 'friend', 'feliz', 'happy', 'sonrisa', 'smile'
                ]
            },
            'chinese': {
                'hate_keywords': [
                    # Chinese characters hate words
                    '笨', '愚蠢', '白痴', '傻瓜', '傻子',
                    '恐怖分子', '炸弹', '杀死', '死', '死亡',
                    '同性恋', '女同性恋', '男同性恋', '跨性别',
                    '女人', '女性', '家里', '厨房', '家庭',
                    '残疾', '残废', '无用', '无价值',
                    '宗教', '印度教', '基督教', '伊斯兰教', '穆斯林'
                ],
                'neutral_keywords': [
                    # Chinese characters neutral words
                    '好', '美丽', '漂亮', '精彩', '美妙',
                    '家庭', '爱', '谢谢', '感谢', '欢迎', '帮助',
                    '天气', '日子', '晚上', '早晨', '夜晚',
                    '食物', '水', '房子', '学校', '工作',
                    '朋友', '开心', '快乐', '微笑', '笑声', '欢乐'
                ]
            },
            'english': {
                'hate_keywords': [
                    'stupid', 'idiot', 'moron', 'fool', 'dumb',
                    'terrorist', 'bomb', 'kill', 'die', 'murder',
                    'gay', 'lesbian', 'homosexual', 'transgender',
                    'woman', 'women', 'kitchen', 'home', 'house',
                    'disability', 'handicap', 'useless', 'worthless',
                    'religion', 'hindu', 'christian', 'islam', 'muslim'
                ],
                'neutral_keywords': [
                    'good', 'beautiful', 'nice', 'wonderful', 'great',
                    'family', 'love', 'thank', 'welcome', 'help',
                    'weather', 'day', 'evening', 'morning', 'night',
                    'food', 'water', 'house', 'school', 'work',
                    'friend', 'happy', 'smile', 'laugh', 'joy'
                ]
            }
        }
    
    def clean_text(self, text: str, language: str) -> str:
        """Clean and normalize text for better processing"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip().lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Language-specific cleaning
        if language in ['tamil', 'hindi']:
            # Remove common transliteration artifacts
            text = re.sub(r'[^\w\s]', ' ', text)
        elif language == 'chinese':
            # Keep Chinese characters and basic punctuation
            text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        elif language == 'spanish':
            # Keep Spanish accented characters
            text = re.sub(r'[^\w\sáéíóúüñ]', ' ', text)
        
        return text.strip()
    
    def is_hate_speech(self, text: str, language: str) -> Tuple[bool, float]:
        """Determine if text is hate speech using improved pattern matching"""
        text = self.clean_text(text, language)
        
        if not text:
            return False, 0.0
        
        hate_keywords = self.language_patterns[language]['hate_keywords']
        neutral_keywords = self.language_patterns[language]['neutral_keywords']
        
        # Count keyword matches
        hate_matches = sum(1 for keyword in hate_keywords if keyword in text)
        neutral_matches = sum(1 for keyword in neutral_keywords if keyword in text)
        
        # Calculate confidence score
        total_matches = hate_matches + neutral_matches
        if total_matches == 0:
            return False, 0.0
        
        hate_ratio = hate_matches / total_matches
        neutral_ratio = neutral_matches / total_matches
        
        # Decision logic
        if hate_matches >= 2:  # Strong hate indicators
            return True, min(0.9, 0.5 + hate_ratio * 0.4)
        elif hate_matches == 1 and neutral_matches == 0:  # Single hate word, no neutral
            return True, 0.6
        elif hate_matches == 1 and neutral_matches <= 1:  # Single hate word, few neutral
            return True, 0.5
        elif hate_ratio > 0.6:  # High hate ratio
            return True, 0.4 + hate_ratio * 0.3
        else:
            return False, neutral_ratio * 0.3
    
    def create_balanced_dataset(self, target_samples_per_language: int = 2000) -> pd.DataFrame:
        """Create a more balanced multilingual dataset"""
        logger.info("Creating improved balanced multilingual dataset...")
        
        all_data = []
        
        # Process each language
        for lang_code, lang_name in [('tam', 'tamil'), ('hin', 'hindi'), 
                                   ('spa', 'spanish'), ('cmn', 'chinese'), ('eng', 'english')]:
            logger.info(f"Processing {lang_name}...")
            
            if lang_code == 'eng':
                # Use ETHOS dataset for English
                from configs.config import DATA_PATHS
                df = pd.read_csv(DATA_PATHS['ethos_binary'], sep=';')
                texts = df['comment'].astype(str).tolist()
                labels = (df['isHate'] > 0).astype(int).tolist()
            else:
                # Load language-specific dataset
                data_path = f'Data/{lang_code}_sentences_detailed.tsv.bz2'
                if not os.path.exists(data_path):
                    logger.warning(f"Dataset not found: {data_path}")
                    continue
                
                with bz2.open(data_path, 'rt', encoding='utf-8') as f:
                    lines = f.readlines()
                
                texts = []
                for line in lines[:5000]:  # Limit to 5000 samples
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:  # Text is in 3rd column (index 2)
                        text = parts[2].strip()
                        if text and text != '\\N':  # Skip empty or null values
                            texts.append(text)
            
            # Process texts and create labels
            processed_texts = []
            labels = []
            hate_count = 0
            neutral_count = 0
            
            for text in texts:
                if len(text.strip()) < 10:  # Skip very short texts
                    continue
                
                is_hate, confidence = self.is_hate_speech(text, lang_name)
                
                # Balance the dataset
                if is_hate and hate_count < target_samples_per_language // 2:
                    processed_texts.append(text)
                    labels.append(1)
                    hate_count += 1
                elif not is_hate and neutral_count < target_samples_per_language // 2:
                    processed_texts.append(text)
                    labels.append(0)
                    neutral_count += 1
                
                # Stop if we have enough samples
                if len(processed_texts) >= target_samples_per_language:
                    break
            
            # Add to combined dataset
            for text, label in zip(processed_texts, labels):
                all_data.append({
                    'text': text,
                    'isHate': label,
                    'original_language': lang_code,
                    'confidence': self.is_hate_speech(text, lang_name)[1]
                })
            
            logger.info(f"{lang_name}: {len(processed_texts)} samples, "
                       f"{hate_count} hate, {neutral_count} neutral")
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Created improved dataset with {len(df)} samples")
        logger.info(f"Overall hate speech ratio: {df['isHate'].mean():.2%}")
        
        # Language distribution
        lang_dist = df['original_language'].value_counts()
        logger.info("Language distribution:")
        for lang, count in lang_dist.items():
            hate_ratio = df[df['original_language'] == lang]['isHate'].mean()
            logger.info(f"  {lang}: {count} samples, {hate_ratio:.1%} hate")
        
        return df


def main():
    preprocessor = ImprovedMultilingualPreprocessor()
    
    # Create improved dataset
    df = preprocessor.create_balanced_dataset(target_samples_per_language=2000)
    
    # Save dataset
    output_path = 'Data/improved_multilingual_hate_speech_v2.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Improved dataset saved to: {output_path}")
    
    # Show sample data
    logger.info("\nSample data:")
    print(df.head(10))
    
    # Show statistics
    logger.info(f"\nDataset statistics:")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Hate speech ratio: {df['isHate'].mean():.1%}")
    logger.info(f"Language distribution:")
    print(df['original_language'].value_counts())


if __name__ == "__main__":
    main()
