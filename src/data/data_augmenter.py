import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import random
from pathlib import Path
import re
from tqdm import tqdm

# Ensure project root is on path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedDataAugmenter:
    """
    Advanced data augmentation pipeline for hate speech detection.
    Uses multiple augmentation techniques to generate synthetic data.
    """
    
    def __init__(self, target_samples_per_language: int = 5000):
        self.target_samples_per_language = target_samples_per_language
        
        # Hate speech patterns for different languages
        self.hate_patterns = {
            'eng': {
                'templates': [
                    "You are a {adjective} {target} and should {action}",
                    "All {group} are {negative_adj}",
                    "I hate {target} people",
                    "{target} belong in {place}",
                    "Kill all {group}",
                    "I wish {target} would {action}",
                    "{target} are {negative_adj} and {negative_adj2}",
                    "I hope {target} {action}",
                    "{group} should {action}",
                    "I despise {target}"
                ],
                'adjectives': ['stupid', 'idiot', 'moron', 'fool', 'dumb', 'useless', 'worthless'],
                'targets': ['woman', 'women', 'gay', 'lesbian', 'transgender', 'immigrant', 'muslim', 'jew', 'black', 'white'],
                'groups': ['muslims', 'jews', 'gays', 'women', 'immigrants', 'blacks', 'whites', 'asians'],
                'negative_adj': ['stupid', 'useless', 'worthless', 'disgusting', 'pathetic', 'terrible'],
                'negative_adj2': ['disgusting', 'pathetic', 'terrible', 'awful', 'horrible'],
                'actions': ['die', 'disappear', 'go away', 'suffer', 'burn', 'rot'],
                'places': ['kitchen', 'home', 'hell', 'jail', 'prison']
            },
            'tam': {
                'templates': [
                    "நீ ஒரு {adjective} {target}",
                    "எல்லா {group}யும் {negative_adj}",
                    "நான் {target}வை வெறுக்கிறேன்",
                    "{target} {place}இல் இருக்க வேண்டும்",
                    "எல்லா {group}யும் {action}",
                    "நான் {target}வை {negative_adj} என்று நினைக்கிறேன்",
                    "{target} {negative_adj} மற்றும் {negative_adj2}",
                    "நான் {target}வை {action} விரும்புகிறேன்",
                    "{group} {action} வேண்டும்",
                    "நான் {target}வை வெறுக்கிறேன்"
                ],
                'adjectives': ['முட்டாள்', 'வேசி', 'பயனற்ற', 'வெறுக்கத்தக்க', 'அருவருப்பான'],
                'targets': ['பெண்', 'பெண்கள்', 'கே', 'லெஸ்பியன்', 'திருநங்கை', 'வெளிநாட்டவர்', 'முஸ்லிம்', 'யூதர்'],
                'groups': ['முஸ்லிம்கள்', 'யூதர்கள்', 'கேக்கள்', 'பெண்கள்', 'வெளிநாட்டவர்கள்'],
                'negative_adj': ['முட்டாள்', 'பயனற்ற', 'வெறுக்கத்தக்க', 'அருவருப்பான', 'பயங்கரமான'],
                'negative_adj2': ['அருவருப்பான', 'பயங்கரமான', 'வெறுக்கத்தக்க', 'அருவருப்பான'],
                'actions': ['சாக', 'மறைந்துவிடு', 'தொலைந்துவிடு', 'பாதிக்கப்படு', 'எரிந்துவிடு'],
                'places': ['சமையலறை', 'வீடு', 'நரகம்', 'சிறை', 'காவல் நிலையம்']
            },
            'hin': {
                'templates': [
                    "तुम एक {adjective} {target} हो",
                    "सभी {group} {negative_adj} हैं",
                    "मैं {target} से नफरत करता हूं",
                    "{target} {place} में रहना चाहिए",
                    "सभी {group} को {action}",
                    "मैं {target} को {negative_adj} मानता हूं",
                    "{target} {negative_adj} और {negative_adj2} हैं",
                    "मैं चाहता हूं कि {target} {action}",
                    "{group} को {action} चाहिए",
                    "मैं {target} से घृणा करता हूं"
                ],
                'adjectives': ['बेकार', 'मूर्ख', 'बेवकूफ', 'अनुपयोगी', 'घृणित'],
                'targets': ['औरत', 'महिला', 'गे', 'लेस्बियन', 'ट्रांसजेंडर', 'प्रवासी', 'मुस्लिम', 'यहूदी'],
                'groups': ['मुसलमान', 'यहूदी', 'गे', 'महिलाएं', 'प्रवासी'],
                'negative_adj': ['बेकार', 'अनुपयोगी', 'घृणित', 'बेहूदा', 'भयानक'],
                'negative_adj2': ['बेहूदा', 'भयानक', 'घृणित', 'अरुचिकर'],
                'actions': ['मरो', 'गायब हो जाओ', 'दूर हो जाओ', 'पीड़ित हो', 'जलो'],
                'places': ['रसोई', 'घर', 'नरक', 'जेल', 'कैद']
            },
            'spa': {
                'templates': [
                    "Eres una {adjective} {target} y deberías {action}",
                    "Todos los {group} son {negative_adj}",
                    "Odio a los {target}",
                    "{target} pertenecen a {place}",
                    "Mata a todos los {group}",
                    "Deseo que {target} {action}",
                    "{target} son {negative_adj} y {negative_adj2}",
                    "Espero que {target} {action}",
                    "{group} deberían {action}",
                    "Desprecio a los {target}"
                ],
                'adjectives': ['estúpida', 'idiota', 'inútil', 'despreciable', 'repugnante'],
                'targets': ['mujer', 'mujeres', 'gay', 'lesbiana', 'transgénero', 'inmigrante', 'musulmán', 'judío'],
                'groups': ['musulmanes', 'judíos', 'gays', 'mujeres', 'inmigrantes'],
                'negative_adj': ['estúpidos', 'inútiles', 'despreciables', 'repugnantes', 'terribles'],
                'negative_adj2': ['repugnantes', 'terribles', 'despreciables', 'horribles'],
                'actions': ['morir', 'desaparecer', 'irse', 'sufrir', 'arder'],
                'places': ['cocina', 'casa', 'infierno', 'cárcel', 'prisión']
            },
            'cmn': {
                'templates': [
                    "你是个{adjective}{target}应该{action}",
                    "所有{group}都是{negative_adj}",
                    "我恨{target}",
                    "{target}应该待在{place}",
                    "杀死所有{group}",
                    "我希望{target}{action}",
                    "{target}是{negative_adj}和{negative_adj2}",
                    "我希望{target}{action}",
                    "{group}应该{action}",
                    "我鄙视{target}"
                ],
                'adjectives': ['笨', '愚蠢', '白痴', '无用', '可恨'],
                'targets': ['女人', '女性', '同性恋', '女同性恋', '跨性别', '移民', '穆斯林', '犹太人'],
                'groups': ['穆斯林', '犹太人', '同性恋', '女性', '移民'],
                'negative_adj': ['愚蠢', '无用', '可恨', '恶心', '可怕'],
                'negative_adj2': ['恶心', '可怕', '可恨', '讨厌'],
                'actions': ['死', '消失', '走开', '受苦', '燃烧'],
                'places': ['厨房', '家里', '地狱', '监狱', '牢房']
            }
        }
        
        # Neutral patterns for generating non-hate examples
        self.neutral_patterns = {
            'eng': [
                "This is a {positive_adj} day",
                "I love my {family_member}",
                "The weather is {weather_adj} today",
                "Thank you for your {help_type}",
                "Have a {positive_adj} {time_of_day}",
                "I enjoy {activity}",
                "This {object} is {positive_adj}",
                "I'm {emotion} about {topic}",
                "Let's {action} together",
                "I hope you have a {positive_adj} day"
            ],
            'tam': [
                "இன்று {positive_adj} நாள்",
                "நான் என் {family_member}வை காதலிக்கிறேன்",
                "இன்று வானிலை {weather_adj}",
                "உங்கள் {help_type}க்கு நன்றி",
                "{time_of_day} {positive_adj}",
                "நான் {activity} விரும்புகிறேன்",
                "இந்த {object} {positive_adj}",
                "நான் {topic} பற்றி {emotion}",
                "நாம் ஒன்றாக {action} செய்யலாம்",
                "உங்களுக்கு {positive_adj} நாள் கிடைக்கட்டும்"
            ],
            'hin': [
                "आज {positive_adj} दिन है",
                "मैं अपने {family_member} से प्यार करता हूं",
                "आज मौसम {weather_adj} है",
                "आपकी {help_type} के लिए धन्यवाद",
                "{time_of_day} {positive_adj}",
                "मुझे {activity} पसंद है",
                "यह {object} {positive_adj} है",
                "मैं {topic} के बारे में {emotion} हूं",
                "चलिए मिलकर {action} करते हैं",
                "आपका दिन {positive_adj} हो"
            ],
            'spa': [
                "Hoy es un día {positive_adj}",
                "Amo a mi {family_member}",
                "El clima está {weather_adj} hoy",
                "Gracias por tu {help_type}",
                "Que tengas una {time_of_day} {positive_adj}",
                "Disfruto {activity}",
                "Este {object} es {positive_adj}",
                "Estoy {emotion} sobre {topic}",
                "Hagamos {action} juntos",
                "Espero que tengas un día {positive_adj}"
            ],
            'cmn': [
                "今天是{positive_adj}的一天",
                "我爱我的{family_member}",
                "今天天气{weather_adj}",
                "谢谢你的{help_type}",
                "祝你{time_of_day}{positive_adj}",
                "我喜欢{activity}",
                "这个{object}很{positive_adj}",
                "我对{topic}感到{emotion}",
                "我们一起{action}吧",
                "希望你今天{positive_adj}"
            ]
        }
        
        # Common words for neutral generation
        self.neutral_words = {
            'positive_adj': ['好', '美丽', '精彩', '美妙', '优秀', '棒', 'great', 'beautiful', 'wonderful', 'excellent', 'amazing', 'fantastic'],
            'family_member': ['家人', '父母', '朋友', 'family', 'parents', 'friends', 'loved ones'],
            'weather_adj': ['好', '美丽', '晴朗', 'nice', 'beautiful', 'sunny', 'pleasant'],
            'help_type': ['帮助', '支持', 'help', 'support', 'assistance'],
            'time_of_day': ['早上', '下午', '晚上', 'morning', 'afternoon', 'evening'],
            'activity': ['阅读', '学习', '工作', 'reading', 'learning', 'working'],
            'object': ['书', '电影', '音乐', 'book', 'movie', 'music'],
            'emotion': ['高兴', '兴奋', 'happy', 'excited', 'pleased'],
            'topic': ['工作', '学习', '生活', 'work', 'study', 'life'],
            'action': ['学习', '工作', '玩耍', 'learn', 'work', 'play']
        }
    
    def generate_hate_speech(self, language: str, num_samples: int) -> List[Dict]:
        """
        Generate synthetic hate speech samples for a specific language.
        
        Args:
            language: Language code (eng, tam, hin, spa, cmn)
            num_samples: Number of samples to generate
            
        Returns:
            List of generated hate speech samples
        """
        if language not in self.hate_patterns:
            logger.warning(f"No hate patterns for language: {language}")
            return []
        
        patterns = self.hate_patterns[language]
        generated_samples = []
        
        for _ in range(num_samples):
            # Select random template
            template = random.choice(patterns['templates'])
            
            # Fill in placeholders
            text = template
            for placeholder, word_list in patterns.items():
                if placeholder != 'templates':
                    if f"{{{placeholder}}}" in text:
                        word = random.choice(word_list)
                        text = text.replace(f"{{{placeholder}}}", word)
            
            generated_samples.append({
                'text': text,
                'isHate': 1,
                'original_language': language,
                'source': 'synthetic_hate'
            })
        
        return generated_samples
    
    def generate_neutral_speech(self, language: str, num_samples: int) -> List[Dict]:
        """
        Generate synthetic neutral speech samples for a specific language.
        
        Args:
            language: Language code (eng, tam, hin, spa, cmn)
            num_samples: Number of samples to generate
            
        Returns:
            List of generated neutral speech samples
        """
        if language not in self.neutral_patterns:
            logger.warning(f"No neutral patterns for language: {language}")
            return []
        
        patterns = self.neutral_patterns[language]
        generated_samples = []
        
        for _ in range(num_samples):
            # Select random template
            template = random.choice(patterns)
            
            # Fill in placeholders
            text = template
            for placeholder, word_list in self.neutral_words.items():
                if f"{{{placeholder}}}" in text:
                    word = random.choice(word_list)
                    text = text.replace(f"{{{placeholder}}}", word)
            
            generated_samples.append({
                'text': text,
                'isHate': 0,
                'original_language': language,
                'source': 'synthetic_neutral'
            })
        
        return generated_samples
    
    def augment_existing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment existing dataset with synthetic data.
        
        Args:
            df: Existing dataset
            
        Returns:
            Augmented dataset
        """
        logger.info("Starting data augmentation...")
        
        # Calculate samples needed per language
        language_counts = df['original_language'].value_counts()
        augmented_data = []
        
        for language in ['eng', 'tam', 'hin', 'spa', 'cmn']:
            current_count = language_counts.get(language, 0)
            hate_count = df[(df['original_language'] == language) & (df['isHate'] == 1)].shape[0]
            neutral_count = df[(df['original_language'] == language) & (df['isHate'] == 0)].shape[0]
            
            logger.info(f"Language {language}: {current_count} total, {hate_count} hate, {neutral_count} neutral")
            
            # Calculate how many more samples we need
            needed_samples = self.target_samples_per_language - current_count
            
            if needed_samples > 0:
                # Generate synthetic data
                hate_needed = max(0, (self.target_samples_per_language // 2) - hate_count)
                neutral_needed = max(0, (self.target_samples_per_language // 2) - neutral_count)
                
                # Generate hate speech
                if hate_needed > 0:
                    hate_samples = self.generate_hate_speech(language, hate_needed)
                    augmented_data.extend(hate_samples)
                    logger.info(f"Generated {len(hate_samples)} hate samples for {language}")
                
                # Generate neutral speech
                if neutral_needed > 0:
                    neutral_samples = self.generate_neutral_speech(language, neutral_needed)
                    augmented_data.extend(neutral_samples)
                    logger.info(f"Generated {len(neutral_samples)} neutral samples for {language}")
        
        # Convert to DataFrame
        if augmented_data:
            augmented_df = pd.DataFrame(augmented_data)
            
            # Combine with original data
            combined_df = pd.concat([df, augmented_df], ignore_index=True)
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['text'])
            
            # Shuffle
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            logger.info(f"Augmentation complete: {len(combined_df)} total samples")
            logger.info(f"Added {len(augmented_data)} synthetic samples")
            
            return combined_df
        else:
            logger.info("No augmentation needed")
            return df
    
    def save_augmented_data(self, df: pd.DataFrame, filename: str = "augmented_hate_speech_data.csv"):
        """
        Save augmented data to file.
        
        Args:
            df: Augmented DataFrame
            filename: Output filename
        """
        file_path = Path("Data") / filename
        df.to_csv(file_path, index=False)
        logger.info(f"Saved augmented data to {file_path}")
        
        # Save statistics
        stats_path = Path("Data") / "augmentation_stats.json"
        stats = {
            'total_samples': len(df),
            'hate_speech_ratio': float(df['isHate'].mean()),
            'sources': df['source'].value_counts().to_dict(),
            'languages': df['original_language'].value_counts().to_dict()
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved augmentation statistics to {stats_path}")


def main():
    """Main function to run data augmentation"""
    logger.info("=" * 80)
    logger.info("HATE SPEECH DATA AUGMENTATION PIPELINE")
    logger.info("=" * 80)
    
    # Load existing data
    data_path = "Data/improved_multilingual_hate_speech_v2.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded existing data: {len(df)} samples")
    
    # Initialize augmenter
    augmenter = AdvancedDataAugmenter(target_samples_per_language=5000)
    
    # Augment data
    augmented_df = augmenter.augment_existing_data(df)
    
    # Save augmented data
    augmenter.save_augmented_data(augmented_df)
    
    logger.info("=" * 80)
    logger.info("DATA AUGMENTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total samples: {len(augmented_df)}")
    logger.info(f"Hate speech ratio: {augmented_df['isHate'].mean():.2%}")
    logger.info(f"Sources: {augmented_df['source'].value_counts().to_dict()}")
    logger.info(f"Languages: {augmented_df['original_language'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
