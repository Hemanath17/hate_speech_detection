import pandas as pd
import numpy as np
import bz2
import logging
from typing import Dict, List, Tuple
import re
import random
from pathlib import Path

logger = logging.getLogger(__name__)

class ImprovedMultilingualDatasetCreator:
    """
    Improved multilingual hate speech dataset creator with better
    Tamil and Chinese language support
    """
    
    def __init__(self, data_path: str = "Data/"):
        self.data_path = Path(data_path)
        self.languages = {
            'tam': 'Tamil',
            'hin': 'Hindi', 
            'spa': 'Spanish',
            'cmn': 'Chinese',
            'eng': 'English'
        }
        
        # Enhanced hate speech patterns for each language
        self.hate_patterns = {
            'tam': {
                'gender': ['பெண்', 'ஆண்', 'முட்டாள்', 'பைத்தியம்', 'வெட்கமில்லாத', 'துஷ்ட', 'கெட்ட'],
                'religion': ['முஸ்லிம்', 'இந்து', 'கிறிஸ்தவன்', 'புத்தர்', 'சைவன்'],
                'violence': ['கொல்ல', 'சாக', 'தாக்கு', 'அடி', 'வெட்டு', 'கொலை'],
                'disability': ['முட்டாள்', 'பைத்தியம்', 'குருடன்', 'செவிடன்', 'ஊமை'],
                'race': ['கருப்பு', 'வெள்ளை', 'இந்தியன்', 'தமிழன்', 'தெலுங்கன்'],
                'caste': ['பார்ப்பனன்', 'தலித்', 'ஓபிசி', 'எஸ்சி', 'எஸ்டி'],
                'sexual': ['காமம்', 'வெட்கமில்லாத', 'துஷ்ட', 'கெட்ட']
            },
            'hin': {
                'gender': ['औरत', 'आदमी', 'बेवकूफ', 'पागल', 'बेशर्म', 'दुष्ट', 'बुरा'],
                'religion': ['मुसलमान', 'हिंदू', 'ईसाई', 'बौद्ध', 'सिख'],
                'violence': ['मार', 'मर', 'हत्या', 'पीट', 'काट', 'मारो'],
                'disability': ['बेवकूफ', 'पागल', 'अंधा', 'बहरा', 'गूंगा'],
                'race': ['काला', 'गोरा', 'भारतीय', 'तमिल', 'तेलुगु'],
                'caste': ['ब्राह्मण', 'दलित', 'ओबीसी', 'एससी', 'एसटी'],
                'sexual': ['काम', 'बेशर्म', 'दुष्ट', 'बुरा']
            },
            'spa': {
                'gender': ['mujer', 'hombre', 'estúpida', 'loca', 'desvergonzada', 'mala', 'perra'],
                'religion': ['musulmán', 'hindú', 'cristiano', 'budista', 'sikh'],
                'violence': ['matar', 'morir', 'atacar', 'golpear', 'cortar', 'asesinar'],
                'disability': ['estúpida', 'loca', 'ciega', 'sorda', 'muda'],
                'race': ['negro', 'blanco', 'indio', 'tamil', 'telugu'],
                'caste': ['brahmán', 'dalit', 'obc', 'sc', 'st'],
                'sexual': ['sexo', 'desvergonzada', 'mala', 'puta']
            },
            'cmn': {
                'gender': ['女人', '男人', '愚蠢', '疯狂', '无耻', '坏', '婊子'],
                'religion': ['穆斯林', '印度教', '基督教', '佛教', '锡克教'],
                'violence': ['杀', '死', '攻击', '打', '切', '谋杀'],
                'disability': ['愚蠢', '疯狂', '盲人', '聋子', '哑巴'],
                'race': ['黑人', '白人', '印度人', '泰米尔人', '泰卢固人'],
                'caste': ['婆罗门', '达利特', 'OBC', 'SC', 'ST'],
                'sexual': ['性', '无耻', '坏', '妓女']
            },
            'eng': {
                'gender': ['woman', 'man', 'stupid', 'crazy', 'shameless', 'bad', 'bitch'],
                'religion': ['muslim', 'hindu', 'christian', 'buddhist', 'sikh'],
                'violence': ['kill', 'die', 'attack', 'hit', 'cut', 'murder'],
                'disability': ['stupid', 'crazy', 'blind', 'deaf', 'dumb'],
                'race': ['black', 'white', 'indian', 'tamil', 'telugu'],
                'caste': ['brahmin', 'dalit', 'obc', 'sc', 'st'],
                'sexual': ['sex', 'shameless', 'bad', 'whore']
            }
        }
        
        # Additional hate speech templates for better coverage
        self.hate_templates = {
            'tam': [
                "{} முட்டாள் {}",
                "{} பைத்தியம் {}",
                "{} கெட்ட {}",
                "{} துஷ்ட {}",
                "{} வெட்கமில்லாத {}",
                "{} கொல்ல {}",
                "{} சாக {}",
                "{} அடி {}"
            ],
            'cmn': [
                "{} 愚蠢 {}",
                "{} 疯狂 {}",
                "{} 坏 {}",
                "{} 无耻 {}",
                "{} 杀 {}",
                "{} 死 {}",
                "{} 打 {}",
                "{} 切 {}"
            ],
            'hin': [
                "{} बेवकूफ {}",
                "{} पागल {}",
                "{} बुरा {}",
                "{} दुष्ट {}",
                "{} बेशर्म {}",
                "{} मार {}",
                "{} मर {}",
                "{} पीट {}"
            ],
            'spa': [
                "{} estúpida {}",
                "{} loca {}",
                "{} mala {}",
                "{} desvergonzada {}",
                "{} matar {}",
                "{} morir {}",
                "{} golpear {}",
                "{} cortar {}"
            ],
            'eng': [
                "{} stupid {}",
                "{} crazy {}",
                "{} bad {}",
                "{} shameless {}",
                "{} kill {}",
                "{} die {}",
                "{} hit {}",
                "{} cut {}"
            ]
        }
    
    def load_language_dataset(self, language: str, sample_size: int = 20000) -> pd.DataFrame:
        """Load and sample dataset for a specific language with more samples"""
        file_path = self.data_path / f"{language}_sentences_detailed.tsv.bz2"
        
        if not file_path.exists():
            logger.error(f"Dataset not found: {file_path}")
            return pd.DataFrame()
        
        logger.info(f"Loading {self.languages[language]} dataset...")
        
        with bz2.open(file_path, 'rt', encoding='utf-8') as f:
            # Read in chunks to handle large files
            chunk_size = 1000
            chunks = []
            
            for chunk in pd.read_csv(f, sep='\t', chunksize=chunk_size):
                chunks.append(chunk)
                if len(chunks) * chunk_size >= sample_size:
                    break
            
            df = pd.concat(chunks, ignore_index=True)
            df = df.head(sample_size)
        
        # Rename columns for consistency
        df.columns = ['id', 'language', 'text', 'user', 'created_at', 'updated_at']
        
        logger.info(f"Loaded {len(df)} samples from {self.languages[language]} dataset")
        return df
    
    def generate_synthetic_hate_speech(self, language: str, num_samples: int = 100) -> List[Dict]:
        """Generate synthetic hate speech samples for better training data"""
        synthetic_samples = []
        
        if language not in self.hate_patterns:
            return synthetic_samples
        
        patterns = self.hate_patterns[language]
        templates = self.hate_templates.get(language, [])
        
        # Generate samples using patterns
        for category, words in patterns.items():
            for word in words:
                for _ in range(3):  # Generate 3 samples per word
                    # Create simple hate speech sentence
                    hate_sentence = f"நீ {word} ஆக இருக்கிறாய்" if language == 'tam' else \
                                  f"你是一个{word}" if language == 'cmn' else \
                                  f"तुम {word} हो" if language == 'hin' else \
                                  f"Eres {word}" if language == 'spa' else \
                                  f"You are {word}"
                    
                    # Generate labels
                    labels = self.generate_hate_speech_labels(hate_sentence, language)
                    synthetic_samples.append({
                        'text': hate_sentence,
                        'language': language,
                        'original_language': language,
                        **labels
                    })
        
        # Generate samples using templates
        for template in templates:
            for _ in range(5):  # Generate 5 samples per template
                # Fill template with random words
                word1 = random.choice(list(patterns.values())[0])  # Use first category words
                word2 = random.choice(list(patterns.values())[1])  # Use second category words
                
                hate_sentence = template.format(word1, word2)
                labels = self.generate_hate_speech_labels(hate_sentence, language)
                synthetic_samples.append({
                    'text': hate_sentence,
                    'language': language,
                    'original_language': language,
                    **labels
                })
        
        return synthetic_samples[:num_samples]
    
    def generate_hate_speech_labels(self, text: str, language: str) -> Dict[str, int]:
        """Generate hate speech labels based on pattern matching"""
        text_lower = text.lower()
        labels = {
            'violence': 0,
            'directed_vs_generalized': 0,
            'gender': 0,
            'race': 0,
            'national_origin': 0,
            'disability': 0,
            'religion': 0,
            'sexual_orientation': 0
        }
        
        if language not in self.hate_patterns:
            return labels
        
        patterns = self.hate_patterns[language]
        
        # Check for hate speech patterns
        for category, words in patterns.items():
            for word in words:
                if word.lower() in text_lower:
                    # Map to our label categories
                    if category in ['gender', 'sexual']:
                        labels['gender'] = 1
                    elif category == 'religion':
                        labels['religion'] = 1
                    elif category == 'violence':
                        labels['violence'] = 1
                    elif category in ['disability']:
                        labels['disability'] = 1
                    elif category in ['race', 'caste']:
                        labels['race'] = 1
                    break
        
        # Additional pattern matching
        violence_patterns = ['kill', 'die', 'attack', 'hurt', 'destroy', 'கொல்ல', 'சாக', 'தாக்கு', '杀', '死', '攻击']
        if any(pattern in text_lower for pattern in violence_patterns):
            labels['violence'] = 1
        
        # Determine if it's hate speech overall
        is_hate = sum(labels.values()) > 0
        labels['isHate'] = 1 if is_hate else 0
        
        return labels
    
    def create_improved_multilingual_dataset(self, target_hate_ratio: float = 0.4) -> pd.DataFrame:
        """Create improved multilingual hate speech dataset with better Tamil and Chinese support"""
        logger.info("Creating improved multilingual hate speech dataset...")
        
        all_data = []
        
        for lang_code, lang_name in self.languages.items():
            logger.info(f"Processing {lang_name}...")
            
            # Load language dataset with more samples
            df = self.load_language_dataset(lang_code, 5000)  # Increased sample size
            
            if df.empty:
                continue
            
            # Generate synthetic hate speech samples
            synthetic_hate = self.generate_synthetic_hate_speech(lang_code, 200)
            synthetic_df = pd.DataFrame(synthetic_hate)
            
            # Generate labels for original data
            hate_labels = []
            for text in df['text']:
                labels = self.generate_hate_speech_labels(str(text), lang_code)
                hate_labels.append(labels)
            
            # Convert to DataFrame
            labels_df = pd.DataFrame(hate_labels)
            
            # Combine with original data
            combined_df = pd.concat([df[['text', 'language']], labels_df], axis=1)
            combined_df['original_language'] = lang_code
            
            # Add synthetic hate speech
            if not synthetic_df.empty:
                all_data.append(synthetic_df)
            
            all_data.append(combined_df)
        
        # Combine all languages
        multilingual_df = pd.concat(all_data, ignore_index=True)
        
        # Balance the dataset
        hate_samples = multilingual_df[multilingual_df['isHate'] == 1]
        non_hate_samples = multilingual_df[multilingual_df['isHate'] == 0]
        
        # Calculate target counts
        target_hate_count = len(hate_samples)
        target_non_hate_count = int(target_hate_count * (1 - target_hate_ratio) / target_hate_ratio)
        
        # Sample non-hate samples
        if len(non_hate_samples) > target_non_hate_count:
            non_hate_samples = non_hate_samples.sample(n=target_non_hate_count, random_state=42)
        
        # Combine balanced dataset
        balanced_df = pd.concat([hate_samples, non_hate_samples], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Created improved multilingual dataset with {len(balanced_df)} samples")
        
        # Log statistics
        for lang in self.languages.keys():
            lang_data = balanced_df[balanced_df['original_language'] == lang]
            if len(lang_data) > 0:
                hate_count = lang_data['isHate'].sum()
                logger.info(f"{self.languages[lang]}: {len(lang_data)} samples, {hate_count} hate speech ({hate_count/len(lang_data):.1%})")
        
        return balanced_df

def main():
    """Test the improved multilingual dataset creator"""
    logging.basicConfig(level=logging.INFO)
    
    creator = ImprovedMultilingualDatasetCreator()
    
    # Create improved multilingual dataset
    dataset = creator.create_improved_multilingual_dataset(target_hate_ratio=0.4)
    
    # Save dataset
    output_path = "Data/improved_multilingual_hate_speech_dataset.csv"
    dataset.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to: {output_path}")
    
    # Show sample
    print("\nSample data:")
    print(dataset[['text', 'original_language', 'isHate', 'violence', 'gender', 'religion']].head(10))
    
    # Show statistics
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Hate speech ratio: {dataset['isHate'].mean():.1%}")
    print(f"Language distribution:")
    print(dataset['original_language'].value_counts())

if __name__ == "__main__":
    main()
