import os
import sys
import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Tuple
import random
import bz2
from collections import Counter

# Ensure project root is on path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChineseDetectionImprover:
    """
    Comprehensive improvement system for Chinese hate speech detection.
    Focuses on expanding Chinese data and enhancing preprocessing.
    """
    
    def __init__(self):
        # Enhanced Chinese hate speech keywords and patterns
        self.chinese_hate_patterns = {
            # Gender-based hate
            'gender_hate': [
                '女人', '女性', '女的', '娘们', '婊子', '贱人', '荡妇', '妓女',
                '母狗', '死女人', '臭女人', '滚回厨房', '回家做饭', '女人就该',
                '女权', '女拳', '田园女权', '女权婊', '女权癌'
            ],
            # Sexual orientation hate
            'sexual_orientation_hate': [
                '同性恋', '基佬', '死gay', '玻璃', '同志', '拉拉', '女同',
                '男同', '跨性别', '变性人', '人妖', '不男不女', '变态',
                '性取向', '搞基', '搞同性恋'
            ],
            # Religious hate
            'religious_hate': [
                '穆斯林', '回民', '回回', '绿教', '清真', '伊斯兰', '伊斯兰教',
                '基督徒', '基督教', '耶稣', '佛教', '和尚', '道士',
                '异教徒', '邪教', '宗教极端'
            ],
            # Racial/ethnic hate
            'racial_hate': [
                '黑鬼', '黑奴', '黑人', '尼哥', '老黑',
                '白皮', '白鬼', '洋鬼子', '外国佬',
                '小日本', '日本鬼子', '棒子', '高丽棒子',
                '阿三', '印度人', '印巴', '东南亚猴子'
            ],
            # Disability hate
            'disability_hate': [
                '残废', '残疾人', '智障', '弱智', '白痴', '傻子', '脑残',
                '神经病', '精神病', '疯子', '有病', '不正常',
                '聋子', '瞎子', '瘸子', '哑巴'
            ],
            # National origin hate
            'national_origin_hate': [
                '移民', '外来人', '外地人', '外省人', '乡下人', '土包子',
                '滚回', '滚出去', '不欢迎', '外来入侵',
                '偷渡', '非法移民', '难民'
            ],
            # General hate/insults
            'general_hate': [
                '死', '去死', '该死', '死全家', '全家死光',
                '滚', '滚蛋', '滚开', '滚远点',
                '垃圾', '废物', '人渣', '败类', '杂种',
                '蠢货', '白痴', '笨蛋', '傻逼', '智障',
                '恶心', '讨厌', '憎恨', '恨', '厌恶',
                '消灭', '清除', '杀掉', '杀死', '弄死',
                '不应该存在', '不该活着', '没有价值'
            ],
            # Violence/threats
            'violence_hate': [
                '杀', '杀死', '弄死', '干掉', '解决',
                '打', '揍', '暴打', '痛打',
                '烧', '烧死', '烧掉',
                '炸', '炸死', '炸掉', '炸弹',
                '砍', '砍死', '切', '割',
                '枪', '开枪', '射杀'
            ]
        }
        
        # Enhanced neutral keywords
        self.chinese_neutral_keywords = [
            '好', '很好', '非常好', '优秀', '棒', '不错', '可以',
            '美丽', '漂亮', '好看', '精彩', '美妙', '完美',
            '喜欢', '爱', '热爱', '喜爱', '欣赏',
            '感谢', '谢谢', '感激', '感恩',
            '帮助', '支持', '协助', '援助',
            '朋友', '友谊', '友好', '友善',
            '家庭', '家人', '亲人', '亲情',
            '工作', '学习', '努力', '奋斗',
            '成功', '成就', '进步', '发展',
            '健康', '快乐', '幸福', '开心',
            '天气', '阳光', '晴朗', '美好',
            '食物', '美食', '美味', '好吃',
            '旅行', '旅游', '游玩', '放松',
            '音乐', '艺术', '文化', '教育'
        ]
        
        # Chinese hate speech templates for synthetic generation
        self.chinese_hate_templates = [
            "{target}都是{negative_adj}",
            "所有{group}都应该{action}",
            "我恨{target}",
            "{target}是{negative_adj}和{negative_adj2}",
            "我希望{target}{action}",
            "{group}不应该存在",
            "{target}应该待在{place}",
            "杀死所有{group}",
            "{target}是{negative_adj}的{insult}",
            "我鄙视{target}",
            "{group}都是{negative_adj}",
            "{target}应该{action}",
            "所有{group}都是{negative_adj}的",
            "{target}没有价值",
            "{group}应该被{action}",
            "我讨厌所有{group}",
            "{target}是{insult}",
            "{group}不配活着",
            "{target}应该消失",
            "我恨所有{group}"
        ]
        
        self.chinese_neutral_templates = [
            "今天是{positive_adj}的一天",
            "我爱我的{family_member}",
            "今天天气{weather_adj}",
            "谢谢你的{help_type}",
            "祝你{time_of_day}{positive_adj}",
            "我喜欢{activity}",
            "这个{object}很{positive_adj}",
            "我对{topic}感到{emotion}",
            "我们一起{action}吧",
            "希望你今天{positive_adj}",
            "这是一个{positive_adj}的{time_of_day}",
            "我很{emotion}能够{activity}",
            "感谢你的{help_type}",
            "今天是个{positive_adj}的日子"
        ]
    
    def detect_chinese_characters(self, text: str) -> bool:
        """Check if text contains Chinese characters"""
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(chinese_pattern.search(text))
    
    def clean_chinese_text(self, text: str) -> str:
        """Enhanced Chinese text cleaning"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace but preserve Chinese spacing
        text = re.sub(r'\s+', ' ', text)
        
        # Keep Chinese characters, numbers, and basic punctuation
        text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？、；：""''（）【】《》]', ' ', text)
        
        # Remove English-only words if text is primarily Chinese
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        if chinese_chars > 0 and chinese_chars / max(total_chars, 1) > 0.3:
            # Keep some common English words but remove others
            text = re.sub(r'\b[a-zA-Z]{1,2}\b', '', text)
        
        return text.strip()
    
    def calculate_hate_score(self, text: str) -> Tuple[bool, float]:
        """Calculate hate speech score for Chinese text"""
        text = self.clean_chinese_text(text)
        
        if not text or len(text) < 2:
            return False, 0.0
        
        # Count matches in each category
        category_scores = {}
        total_hate_matches = 0
        
        for category, keywords in self.chinese_hate_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > 0:
                category_scores[category] = matches
                total_hate_matches += matches
        
        # Count neutral matches
        neutral_matches = sum(1 for keyword in self.chinese_neutral_keywords if keyword in text)
        
        # Calculate confidence
        if total_hate_matches == 0:
            if neutral_matches > 0:
                return False, 0.1
            return False, 0.0
        
        # Strong hate indicators
        if total_hate_matches >= 3:
            return True, min(0.95, 0.7 + (total_hate_matches * 0.05))
        elif total_hate_matches >= 2:
            return True, 0.75
        elif total_hate_matches == 1:
            # Check if it's a strong hate word
            strong_hate_words = ['死', '杀', '滚', '垃圾', '废物', '人渣', '杂种', '婊子', '贱人']
            if any(word in text for word in strong_hate_words):
                return True, 0.65
            else:
                return True, 0.55
        
        return False, 0.0
    
    def generate_synthetic_chinese_hate(self, num_samples: int = 500) -> List[Dict]:
        """Generate synthetic Chinese hate speech samples"""
        logger.info(f"Generating {num_samples} synthetic Chinese hate speech samples...")
        
        targets = ['女人', '女性', '同性恋', '穆斯林', '移民', '黑人', '残疾人', '犹太人', '跨性别']
        groups = ['穆斯林', '同性恋', '女性', '移民', '黑人', '犹太人']
        negative_adjs = ['愚蠢', '无用', '可恨', '恶心', '讨厌', '垃圾', '废物', '低贱', '卑劣']
        negative_adjs2 = ['恶心', '可怕', '可恨', '讨厌', '卑劣', '无耻']
        actions = ['死', '消失', '滚开', '受苦', '被消灭', '被清除']
        places = ['厨房', '家里', '地狱', '监狱', '牢房', '垃圾堆']
        insults = ['垃圾', '废物', '人渣', '败类', '杂种', '蠢货']
        
        positive_adjs = ['好', '美丽', '精彩', '美妙', '优秀', '棒']
        family_members = ['家人', '父母', '朋友', '亲人']
        weather_adjs = ['好', '美丽', '晴朗', '宜人']
        help_types = ['帮助', '支持', '协助']
        time_of_days = ['早上', '下午', '晚上']
        activities = ['阅读', '学习', '工作', '运动']
        objects = ['书', '电影', '音乐', '食物']
        topics = ['这个', '那个', '今天', '明天']
        emotions = ['开心', '快乐', '兴奋', '满意']
        
        synthetic_data = []
        
        # Generate hate speech samples
        hate_count = 0
        for _ in range(num_samples * 2):  # Generate more to filter
            if hate_count >= num_samples:
                break
            
            template = random.choice(self.chinese_hate_templates)
            
            try:
                text = template.format(
                    target=random.choice(targets),
                    group=random.choice(groups),
                    negative_adj=random.choice(negative_adjs),
                    negative_adj2=random.choice(negative_adjs2),
                    action=random.choice(actions),
                    place=random.choice(places),
                    insult=random.choice(insults)
                )
                
                # Verify it's actually hate speech
                is_hate, confidence = self.calculate_hate_score(text)
                if is_hate and confidence > 0.5:
                    synthetic_data.append({
                        'text': text,
                        'isHate': 1,
                        'original_language': 'cmn',
                        'source': 'synthetic_hate',
                        'confidence': confidence
                    })
                    hate_count += 1
            except:
                continue
        
        logger.info(f"Generated {len(synthetic_data)} synthetic Chinese hate speech samples")
        return synthetic_data
    
    def generate_synthetic_chinese_neutral(self, num_samples: int = 500) -> List[Dict]:
        """Generate synthetic Chinese neutral samples"""
        logger.info(f"Generating {num_samples} synthetic Chinese neutral samples...")
        
        positive_adjs = ['好', '美丽', '精彩', '美妙', '优秀', '棒', '不错', '可以']
        family_members = ['家人', '父母', '朋友', '亲人', '孩子']
        weather_adjs = ['好', '美丽', '晴朗', '宜人', '舒适']
        help_types = ['帮助', '支持', '协助', '援助']
        time_of_days = ['早上', '下午', '晚上', '中午']
        activities = ['阅读', '学习', '工作', '运动', '旅行', '听音乐']
        objects = ['书', '电影', '音乐', '食物', '礼物', '花']
        topics = ['这个', '那个', '今天', '明天', '未来']
        emotions = ['开心', '快乐', '兴奋', '满意', '高兴', '愉快']
        
        synthetic_data = []
        
        for _ in range(num_samples):
            template = random.choice(self.chinese_neutral_templates)
            
            try:
                text = template.format(
                    positive_adj=random.choice(positive_adjs),
                    family_member=random.choice(family_members),
                    weather_adj=random.choice(weather_adjs),
                    help_type=random.choice(help_types),
                    time_of_day=random.choice(time_of_days),
                    activity=random.choice(activities),
                    object=random.choice(objects),
                    topic=random.choice(topics),
                    emotion=random.choice(emotions)
                )
                
                # Verify it's neutral
                is_hate, confidence = self.calculate_hate_score(text)
                if not is_hate:
                    synthetic_data.append({
                        'text': text,
                        'isHate': 0,
                        'original_language': 'cmn',
                        'source': 'synthetic_neutral',
                        'confidence': 1.0 - confidence
                    })
            except:
                continue
        
        logger.info(f"Generated {len(synthetic_data)} synthetic Chinese neutral samples")
        return synthetic_data
    
    def process_existing_chinese_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Re-process existing Chinese data with improved detection"""
        logger.info("Re-processing existing Chinese data with improved detection...")
        
        chinese_df = df[df['original_language'] == 'cmn'].copy()
        logger.info(f"Found {len(chinese_df)} existing Chinese samples")
        
        # Re-classify with improved detection
        improved_labels = []
        improved_confidences = []
        
        for idx, row in chinese_df.iterrows():
            text = str(row['text'])
            is_hate, confidence = self.calculate_hate_score(text)
            improved_labels.append(1 if is_hate else 0)
            improved_confidences.append(confidence)
        
        chinese_df['isHate'] = improved_labels
        chinese_df['confidence'] = improved_confidences
        
        # Statistics
        hate_count = chinese_df['isHate'].sum()
        logger.info(f"Re-classified Chinese data: {hate_count} hate ({hate_count/len(chinese_df)*100:.1f}%), "
                   f"{len(chinese_df)-hate_count} neutral")
        
        return chinese_df
    
    def extract_chinese_from_mixed_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract Chinese text from mixed-language dataset"""
        logger.info("Extracting Chinese text from mixed dataset...")
        
        chinese_samples = []
        
        for idx, row in df.iterrows():
            text = str(row.get('text', ''))
            if self.detect_chinese_characters(text):
                # Check if it's primarily Chinese
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
                total_chars = len(re.sub(r'\s', '', text))
                if total_chars > 0 and chinese_chars / total_chars > 0.3:
                    is_hate, confidence = self.calculate_hate_score(text)
                    chinese_samples.append({
                        'text': text,
                        'isHate': 1 if is_hate else 0,
                        'original_language': 'cmn',
                        'source': 'extracted_mixed',
                        'confidence': confidence
                    })
        
        if chinese_samples:
            extracted_df = pd.DataFrame(chinese_samples)
            logger.info(f"Extracted {len(extracted_df)} Chinese samples from mixed dataset")
            return extracted_df
        else:
            return pd.DataFrame()
    
    def create_improved_chinese_dataset(self, target_hate_samples: int = 2000, 
                                       target_neutral_samples: int = 2000) -> pd.DataFrame:
        """Create improved Chinese dataset with expanded data"""
        logger.info("=" * 80)
        logger.info("CREATING IMPROVED CHINESE HATE SPEECH DATASET")
        logger.info("=" * 80)
        
        all_chinese_data = []
        
        # Step 1: Load and re-process existing Chinese data
        logger.info("\nStep 1: Processing existing Chinese data...")
        existing_paths = [
            'Data/expanded_multilingual_hate_speech.csv',
            'Data/improved_multilingual_hate_speech_dataset.csv',
            'Data/improved_multilingual_hate_speech_v2.csv'
        ]
        
        existing_chinese = []
        for path in existing_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if 'original_language' in df.columns:
                        chinese_df = self.process_existing_chinese_data(df)
                        existing_chinese.append(chinese_df)
                except Exception as e:
                    logger.warning(f"Could not process {path}: {e}")
        
        if existing_chinese:
            combined_existing = pd.concat(existing_chinese, ignore_index=True)
            combined_existing = combined_existing.drop_duplicates(subset=['text'])
            all_chinese_data.append(combined_existing)
            logger.info(f"Processed {len(combined_existing)} existing Chinese samples")
        
        # Step 2: Load Chinese sentences from Tatoeba
        logger.info("\nStep 2: Loading Chinese sentences from Tatoeba...")
        tatoeba_path = 'Data/cmn_sentences_detailed.tsv.bz2'
        if os.path.exists(tatoeba_path):
            try:
                with bz2.open(tatoeba_path, 'rt', encoding='utf-8') as f:
                    lines = f.readlines()
                
                tatoeba_samples = []
                for line in lines[:10000]:  # Process up to 10k samples
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        text = parts[2].strip()
                        if text and text != '\\N' and len(text) > 5:
                            is_hate, confidence = self.calculate_hate_score(text)
                            tatoeba_samples.append({
                                'text': text,
                                'isHate': 1 if is_hate else 0,
                                'original_language': 'cmn',
                                'source': 'tatoeba',
                                'confidence': confidence
                            })
                
                if tatoeba_samples:
                    tatoeba_df = pd.DataFrame(tatoeba_samples)
                    all_chinese_data.append(tatoeba_df)
                    logger.info(f"Processed {len(tatoeba_df)} Tatoeba Chinese samples")
            except Exception as e:
                logger.warning(f"Could not process Tatoeba data: {e}")
        
        # Step 3: Generate synthetic hate speech
        logger.info("\nStep 3: Generating synthetic Chinese hate speech...")
        synthetic_hate = self.generate_synthetic_chinese_hate(target_hate_samples)
        if synthetic_hate:
            all_chinese_data.append(pd.DataFrame(synthetic_hate))
        
        # Step 4: Generate synthetic neutral samples
        logger.info("\nStep 4: Generating synthetic Chinese neutral samples...")
        synthetic_neutral = self.generate_synthetic_chinese_neutral(target_neutral_samples)
        if synthetic_neutral:
            all_chinese_data.append(pd.DataFrame(synthetic_neutral))
        
        # Step 5: Combine all data
        logger.info("\nStep 5: Combining all Chinese data...")
        if not all_chinese_data:
            logger.error("No Chinese data collected!")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_chinese_data, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['text'])
        
        # Balance dataset
        hate_df = combined_df[combined_df['isHate'] == 1]
        neutral_df = combined_df[combined_df['isHate'] == 0]
        
        # Limit to target sizes
        if len(hate_df) > target_hate_samples:
            hate_df = hate_df.nlargest(target_hate_samples, 'confidence')
        if len(neutral_df) > target_neutral_samples:
            neutral_df = neutral_df.sample(n=target_neutral_samples, random_state=42)
        
        final_df = pd.concat([hate_df, neutral_df], ignore_index=True)
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Statistics
        logger.info("\n" + "=" * 80)
        logger.info("IMPROVED CHINESE DATASET STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total samples: {len(final_df)}")
        logger.info(f"Hate speech: {final_df['isHate'].sum()} ({final_df['isHate'].mean()*100:.1f}%)")
        logger.info(f"Neutral: {(final_df['isHate'] == 0).sum()} ({(final_df['isHate'] == 0).mean()*100:.1f}%)")
        logger.info(f"\nSource distribution:")
        print(final_df['source'].value_counts())
        
        return final_df


def main():
    """Main function to create improved Chinese dataset"""
    improver = ChineseDetectionImprover()
    
    # Create improved dataset
    improved_df = improver.create_improved_chinese_dataset(
        target_hate_samples=2000,
        target_neutral_samples=2000
    )
    
    if len(improved_df) > 0:
        # Save improved dataset
        output_path = 'Data/improved_chinese_hate_speech.csv'
        improved_df.to_csv(output_path, index=False)
        logger.info(f"\nImproved Chinese dataset saved to: {output_path}")
        
        # Show sample data
        logger.info("\nSample hate speech:")
        print(improved_df[improved_df['isHate'] == 1][['text', 'isHate', 'confidence']].head(10))
        
        logger.info("\nSample neutral:")
        print(improved_df[improved_df['isHate'] == 0][['text', 'isHate', 'confidence']].head(10))
    else:
        logger.error("Failed to create improved Chinese dataset")


if __name__ == "__main__":
    main()

