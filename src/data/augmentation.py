"""
Data augmentation for hate speech detection
"""

import random
import re
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from nlpaug.augmenter.word import SynonymAug, RandomWordAug
# from nlpaug.augmenter.sentence import BackTranslationAug  # Not available in this version
import logging

logger = logging.getLogger(__name__)

class HateSpeechAugmenter:
    """Data augmentation for hate speech detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.synonym_aug = SynonymAug(aug_src='wordnet', aug_max=2)
        self.random_word_aug = RandomWordAug(action="insert", aug_max=2)
        # self.back_translation_aug = BackTranslationAug(
        #     from_model_name='facebook/mbart-large-50-many-to-many-mmt',
        #     to_model_name='facebook/mbart-large-50-many-to-many-mmt'
        # )
        
        # Hate speech specific patterns for augmentation
        self.hate_patterns = {
            'gender': [
                (r'\bwomen\b', 'females'),
                (r'\bmen\b', 'males'),
                (r'\bgirls\b', 'girls'),
                (r'\bboys\b', 'boys')
            ],
            'race': [
                (r'\bblack\b', 'african american'),
                (r'\bwhite\b', 'caucasian'),
                (r'\bhispanic\b', 'latino')
            ],
            'religion': [
                (r'\bchristian\b', 'christian'),
                (r'\bmuslim\b', 'islamic'),
                (r'\bjew\b', 'jewish')
            ]
        }
    
    def simple_augmentation(self, text: str) -> str:
        """Fallback augmentation that does not rely on NLTK or external models.
        Applies lightweight, semantics-preserving tweaks: word swap, punctuation, or duplication.
        """
        try:
            tokens = text.split()
            if len(tokens) >= 4 and random.random() < 0.4:
                # swap two random interior tokens
                i, j = sorted(random.sample(range(1, len(tokens) - 1), 2))
                tokens[i], tokens[j] = tokens[j], tokens[i]
                return " ".join(tokens)
            if random.random() < 0.3:
                # add a punctuation at end
                return text + random.choice([".", "!", "?"])
            if len(tokens) >= 2 and random.random() < 0.3:
                # duplicate a non-stopword-ish token (heuristic)
                k = random.randrange(0, len(tokens))
                tokens.insert(k, tokens[k])
                return " ".join(tokens)
        except Exception:
            pass
        # as a last resort, return text unchanged
        return text

    def synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms"""
        try:
            return self.synonym_aug.augment(text)
        except Exception as e:
            logger.warning(f"Synonym replacement failed: {e}")
            return text
    
    def random_insertion(self, text: str) -> str:
        """Insert random words"""
        try:
            return self.random_word_aug.augment(text)
        except Exception as e:
            logger.warning(f"Random insertion failed: {e}")
            return text
    
    def back_translation(self, text: str) -> str:
        """Back translation augmentation"""
        try:
            # Translate to another language and back
            languages = ['es_XX', 'fr_XX', 'de_DE']
            target_lang = random.choice(languages)
            
            # This is a simplified version - in practice, you'd use proper translation
            return text  # Placeholder for now
        except Exception as e:
            logger.warning(f"Back translation failed: {e}")
            return text
    
    def pattern_based_augmentation(self, text: str, category: str) -> str:
        """Apply category-specific pattern augmentation"""
        if category not in self.hate_patterns:
            return text
            
        patterns = self.hate_patterns[category]
        for pattern, replacement in patterns:
            if random.random() < 0.3:  # 30% chance to apply
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def contextual_augmentation(self, text: str, labels: List[int]) -> str:
        """Apply contextual augmentation based on labels"""
        augmented_text = text
        
        # Apply category-specific augmentations
        category_names = ['violence', 'directed_vs_generalized', 'gender', 'race', 
                         'national_origin', 'disability', 'religion', 'sexual_orientation']
        
        for i, label in enumerate(labels):
            if label == 1:  # If this category is present
                category = category_names[i]
                augmented_text = self.pattern_based_augmentation(augmented_text, category)
        
        return augmented_text
    
    def augment_sample(self, text: str, labels: List[int]) -> List[Tuple[str, List[int]]]:
        """Generate augmented samples for a given text-label pair"""
        augmented_samples = [(text, labels)]  # Include original
        
        # Apply different augmentation techniques
        augmentation_methods = [
            ('synonym', self.synonym_replacement),
            ('insertion', self.random_insertion),
            ('contextual', lambda t: self.contextual_augmentation(t, labels)),
            ('simple', self.simple_augmentation)
        ]
        
        for method_name, method in augmentation_methods:
            try:
                if random.random() < 0.7:  # 70% chance to apply each method
                    augmented_text = method(text)
                    if augmented_text != text and len(augmented_text.strip()) > 0:
                        augmented_samples.append((augmented_text, labels))
            except Exception as e:
                logger.warning(f"Augmentation method {method_name} failed: {e}")
                continue
        
        return augmented_samples
    
    def augment_dataset(self, texts: List[str], labels: List[List[int]], 
                       target_multiplier: float = 2.0) -> Tuple[List[str], List[List[int]]]:
        """Augment entire dataset"""
        logger.info(f"Starting augmentation with target multiplier: {target_multiplier}")
        
        augmented_texts = []
        augmented_labels = []
        
        # Calculate how many samples to generate
        target_samples = int(len(texts) * target_multiplier)
        current_samples = len(texts)
        
        # Add original samples
        augmented_texts.extend(texts)
        augmented_labels.extend(labels)
        
        # Generate augmented samples
        samples_needed = target_samples - current_samples
        samples_generated = 0
        
        # Create progress bar for augmentation
        from tqdm import tqdm
        progress_bar = tqdm(total=samples_needed, 
                           desc="Augmenting Data", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        max_global_attempts = max(1000, samples_needed * 10)
        attempts = 0
        while samples_generated < samples_needed and attempts < max_global_attempts:
            # Randomly select a sample to augment
            idx = random.randint(0, len(texts) - 1)
            text = texts[idx]
            label = labels[idx]
            
            # Generate augmented samples
            new_samples = self.augment_sample(text, label)
            
            # Add new samples (excluding original)
            for aug_text, aug_label in new_samples[1:]:  # Skip original
                if samples_generated >= samples_needed:
                    break
                    
                augmented_texts.append(aug_text)
                augmented_labels.append(aug_label)
                samples_generated += 1
                progress_bar.update(1)
            
            # If none were added this round, force-add a simple augmentation to avoid stalling
            if len(new_samples) <= 1 and samples_generated < samples_needed:
                forced = self.simple_augmentation(text)
                if forced and forced.strip() and forced != text:
                    augmented_texts.append(forced)
                    augmented_labels.append(label)
                    samples_generated += 1
                    progress_bar.update(1)
            
            attempts += 1
        
        if attempts >= max_global_attempts and samples_generated < samples_needed:
            logger.warning(
                f"Augmentation stopped after {attempts} attempts with {samples_generated}/{samples_needed} samples."
            )
        
        progress_bar.close()
        logger.info(f"Augmentation complete. Generated {samples_generated} new samples")
        logger.info(f"Total samples: {len(augmented_texts)}")
        
        return augmented_texts, augmented_labels
    
    def balance_dataset(self, texts: List[str], labels: List[List[int]]) -> Tuple[List[str], List[List[int]]]:
        """Balance dataset by oversampling minority classes"""
        logger.info("Balancing dataset...")
        
        # Count samples per class
        class_counts = {}
        for i in range(len(self.config['categories'])):
            class_counts[i] = sum(1 for label_list in labels if i in label_list)
        
        # Find the maximum count
        max_count = max(class_counts.values())
        
        # Oversample minority classes
        balanced_texts = []
        balanced_labels = []
        
        for i, (text, label_list) in enumerate(zip(texts, labels)):
            # Add original sample
            balanced_texts.append(text)
            balanced_labels.append(label_list)
            
            # For each positive class in this sample
            for class_idx in label_list:
                current_count = class_counts[class_idx]
                if current_count < max_count:
                    # Generate augmented sample
                    augmented_samples = self.augment_sample(text, label_list)
                    for aug_text, aug_label in augmented_samples[1:]:  # Skip original
                        balanced_texts.append(aug_text)
                        balanced_labels.append(aug_label)
                        class_counts[class_idx] += 1
                        
                        if class_counts[class_idx] >= max_count:
                            break
        
        logger.info(f"Balanced dataset size: {len(balanced_texts)}")
        return balanced_texts, balanced_labels

def main():
    """Test augmentation pipeline"""
    config = {'categories': ['violence', 'gender', 'race', 'religion']}
    augmenter = HateSpeechAugmenter(config)
    
    # Test sample
    text = "Women are terrible at sports"
    labels = [0, 1, 0, 0]  # gender = 1
    
    augmented = augmenter.augment_sample(text, labels)
    print(f"Original: {text}")
    for i, (aug_text, aug_labels) in enumerate(augmented):
        print(f"Augmented {i}: {aug_text}")

if __name__ == "__main__":
    main()
