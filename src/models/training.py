"""
Training script for multi-label hate speech detection
Optimized for 0.90 F1 score
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import DistilBertTokenizer
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import HateSpeechPreprocessor
from src.data.augmentation import HateSpeechAugmenter
from src.models.distilbert_model import MultiLabelDistilBERT, HateSpeechTrainer, create_data_loaders
from configs.config import MODEL_CONFIG, CATEGORIES, PERFORMANCE_TARGETS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_experiment():
    """Setup experiment tracking"""
    experiment_name = f"hate_speech_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Starting experiment: {experiment_name}")
    return experiment_name

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    logger.info("Loading and preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = HateSpeechPreprocessor({'categories': CATEGORIES})
    
    # Load dataset
    data_path = Path(__file__).parent.parent.parent / "Data" / "Ethos_Dataset_Multi_Label.csv"
    df = preprocessor.load_ethos_dataset(str(data_path))
    
    # Create multi-label data
    texts, labels = preprocessor.create_multilabel_data(df)
    
    # Split data
    data_splits = preprocessor.split_data(texts, labels, test_size=0.2, val_size=0.1)
    
    logger.info(f"Data loaded: {len(texts)} samples")
    logger.info(f"Train: {len(data_splits['X_train'])}, Val: {len(data_splits['X_val'])}, Test: {len(data_splits['X_test'])}")
    
    return data_splits

def augment_data(train_texts, train_labels):
    """Augment training data to handle class imbalance"""
    logger.info("Augmenting training data...")
    
    # Initialize augmenter
    augmenter = HateSpeechAugmenter({'categories': CATEGORIES})
    
    # Augment data
    augmented_texts, augmented_labels = augmenter.augment_dataset(
        train_texts.tolist(), 
        [list(labels) for labels in train_labels],
        target_multiplier=2.0  # Double the dataset size
    )
    
    logger.info(f"Augmented data: {len(augmented_texts)} samples")
    
    return np.array(augmented_texts), np.array(augmented_labels)

def create_model_and_tokenizer():
    """Create model and tokenizer"""
    logger.info("Creating model and tokenizer...")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CONFIG['distilbert_model'])
    
    # Create model
    model = MultiLabelDistilBERT({'categories': CATEGORIES, **MODEL_CONFIG})
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model, tokenizer

def train_model(model, tokenizer, data_splits):
    """Train the model"""
    logger.info("Starting model training...")
    
    # Augment training data
    train_texts, train_labels = augment_data(data_splits['X_train'], data_splits['y_train'])
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_texts, train_labels,
        data_splits['X_val'], data_splits['y_val'],
        data_splits['X_test'], data_splits['y_test'],
        tokenizer, MODEL_CONFIG
    )
    
    # Create trainer
    trainer = HateSpeechTrainer(model, MODEL_CONFIG)
    
    # Train model
    test_metrics = trainer.train(train_loader, val_loader, test_loader)
    
    return test_metrics

def evaluate_model_performance(metrics):
    """Evaluate if model meets performance targets"""
    logger.info("Evaluating model performance...")
    
    f1_macro = metrics['f1_macro']
    f1_micro = metrics['f1_micro']
    f1_per_class = metrics['f1_per_class']
    
    # Check overall F1 score
    target_f1 = PERFORMANCE_TARGETS['overall_f1']
    if f1_macro >= target_f1:
        logger.info(f"✅ Overall F1 score target met: {f1_macro:.4f} >= {target_f1}")
    else:
        logger.warning(f"❌ Overall F1 score target not met: {f1_macro:.4f} < {target_f1}")
    
    # Check per-class F1 scores
    min_target = PERFORMANCE_TARGETS['min_category_f1']
    for i, f1 in enumerate(f1_per_class):
        category = CATEGORIES[i]
        if f1 >= min_target:
            logger.info(f"✅ {category} F1: {f1:.4f} >= {min_target}")
        else:
            logger.warning(f"❌ {category} F1: {f1:.4f} < {min_target}")
    
    # Overall assessment
    all_targets_met = f1_macro >= target_f1 and all(f1 >= min_target for f1 in f1_per_class)
    
    if all_targets_met:
        logger.info("🎉 All performance targets met!")
    else:
        logger.warning("⚠️ Some performance targets not met. Consider:")
        logger.warning("  - More data augmentation")
        logger.warning("  - Different model architecture")
        logger.warning("  - Hyperparameter tuning")
        logger.warning("  - Ensemble methods")
    
    return all_targets_met

def save_model_and_results(model, tokenizer, metrics, experiment_name):
    """Save model and results"""
    logger.info("Saving model and results...")
    
    # Create models directory
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / f"{experiment_name}_model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer
    tokenizer_path = models_dir / f"{experiment_name}_tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    
    # Save results
    results_path = models_dir / f"{experiment_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'f1_macro': float(metrics['f1_macro']),
            'f1_micro': float(metrics['f1_micro']),
            'f1_weighted': float(metrics['f1_weighted']),
            'f1_per_class': [float(f1) for f1 in metrics['f1_per_class']],
            'categories': CATEGORIES,
            'model_config': MODEL_CONFIG,
            'performance_targets': PERFORMANCE_TARGETS
        }, f, indent=2)
    
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Tokenizer saved to: {tokenizer_path}")
    logger.info(f"Results saved to: {results_path}")

def main():
    """Main training pipeline"""
    try:
        # Setup experiment
        experiment_name = setup_experiment()
        
        # Load and preprocess data
        data_splits = load_and_preprocess_data()
        
        # Create model and tokenizer
        model, tokenizer = create_model_and_tokenizer()
        
        # Train model
        metrics = train_model(model, tokenizer, data_splits)
        
        # Evaluate performance
        targets_met = evaluate_model_performance(metrics)
        
        # Save model and results
        save_model_and_results(model, tokenizer, metrics, experiment_name)
        
        # Final summary
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"F1 Micro: {metrics['f1_micro']:.4f}")
        logger.info(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        logger.info(f"Targets Met: {targets_met}")
        logger.info("=" * 50)
        
        return metrics, targets_met
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
