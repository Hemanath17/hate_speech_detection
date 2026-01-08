"""
Training script for Hate Speech Detection using larger ETHOS dataset
Uses 998 samples instead of 433 for better performance
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from configs.config import MODEL_CONFIG, DATA_PATHS, CATEGORIES, PERFORMANCE_TARGETS
from src.data.ethos_loader import EthosDataLoader
from src.data.preprocessing import HateSpeechPreprocessor
from src.data.augmentation import HateSpeechAugmenter
from src.models.distilbert_model import MultiLabelDistilBERT, HateSpeechTrainer, create_data_loaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function using larger ETHOS dataset"""
    logger.info("=" * 50)
    logger.info("HATE SPEECH DETECTION - ETHOS LARGE DATASET")
    logger.info("=" * 50)
    
    # Create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"hate_speech_ethos_large_{timestamp}"
    
    # Initialize data loader
    ethos_loader = EthosDataLoader({
        'ethos_binary_path': DATA_PATHS['ethos_binary'],
        'ethos_multilabel_path': DATA_PATHS['ethos_multilabel_full']
    })
    
    # Load enhanced dataset
    logger.info("Loading enhanced ETHOS dataset...")
    df, categories = ethos_loader.create_enhanced_multilabel_dataset()
    
    # Get dataset statistics
    stats = ethos_loader.get_dataset_stats(df, categories)
    logger.info(f"Dataset Statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Hate samples: {stats['hate_samples']}")
    logger.info(f"  Non-hate samples: {stats['non_hate_samples']}")
    logger.info(f"  Average text length: {stats['avg_text_length']:.1f}")
    logger.info(f"  Max text length: {stats['max_text_length']}")
    
    for cat, count in stats['category_distribution'].items():
        logger.info(f"  {cat}: {count} samples")
    
    # Initialize preprocessor
    preprocessor = HateSpeechPreprocessor(MODEL_CONFIG)
    
    # Create multi-label data
    logger.info("Creating multi-label data...")
    texts, labels = preprocessor.create_multilabel_data(df)
    
    # Split data
    logger.info("Splitting data...")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = preprocessor.split_data(
        texts, labels, test_size=0.2, val_size=0.1
    )
    
    logger.info(f"Train: {len(train_texts)} samples")
    logger.info(f"Validation: {len(val_texts)} samples")
    logger.info(f"Test: {len(test_texts)} samples")
    
    # Data augmentation
    logger.info("Applying data augmentation...")
    augmenter = HateSpeechAugmenter(MODEL_CONFIG)
    augmented_texts, augmented_labels = augmenter.augment_dataset(
        train_texts, train_labels, target_multiplier=2.0
    )
    
    logger.info(f"Augmented data: {len(augmented_texts)} samples")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        augmented_texts, augmented_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        preprocessor.tokenizer, MODEL_CONFIG
    )
    
    # Initialize model and trainer
    logger.info("Initializing model and trainer...")
    
    # Add categories to model config
    model_config = MODEL_CONFIG.copy()
    model_config['categories'] = categories
    
    model = MultiLabelDistilBERT(model_config)
    trainer = HateSpeechTrainer(model, model_config)
    
    # Train model
    logger.info("Starting training...")
    test_metrics = trainer.train(train_loader, val_loader, test_loader)
    
    # Evaluate performance
    logger.info("Evaluating model performance...")
    targets_met = True
    
    # Check overall F1 score
    if test_metrics['f1_macro'] < PERFORMANCE_TARGETS['overall_f1']:
        logger.warning(f"❌ Overall F1 score target not met: {test_metrics['f1_macro']:.4f} < {PERFORMANCE_TARGETS['overall_f1']}")
        targets_met = False
    else:
        logger.info(f"✅ Overall F1 score target met: {test_metrics['f1_macro']:.4f}")
    
    # Check per-category F1 scores
    for i, (cat, f1) in enumerate(zip(categories, test_metrics['f1_per_class'])):
        if f1 < PERFORMANCE_TARGETS['min_category_f1']:
            logger.warning(f"❌ {cat} F1: {f1:.4f} < {PERFORMANCE_TARGETS['min_category_f1']}")
            targets_met = False
        else:
            logger.info(f"✅ {cat} F1: {f1:.4f}")
    
    if not targets_met:
        logger.warning("⚠️ Some performance targets not met. Consider:")
        logger.warning("  - More data augmentation")
        logger.warning("  - Different model architecture")
        logger.warning("  - Hyperparameter tuning")
        logger.warning("  - Ensemble methods")
    
    # Save model and results
    logger.info("Saving model and results...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / f"{experiment_name}_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save tokenizer
    tokenizer_path = models_dir / f"{experiment_name}_tokenizer"
    preprocessor.tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer saved to: {tokenizer_path}")
    
    # Save results
    results = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'f1_macro': test_metrics['f1_macro'],
        'f1_micro': test_metrics['f1_micro'],
        'f1_weighted': test_metrics['f1_weighted'],
        'f1_per_class': test_metrics['f1_per_class'],
        'categories': categories,
        'model_config': MODEL_CONFIG,
        'performance_targets': PERFORMANCE_TARGETS,
        'dataset_stats': stats,
        'targets_met': targets_met
    }
    
    results_path = models_dir / f"{experiment_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")
    
    # Final summary
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"F1 Macro: {test_metrics['f1_macro']:.4f}")
    logger.info(f"F1 Micro: {test_metrics['f1_micro']:.4f}")
    logger.info(f"F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    logger.info(f"Targets Met: {targets_met}")
    logger.info("=" * 50)

if __name__ == "__main__":
    import torch
    main()
