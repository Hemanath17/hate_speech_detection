"""
Binary Classification Training for Hate Speech Detection using ETHOS dataset
Uses 998 samples for better performance
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

from configs.config import MODEL_CONFIG, DATA_PATHS
from src.data.preprocessing import HateSpeechPreprocessor
from src.data.augmentation import HateSpeechAugmenter
from src.models.distilbert_model import MultiLabelDistilBERT, HateSpeechTrainer, create_data_loaders
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BinaryHateSpeechDataset(Dataset):
    """Binary hate speech dataset"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BinaryDistilBERT(torch.nn.Module):
    """Binary classification DistilBERT model"""
    
    def __init__(self, config):
        super(BinaryDistilBERT, self).__init__()
        
        self.config = config
        
        # Load pre-trained DistilBERT
        from transformers import DistilBertModel, DistilBertConfig
        self.distilbert = DistilBertModel.from_pretrained(
            config['distilbert_model'],
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Binary classification head
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.distilbert.config.dim, 2)  # Binary: 0 or 1
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class BinaryHateSpeechTrainer:
    """Trainer for binary hate speech detection"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.7
        )
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Best model tracking
        self.best_f1 = 0.0
        self.patience_counter = 0
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        from tqdm import tqdm
        progress_bar = tqdm(dataloader, desc="Training", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar with current loss
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        from tqdm import tqdm
        progress_bar = tqdm(dataloader, desc="Evaluating", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar with current loss
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self, train_dataloader, val_dataloader, test_dataloader=None):
        """Train the model"""
        logger.info("Starting training...")
        
        from tqdm import tqdm
        epoch_progress = tqdm(range(self.config['num_epochs']), 
                             desc="Training Progress", 
                             bar_format='{l_bar}{bar}| Epoch {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for epoch in epoch_progress:
            # Update epoch progress bar
            epoch_progress.set_description(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader)
            
            # Evaluate
            val_metrics = self.evaluate(val_dataloader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Update epoch progress with metrics
            epoch_progress.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val F1': f'{val_metrics["f1"]:.4f}',
                'Best F1': f'{self.best_f1:.4f}'
            })
            
            # Log metrics
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            logger.info(f"Epoch {epoch + 1} - Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Epoch {epoch + 1} - Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Epoch {epoch + 1} - Val Precision: {val_metrics['precision']:.4f}")
            logger.info(f"Epoch {epoch + 1} - Val Recall: {val_metrics['recall']:.4f}")
            logger.info(f"Epoch {epoch + 1} - Val F1: {val_metrics['f1']:.4f}")
            
            # Check for improvement
            current_f1 = val_metrics['f1']
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), 'best_binary_model.pth')
                logger.info(f"🎉 New best model saved with F1: {current_f1:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"⏳ No improvement. Patience: {self.patience_counter}/{self.config['patience']}")
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_binary_model.pth'))
        logger.info(f"Training completed. Best F1: {self.best_f1:.4f}")
        
        # Final evaluation on test set
        if test_dataloader:
            test_metrics = self.evaluate(test_dataloader)
            logger.info("Final Test Results:")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
            logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
            logger.info(f"Test F1: {test_metrics['f1']:.4f}")
            
            return test_metrics
        
        return val_metrics

def create_binary_data_loaders(train_texts, train_labels, val_texts, val_labels, 
                              test_texts, test_labels, tokenizer, config):
    """Create data loaders for binary classification"""
    
    # Create datasets
    train_dataset = BinaryHateSpeechDataset(train_texts, train_labels, tokenizer, config['max_length'])
    val_dataset = BinaryHateSpeechDataset(val_texts, val_labels, tokenizer, config['max_length'])
    test_dataset = BinaryHateSpeechDataset(test_texts, test_labels, tokenizer, config['max_length'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader

def main():
    """Main training function for binary classification"""
    logger.info("=" * 50)
    logger.info("HATE SPEECH DETECTION - BINARY CLASSIFICATION")
    logger.info("=" * 50)
    
    # Create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"hate_speech_binary_{timestamp}"
    
    # Load binary ETHOS dataset
    logger.info("Loading binary ETHOS dataset...")
    binary_path = DATA_PATHS['ethos_binary']
    df = pd.read_csv(binary_path, sep=';')
    
    logger.info(f"Loaded {len(df)} samples from binary dataset")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Check data distribution
    hate_count = df['isHate'].sum()
    non_hate_count = len(df) - hate_count
    logger.info(f"Hate speech samples: {hate_count}")
    logger.info(f"Non-hate speech samples: {non_hate_count}")
    
    # Prepare data
    texts = df['comment'].values
    labels = df['isHate'].values.astype(int)
    
    # Split data
    logger.info("Splitting data...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )
    
    logger.info(f"Train: {len(train_texts)} samples")
    logger.info(f"Validation: {len(val_texts)} samples")
    logger.info(f"Test: {len(test_texts)} samples")
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CONFIG['distilbert_model'])
    
    # Data augmentation
    logger.info("Applying data augmentation...")
    augmenter = HateSpeechAugmenter(MODEL_CONFIG)
    augmented_texts, augmented_labels = augmenter.augment_dataset(
        train_texts.tolist(), train_labels.tolist(), target_multiplier=2.0
    )
    
    logger.info(f"Augmented data: {len(augmented_texts)} samples")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_binary_data_loaders(
        augmented_texts, augmented_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        tokenizer, MODEL_CONFIG
    )
    
    # Initialize model and trainer
    logger.info("Initializing model and trainer...")
    model = BinaryDistilBERT(MODEL_CONFIG)
    trainer = BinaryHateSpeechTrainer(model, MODEL_CONFIG)
    
    # Train model
    logger.info("Starting training...")
    test_metrics = trainer.train(train_loader, val_loader, test_loader)
    
    # Evaluate performance
    logger.info("Evaluating model performance...")
    targets_met = True
    
    # Check F1 score (target: 0.90)
    if test_metrics['f1'] < 0.90:
        logger.warning(f"❌ F1 score target not met: {test_metrics['f1']:.4f} < 0.90")
        targets_met = False
    else:
        logger.info(f"✅ F1 score target met: {test_metrics['f1']:.4f}")
    
    if not targets_met:
        logger.warning("⚠️ Performance targets not met. Consider:")
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
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer saved to: {tokenizer_path}")
    
    # Save results
    results = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'accuracy': test_metrics['accuracy'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1': test_metrics['f1'],
        'model_config': MODEL_CONFIG,
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
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"F1: {test_metrics['f1']:.4f}")
    logger.info(f"Targets Met: {targets_met}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
