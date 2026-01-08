"""
Multi-label DistilBERT model for hate speech detection
Optimized for 0.90 F1 score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig
from transformers import DistilBertTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)

class MultiLabelDistilBERT(nn.Module):
    """Multi-label DistilBERT model for hate speech detection"""
    
    def __init__(self, config: Dict):
        super(MultiLabelDistilBERT, self).__init__()
        
        self.config = config
        self.num_labels = len(config['categories'])
        
        # Load pre-trained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(
            config['distilbert_model'],
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Freeze DistilBERT layers initially
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        # Custom classification head for multi-label
        self.dropout1 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(768, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(256, self.num_labels)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better performance"""
        for module in [self.linear1, self.linear2, self.linear3]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass"""
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, 768]
        
        # Classification head
        x = self.dropout1(pooled_output)
        x = F.relu(self.linear1(x))
        x = self.dropout2(x)
        x = F.relu(self.linear2(x))
        x = self.dropout3(x)
        logits = self.linear3(x)
        
        return logits
    
    def unfreeze_distilbert(self, num_layers: int = 2):
        """Unfreeze last N layers of DistilBERT for fine-tuning"""
        # Unfreeze last num_layers
        for i in range(6 - num_layers, 6):
            for param in self.distilbert.transformer.layer[i].parameters():
                param.requires_grad = True
        logger.info(f"Unfroze last {num_layers} DistilBERT layers")

class HateSpeechDataset(Dataset):
    """Dataset class for hate speech detection"""
    
    def __init__(self, texts: List[str], labels: List[List[int]], 
                 tokenizer, max_length: int = 512):
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
            'labels': torch.tensor(label, dtype=torch.float)
        }

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        return loss.mean()

class HateSpeechTrainer:
    """Trainer class for hate speech detection model"""
    
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function with focal loss for imbalance
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['num_epochs']
        )
        
        # Best model tracking
        self.best_f1 = 0.0
        self.patience_counter = 0
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
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
                predictions = torch.sigmoid(logits).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar with current loss
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Convert to binary predictions
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Threshold predictions
        binary_predictions = (all_predictions > 0.5).astype(int)
        
        # Calculate metrics
        f1_macro = f1_score(all_labels, binary_predictions, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, binary_predictions, average='micro', zero_division=0)
        f1_weighted = f1_score(all_labels, binary_predictions, average='weighted', zero_division=0)
        
        # Per-class F1 scores
        f1_per_class = f1_score(all_labels, binary_predictions, average=None, zero_division=0)
        
        return {
            'loss': total_loss / len(dataloader),
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'predictions': binary_predictions,
            'labels': all_labels
        }
    
    def train(self, train_dataloader, val_dataloader, test_dataloader=None):
        """Train the model"""
        logger.info("Starting training...")
        
        # Create overall progress bar for epochs
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
                'Val F1': f'{val_metrics["f1_macro"]:.4f}',
                'Best F1': f'{self.best_f1:.4f}'
            })
            
            # Log metrics
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            logger.info(f"Epoch {epoch + 1} - Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Epoch {epoch + 1} - Val F1 Macro: {val_metrics['f1_macro']:.4f}")
            logger.info(f"Epoch {epoch + 1} - Val F1 Micro: {val_metrics['f1_micro']:.4f}")
            logger.info(f"Epoch {epoch + 1} - Val F1 Weighted: {val_metrics['f1_weighted']:.4f}")
            
            # Per-class F1 scores
            for i, f1 in enumerate(val_metrics['f1_per_class']):
                logger.info(f"Epoch {epoch + 1} - Class {i} F1: {f1:.4f}")
            
            # Check for improvement
            current_f1 = val_metrics['f1_macro']
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
                logger.info(f"🎉 New best model saved with F1: {current_f1:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"⏳ No improvement. Patience: {self.patience_counter}/{self.config['patience']}")
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        logger.info(f"Training completed. Best F1: {self.best_f1:.4f}")
        
        # Final evaluation on test set
        if test_dataloader:
            test_metrics = self.evaluate(test_dataloader)
            logger.info("Final Test Results:")
            logger.info(f"Test F1 Macro: {test_metrics['f1_macro']:.4f}")
            logger.info(f"Test F1 Micro: {test_metrics['f1_micro']:.4f}")
            logger.info(f"Test F1 Weighted: {test_metrics['f1_weighted']:.4f}")
            
            return test_metrics
        
        return val_metrics

def create_data_loaders(train_texts, train_labels, val_texts, val_labels, 
                       test_texts, test_labels, tokenizer, config):
    """Create data loaders for training"""
    
    # Create datasets
    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, config['max_length'])
    val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, config['max_length'])
    test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer, config['max_length'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader

def main():
    """Test the model"""
    from configs.config import MODEL_CONFIG, CATEGORIES
    
    # Test model creation
    model = MultiLabelDistilBERT({'categories': CATEGORIES, **MODEL_CONFIG})
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

if __name__ == "__main__":
    main()
