import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# Ensure project root is on path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedChineseMultilingualDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], languages: List[str], 
                 tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        language = str(self.languages[idx])
        
        # Add language token for better multilingual understanding
        if language != 'eng':
            text = f"[{language}] {text}"
        
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        
        return {
            'input_ids': torch.tensor(enc['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(enc['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'language': language
        }


def load_improved_chinese_dataset(data_path: str) -> Tuple[List[str], List[int], List[str]]:
    """Load improved Chinese dataset"""
    logger.info(f"Loading dataset from: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Extract data
    texts = df['text'].astype(str).tolist()
    labels = df['isHate'].astype(int).tolist()
    languages = df['original_language'].astype(str).tolist()
    
    # Statistics
    logger.info(f"Language distribution:")
    lang_counts = pd.Series(languages).value_counts()
    for lang, count in lang_counts.items():
        lang_labels = [labels[i] for i, l in enumerate(languages) if l == lang]
        hate_count = sum(lang_labels)
        logger.info(f"  {lang}: {count} samples, {hate_count} hate ({hate_count/count*100:.1f}%)")
    
    return texts, labels, languages


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.2f}%'
        })
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device, language_specific=False):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_languages = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            languages = batch['language']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_languages.extend(languages)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1
    }
    
    # Language-specific metrics
    if language_specific:
        lang_metrics = {}
        for lang in set(all_languages):
            lang_indices = [i for i, l in enumerate(all_languages) if l == lang]
            if len(lang_indices) > 0:
                lang_labels = [all_labels[i] for i in lang_indices]
                lang_preds = [all_predictions[i] for i in lang_indices]
                lang_metrics[lang] = {
                    'f1': f1_score(lang_labels, lang_preds, average='binary', zero_division=0),
                    'accuracy': accuracy_score(lang_labels, lang_preds),
                    'samples': len(lang_indices)
                }
        metrics['language_specific'] = lang_metrics
    
    return metrics


def main():
    """Main training function"""
    logger.info("=" * 80)
    logger.info("RETRAINING mBERT WITH IMPROVED CHINESE DATASET")
    logger.info("=" * 80)
    
    # Configuration
    model_name = 'bert-base-multilingual-cased'
    data_path = 'Data/expanded_multilingual_hate_speech_improved_chinese.csv'
    output_dir = 'models/mbert_improved_chinese_multilingual'
    max_length = 128
    batch_size = 32
    learning_rate = 2e-5
    num_epochs = 5
    warmup_steps = 500
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    texts, labels, languages = load_improved_chinese_dataset(data_path)
    
    # Split data
    logger.info("\nSplitting data...")
    X_temp, X_test, y_temp, y_test, lang_temp, lang_test = train_test_split(
        texts, labels, languages, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val, lang_train, lang_val = train_test_split(
        X_temp, y_temp, lang_temp, test_size=0.1, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Load tokenizer and model
    logger.info(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # Create datasets
    train_dataset = ImprovedChineseMultilingualDataset(X_train, y_train, lang_train, tokenizer, max_length)
    val_dataset = ImprovedChineseMultilingualDataset(X_val, y_val, lang_val, tokenizer, max_length)
    test_dataset = ImprovedChineseMultilingualDataset(X_test, y_test, lang_test, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    best_val_f1 = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, language_specific=True)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['accuracy']*100:.2f}%")
        logger.info(f"Val F1-Macro: {val_metrics['f1_macro']*100:.2f}%")
        
        # Language-specific results
        if 'language_specific' in val_metrics:
            logger.info("Language-specific validation results:")
            for lang, metrics in val_metrics['language_specific'].items():
                logger.info(f"  {lang}: F1={metrics['f1']*100:.2f}%, Acc={metrics['accuracy']*100:.2f}% ({metrics['samples']} samples)")
        
        # Save best model
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            patience_counter = 0
            
            # Save model
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save validation metrics
            with open(os.path.join(output_dir, 'val_metrics.json'), 'w') as f:
                json.dump(val_metrics, f, indent=2)
            
            logger.info(f"✓ Saved best model (F1: {best_val_f1*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered (patience: {patience})")
                break
    
    # Final test evaluation
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 80)
    
    # Load best model
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    
    test_metrics = evaluate(model, test_loader, device, language_specific=True)
    
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"Test Precision: {test_metrics['precision']*100:.2f}%")
    logger.info(f"Test Recall: {test_metrics['recall']*100:.2f}%")
    logger.info(f"Test F1-Macro: {test_metrics['f1_macro']*100:.2f}%")
    
    # Language-specific test results
    if 'language_specific' in test_metrics:
        logger.info("\nLanguage-specific test results:")
        for lang, metrics in test_metrics['language_specific'].items():
            logger.info(f"  {lang}: F1={metrics['f1']*100:.2f}%, Acc={metrics['accuracy']*100:.2f}% ({metrics['samples']} samples)")
    
    # Save test metrics
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Best Test F1-Macro: {test_metrics['f1_macro']*100:.2f}%")
    logger.info(f"Best Test Accuracy: {test_metrics['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()

