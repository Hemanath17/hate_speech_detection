import os
import sys
import json
import logging
from typing import List, Dict, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.append('/Users/hemanatharumugam/Documents/Projects/Hatespeech')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThresholdTuner:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load model and tokenizer
        logger.info(f"Loading mBERT model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def predict_probabilities(self, text: str, language: str = None, max_length: int = 128) -> float:
        """Get hate speech probability for a given text"""
        # Add language token for better multilingual understanding
        if language and language != 'eng':
            text = f"[{language}] {text}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            hate_probability = probabilities[0][1].item()  # Probability of hate class
        
        return hate_probability
    
    def load_validation_data(self) -> Dict:
        """Load validation data for threshold tuning"""
        # Load the improved dataset
        data_path = 'Data/improved_multilingual_hate_speech_v2.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        import pandas as pd
        df = pd.read_csv(data_path)
        
        # Split into languages
        language_data = {}
        for lang in ['eng', 'tam', 'hin', 'spa', 'cmn']:
            lang_df = df[df['original_language'] == lang]
            if len(lang_df) > 0:
                texts = lang_df['text'].astype(str).tolist()
                labels = lang_df['isHate'].astype(int).tolist()
                language_data[lang] = {'texts': texts, 'labels': labels}
                logger.info(f"Loaded {len(texts)} samples for {lang}")
        
        return language_data
    
    def tune_thresholds(self, language_data: Dict, threshold_range: Tuple[float, float] = (0.1, 0.9), 
                       step: float = 0.05) -> Dict:
        """Tune thresholds for each language"""
        logger.info("Starting threshold tuning...")
        
        results = {}
        
        for lang, data in language_data.items():
            logger.info(f"\nTuning thresholds for {lang}...")
            
            texts = data['texts']
            labels = data['labels']
            
            # Get predictions for all texts
            logger.info("Getting predictions...")
            predictions = []
            for text in texts:
                prob = self.predict_probabilities(text, language=lang)
                predictions.append(prob)
            
            predictions = np.array(predictions)
            labels = np.array(labels)
            
            # Test different thresholds
            thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
            best_f1 = -1
            best_threshold = 0.5
            best_metrics = {}
            
            threshold_results = []
            
            for threshold in thresholds:
                # Apply threshold
                pred_labels = (predictions >= threshold).astype(int)
                
                # Calculate metrics
                f1 = f1_score(labels, pred_labels, average='macro', zero_division=0)
                precision = precision_score(labels, pred_labels, average='macro', zero_division=0)
                recall = recall_score(labels, pred_labels, average='macro', zero_division=0)
                accuracy = accuracy_score(labels, pred_labels)
                
                threshold_results.append({
                    'threshold': threshold,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy
                })
                
                # Update best threshold
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        'f1': f1,
                        'precision': precision,
                        'recall': recall,
                        'accuracy': accuracy
                    }
            
            results[lang] = {
                'best_threshold': best_threshold,
                'best_metrics': best_metrics,
                'all_results': threshold_results
            }
            
            logger.info(f"Best threshold for {lang}: {best_threshold:.3f} (F1: {best_f1:.3f})")
        
        return results
    
    def evaluate_with_tuned_thresholds(self, language_data: Dict, tuned_thresholds: Dict) -> Dict:
        """Evaluate model performance with tuned thresholds"""
        logger.info("Evaluating with tuned thresholds...")
        
        evaluation_results = {}
        
        for lang, data in language_data.items():
            if lang not in tuned_thresholds:
                continue
                
            texts = data['texts']
            labels = data['labels']
            threshold = tuned_thresholds[lang]['best_threshold']
            
            # Get predictions
            predictions = []
            for text in texts:
                prob = self.predict_probabilities(text, language=lang)
                predictions.append(prob)
            
            predictions = np.array(predictions)
            labels = np.array(labels)
            
            # Apply tuned threshold
            pred_labels = (predictions >= threshold).astype(int)
            
            # Calculate metrics
            f1 = f1_score(labels, pred_labels, average='macro', zero_division=0)
            precision = precision_score(labels, pred_labels, average='macro', zero_division=0)
            recall = recall_score(labels, pred_labels, average='macro', zero_division=0)
            accuracy = accuracy_score(labels, pred_labels)
            
            # Calculate confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(labels, pred_labels, labels=[0, 1])
            
            # Handle case where only one class is present
            if cm.shape == (1, 1):
                if 0 in labels and 0 in pred_labels:
                    # Only class 0 present
                    tp, fp, tn, fn = 0, 0, int(cm[0, 0]), 0
                else:
                    # Only class 1 present
                    tp, fp, tn, fn = int(cm[0, 0]), 0, 0, 0
            else:
                tp, fp, tn, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[0, 0]), int(cm[1, 0])
            
            evaluation_results[lang] = {
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
            
            logger.info(f"{lang} with tuned threshold {threshold:.3f}:")
            logger.info(f"  F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Accuracy: {accuracy:.3f}")
            logger.info(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        
        return evaluation_results
    
    def plot_threshold_curves(self, tuned_results: Dict, save_path: str = "models/threshold_curves.png"):
        """Plot threshold tuning curves for each language"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (lang, results) in enumerate(tuned_results.items()):
                if i >= len(axes):
                    break
                    
                all_results = results['all_results']
                thresholds = [r['threshold'] for r in all_results]
                f1_scores = [r['f1'] for r in all_results]
                precisions = [r['precision'] for r in all_results]
                recalls = [r['recall'] for r in all_results]
                
                ax = axes[i]
                ax.plot(thresholds, f1_scores, 'b-', label='F1 Score', linewidth=2)
                ax.plot(thresholds, precisions, 'r--', label='Precision', linewidth=2)
                ax.plot(thresholds, recalls, 'g--', label='Recall', linewidth=2)
                
                # Mark best threshold
                best_threshold = results['best_threshold']
                best_f1 = results['best_metrics']['f1']
                ax.axvline(x=best_threshold, color='k', linestyle=':', alpha=0.7)
                ax.plot(best_threshold, best_f1, 'ko', markersize=8)
                
                ax.set_xlabel('Threshold')
                ax.set_ylabel('Score')
                ax.set_title(f'{lang.upper()} - Threshold Tuning')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(tuned_results), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold curves saved to: {save_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            logger.warning(f"Error generating plots: {e}")


def main():
    logger.info("=" * 80)
    logger.info("mBERT THRESHOLD TUNING FOR IMPROVED DETECTION")
    logger.info("=" * 80)
    
    # Initialize tuner
    model_path = "models/mbert_improved_multilingual"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    
    tuner = ThresholdTuner(model_path)
    
    # Load validation data
    logger.info("Loading validation data...")
    language_data = tuner.load_validation_data()
    
    # Tune thresholds
    logger.info("Tuning thresholds...")
    tuned_results = tuner.tune_thresholds(language_data)
    
    # Save tuning results
    tuning_file = "models/threshold_tuning_results.json"
    with open(tuning_file, 'w') as f:
        json.dump(tuned_results, f, indent=2)
    logger.info(f"Tuning results saved to: {tuning_file}")
    
    # Evaluate with tuned thresholds
    logger.info("Evaluating with tuned thresholds...")
    evaluation_results = tuner.evaluate_with_tuned_thresholds(language_data, tuned_results)
    
    # Save evaluation results
    eval_file = "models/tuned_threshold_evaluation.json"
    with open(eval_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info(f"Evaluation results saved to: {eval_file}")
    
    # Generate plots
    tuner.plot_threshold_curves(tuned_results)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("THRESHOLD TUNING SUMMARY")
    logger.info("="*50)
    
    for lang, results in tuned_results.items():
        best_threshold = results['best_threshold']
        best_metrics = results['best_metrics']
        logger.info(f"{lang.upper()}:")
        logger.info(f"  Best Threshold: {best_threshold:.3f}")
        logger.info(f"  F1 Score: {best_metrics['f1']:.3f}")
        logger.info(f"  Precision: {best_metrics['precision']:.3f}")
        logger.info(f"  Recall: {best_metrics['recall']:.3f}")
        logger.info(f"  Accuracy: {best_metrics['accuracy']:.3f}")
        logger.info("")
    
    # Create optimized model wrapper
    optimized_thresholds = {lang: results['best_threshold'] for lang, results in tuned_results.items()}
    
    # Save optimized thresholds for inference
    thresholds_file = "models/optimized_thresholds.json"
    with open(thresholds_file, 'w') as f:
        json.dump(optimized_thresholds, f, indent=2)
    logger.info(f"Optimized thresholds saved to: {thresholds_file}")
    
    logger.info("Threshold tuning completed successfully!")


if __name__ == "__main__":
    main()
