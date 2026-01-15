#!/usr/bin/env python3
"""
Train NER model for STEM term detection using Transformers and NLTK
"""
"""
Train NER model for STEM term detection using Transformers - Colab Compatible
"""
import sys
import os
from pathlib import Path

# --- Dynamic Path Setup for Colab & Local ---
def get_project_root():
    """Dynamically find the project root in both Colab and local environments."""
    # Try to find a known project directory
    possible_roots = [
        Path.cwd(),  # Current working directory
        Path.cwd().parent,  # One level up
        Path('/content/lecture_summarization'),  # Common Colab path
        Path('/content/drive/MyDrive/lecture_summarization')  # Google Drive path
    ]
    
    for root in possible_roots:
        scripts_dir = root / 'scripts' / 'text_processing'
        if scripts_dir.exists():
            print(f"Project root found at: {root}")
            return root
    
    # Fallback to current directory
    print(f"Warning: Project root not found, using current directory: {Path.cwd()}")
    return Path.cwd()

# Set project root and add to path
# Set project root and add to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

# Previous project_root finder
#  # Add project root to path
# project_root = Path(__file__).parent.parent.parent
# sys.path.append(str(project_root))

import pickle
import random
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import evaluate
from sklearn.metrics import classification_report
import nltk
from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger, log_experiment
import json
from tqdm import tqdm

##  PRRVIOUS CODE CHANGING DUE TO COLAB TILL DATA_LOAD FUNC
# class STEMNERTrainer:
#     def __init__(self, config):
#         self.config = config
#         self.ner_config = config['ner']
#         self.model_dir = Path(config['paths']['models']['ner'])
#         self.model_dir.mkdir(parents=True, exist_ok=True)
        
#         self.logger = setup_logger("ner_training")
        
#         # Download NLTK resources if needed
#         self._setup_nltk()
        
#         # Initialize model and tokenizer
#         self.tokenizer, self.model = self._initialize_model()
        
#         # Label mapping
#         self.label2id = {label: i for i, label in enumerate(self.config['data']['stem_categories'])}
#         self.id2label = {i: label for label, i in self.label2id.items()}
        
#         self.best_f1 = 0.0
        
#     def _setup_nltk(self):
#         """Download required NLTK resources"""
#         try:
#             nltk.data.find('tokenizers/punkt')
#             nltk.data.find('taggers/averaged_perceptron_tagger')
#         except LookupError:
#             self.logger.info("Downloading NLTK resources...")
#             nltk.download('punkt', quiet=True)
#             nltk.download('averaged_perceptron_tagger', quiet=True)
    
#     def _initialize_model(self):
#         """Initialize transformer model for token classification"""
#         self.logger.info(f"Initializing {self.ner_config['model_base']} model...")
        
#         # Load tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(
#             self.ner_config['model_base'],
#             use_fast=True
#         )
        
#         # Load model
#         model = AutoModelForTokenClassification.from_pretrained(
#             self.ner_config['model_base'],
#             num_labels=len(self.config['data']['stem_categories']),
#             id2label=self.id2label,
#             label2id=self.label2id
#         )
        
#         return tokenizer, model
    
#     def load_data(self):
#         """Load training data"""
#         dataset_path = Path(self.config['paths']['data']['datasets']) / "ner_dataset.pkl"
        
#         with open(dataset_path, 'rb') as f:
#             dataset = pickle.load(f)
        
#         self.logger.info(f"Loaded NER dataset: {dataset['stats']}")
        
#         # Convert to transformer format
#         train_data = self._convert_to_transformers_format(dataset['train'])
#         val_data = self._convert_to_transformers_format(dataset['val'])
#         test_data = self._convert_to_transformers_format(dataset['test'])
        
#         # Convert to Dataset objects
#         train_dataset = Dataset.from_dict(train_data)
#         val_dataset = Dataset.from_dict(val_data)
#         test_dataset = Dataset.from_dict(test_data)
        
#         return train_dataset, val_dataset, test_dataset

class STEMNERTrainer:
    def __init__(self, config):
        self.config = config
        self.ner_config = config['ner']
        
        # --- Colab-Compatible Model Directory ---
        # Save to Google Drive if in Colab, else local
        if 'COLAB_GPU' in os.environ or 'colab' in str(project_root).lower():
            # Colab environment - save to Drive for persistence
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            drive_root = '/content/drive/MyDrive'
            self.model_dir = Path(drive_root) / 'lecture_summarization' / 'models' / 'ner'
            print(f"Colab mode: Models will be saved to Google Drive: {self.model_dir}")
        else:
            # Local environment
            self.model_dir = Path(config['paths']['models']['ner'])
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("ner_training")
        
        # Download NLTK resources if needed
        self._setup_nltk()
        
        # Initialize model and tokenizer
        self.tokenizer, self.model = self._initialize_model()
        
        # Label mapping
        self.label2id = {label: i for i, label in enumerate(self.config['data']['stem_categories'])}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        self.best_f1 = 0.0
        
    def _setup_nltk(self):
        """Download required NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            self.logger.info("Downloading NLTK resources...")
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def _initialize_model(self):
        """Initialize transformer model for token classification"""
        model_name = self.ner_config['model_base']
        self.logger.info(f"Initializing {model_name} model...")
        
        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )
        
        # Load model
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.config['data']['stem_categories']),
            id2label=self.id2label,
            label2id=self.label2id
        ).to(device)
        
        return tokenizer, model
    
    def load_data(self):
        """Load training data - Colab compatible paths"""
        # Dynamically find the datasets directory
        possible_data_dirs = [
            project_root / 'data' / 'datasets',
            Path('data/datasets'),
            Path('/content/data/datasets'),
            Path('/content/drive/MyDrive/lecture_summarization/data/datasets')
        ]
        
        dataset_path = None
        for data_dir in possible_data_dirs:
            temp_path = data_dir / "ner_dataset.pkl"
            if temp_path.exists():
                dataset_path = temp_path
                break
        
        if dataset_path is None:
            raise FileNotFoundError("NER dataset not found. Please ensure data preparation is complete.")
        
        self.logger.info(f"Loading NER dataset from: {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # ... [Rest of the load_data method remains the same as your last version] ...
        # Convert to transformer format
        train_data = self._convert_to_transformers_format(dataset['train'])
        val_data = self._convert_to_transformers_format(dataset['val'])
        test_data = self._convert_to_transformers_format(dataset['test'])
        
        # Convert to Dataset objects
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        test_dataset = Dataset.from_dict(test_data)
        
        return train_dataset, val_dataset, test_dataset


    def _convert_to_transformers_format(self, data):
        """Convert data to transformers format"""
        texts = []
        labels = []
        
        for item in tqdm(data, desc="Converting data"):
            text = item['text']
            entities = item['entities']
            
            # Tokenize with NLTK to get word positions
            words = nltk.word_tokenize(text)
            
            # Get character positions for each word
            word_positions = []
            char_pos = 0
            for word in words:
                idx = text.find(word, char_pos)
                if idx != -1:
                    word_positions.append((idx, idx + len(word)))
                    char_pos = idx + len(word)
                else:
                    word_positions.append((char_pos, char_pos + len(word)))
                    char_pos += len(word) + 1
            
            # Assign labels to words
            word_labels = ["O"] * len(words)
            
            for start, end, label in entities:
                # Find words that overlap with entity span
                for i, (word_start, word_end) in enumerate(word_positions):
                    if word_start < end and word_end > start:
                        # Check if word is fully inside entity
                        if start <= word_start and word_end <= end:
                            # First word of multi-word entity gets B- prefix, others get I-
                            if i == 0 or word_labels[i-1] == "O":
                                word_labels[i] = f"B-{label}"
                            else:
                                word_labels[i] = f"I-{label}"
                        else:
                            # Partial overlap (rare) - use B- prefix
                            word_labels[i] = f"B-{label}"
            
            # Convert to label IDs
            label_ids = []
            for label in word_labels:
                if label == "O":
                    label_ids.append(0)
                else:
                    # Extract base label
                    base_label = label.split("-", 1)[1]
                    label_ids.append(self.label2id.get(base_label, 0))
            
            # Join words back to text
            texts.append(" ".join(words))
            labels.append(label_ids)
        
        return {
            "tokens": texts,
            "ner_tags": labels
        }
    
    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels with tokenizer"""
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            padding=True,
            max_length=self.ner_config.get('max_length', 128),
            is_split_into_words=True,
            return_offsets_mapping=True
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special token
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    # Same word, different subword - keep label or use -100
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def compute_metrics(self, p):
        """Compute metrics for evaluation"""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            for p, l in zip(prediction, label):
                if l != -100:
                    true_predictions.append(self.id2label[p])
                    true_labels.append(self.id2label[l])
        
        # Load seqeval metric
        metric = evaluate.load("seqeval")
        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        # Extract metrics
        precision = results["overall_precision"]
        recall = results["overall_recall"]
        f1 = results["overall_f1"]
        accuracy = results["overall_accuracy"]
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }
    
    def train(self, train_dataset, val_dataset):
        """Train the NER model"""
        self.logger.info("Starting NER training with Transformers...")
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        tokenized_val = val_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.model_dir / "checkpoints"),
            num_train_epochs=self.ner_config['n_iter'],
            per_device_train_batch_size=self.ner_config.get('batch_size', 8),
            per_device_eval_batch_size=self.ner_config.get('batch_size', 8),
            warmup_ratio=self.ner_config.get('warmup_ratio', 0.1),
            weight_decay=self.ner_config.get('weight_decay', 0.01),
            logging_dir=str(self.model_dir / "logs"),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=self.ner_config.get('learning_rate', 2e-5),
            report_to="none",
            seed=self.config['project']['random_seed']
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        trainer.train()
        
        # Get best metrics
        self.best_f1 = trainer.state.best_metric
        
        self.logger.info(f"Training completed. Best F1: {self.best_f1:.4f}")
        
        return trainer
    
    def evaluate(self, trainer, test_dataset):
        """Evaluate model on test set"""
        self.logger.info("Evaluating on test set...")
        
        # Tokenize test set
        tokenized_test = test_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=test_dataset.column_names
        )
        
        # Evaluate
        results = trainer.evaluate(tokenized_test)
        
        # Detailed predictions
        predictions = trainer.predict(tokenized_test)
        pred_labels = np.argmax(predictions.predictions, axis=2)
        
        # Collect all predictions and true labels
        all_true_labels = []
        all_pred_labels = []
        
        for i in range(len(pred_labels)):
            for j in range(len(pred_labels[i])):
                if predictions.label_ids[i][j] != -100:
                    true_label = self.id2label[predictions.label_ids[i][j]]
                    pred_label = self.id2label[pred_labels[i][j]]
                    all_true_labels.append(true_label)
                    all_pred_labels.append(pred_label)
        
        # Classification report
        report = classification_report(
            all_true_labels, 
            all_pred_labels,
            target_names=list(self.label2id.keys()) + ["O"],
            output_dict=True
        )
        
        results["detailed_report"] = report
        
        return results
    
    def save_final_model(self, trainer, test_metrics):
        """Save final model"""
        final_dir = self.model_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        trainer.save_model(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        
        # Save model info
        model_info = {
            "model_name": self.ner_config['model_base'],
            "labels": self.config['data']['stem_categories'],
            "label2id": self.label2id,
            "id2label": self.id2label,
            "best_f1": self.best_f1,
            "test_metrics": test_metrics,
            "config": self.ner_config
        }
        
        with open(final_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self.logger.info(f"Final model saved to {final_dir}")
        
        return test_metrics

def main():
    # Load configuration
    config = config_loader.load_all()
    
    # Initialize trainer
    trainer = STEMNERTrainer(config)
    
    try:
        # Load data
        train_dataset, val_dataset, test_dataset = trainer.load_data()
        
        # Train model
        transformer_trainer = trainer.train(train_dataset, val_dataset)
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(transformer_trainer, test_dataset)
        
        # Save final model
        final_test_metrics = trainer.save_final_model(transformer_trainer, test_metrics)
        
        # Log experiment
        logger = setup_logger("experiment")
        experiment_results = {
            "model": "Transformer_NER_Colab",
            "best_f1": trainer.best_f1,
            "test_metrics": final_test_metrics,
            "config": config['ner']
        }
        
        log_experiment(logger, config, experiment_results)
        
        print("\n" + "="*50)
        print("NER Training Complete!")
        print(f"Best F1 Score: {trainer.best_f1:.4f}")
        print(f"Test F1 Score: {test_metrics.get('eval_f1', 0):.4f}")
        print(f"Model saved to: {trainer.model_dir / 'final'}")
        print("="*50)
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
