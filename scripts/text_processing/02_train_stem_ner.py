#!/usr/bin/env python3
"""
Train spaCy NER model for STEM term detection
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pickle
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import torch
from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger, log_experiment
from scripts.utils.evaluation import TextModelEvaluator
from tqdm import tqdm
import json

class STEMNERTrainer:
    def __init__(self, config):
        self.config = config
        self.ner_config = config['ner']
        self.model_dir = Path(config['paths']['models']['ner'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("ner_training")
        self.evaluator = TextModelEvaluator(config)
        
        # Initialize model
        self.nlp = self._initialize_model()
        self.best_f1 = 0.0
        
    def _initialize_model(self):
        """Initialize spaCy model"""
        self.logger.info("Initializing spaCy model...")
        
        try:
            # Try scientific model
            nlp = spacy.load(self.ner_config['model_base'])
            self.logger.info(f"Loaded {self.ner_config['model_base']} model")
        except:
            # Fallback to web model
            self.logger.warning(f"{self.ner_config['model_base']} not found, using en_core_web_sm")
            try:
                nlp = spacy.load("en_core_web_sm")
            except:
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
        
        # Add NER pipe if not present
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")
        
        # Add STEM labels
        for label in self.config['data']['stem_categories']:
            ner.add_label(label)
        
        return nlp
    
    def load_data(self):
        """Load training data"""
        dataset_path = Path(self.config['paths']['data']['datasets']) / "ner_dataset.pkl"
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        self.logger.info(f"Loaded NER dataset: {dataset['stats']}")
        
        # Convert to spaCy format
        train_data = self._convert_to_spacy_format(dataset['train'])
        val_data = self._convert_to_spacy_format(dataset['val'])
        test_data = self._convert_to_spacy_format(dataset['test'])
        
        return train_data, val_data, test_data
    
    def _convert_to_spacy_format(self, data):
        """Convert to spaCy training format"""
        formatted = []
        for item in data:
            entities = [(start, end, label) for start, end, label in item['entities']]
            formatted.append((item['text'], {"entities": entities}))
        return formatted
    
    def train(self, train_data, val_data):
        """Train the NER model"""
        self.logger.info("Starting NER training...")
        
        # Disable other pipes
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.create_optimizer()
            
            # Training loop
            for epoch in range(self.ner_config['n_iter']):
                self.logger.info(f"Epoch {epoch + 1}/{self.ner_config['n_iter']}")
                
                # Shuffle training data
                random.shuffle(train_data)
                losses = {}
                
                # Create batches
                batches = minibatch(train_data, size=compounding(
                    4.0, 32.0, 1.001
                ))
                
                # Training batches
                for batch in tqdm(batches, desc="Training", leave=False):
                    examples = []
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    
                    # Update model
                    self.nlp.update(examples, drop=self.ner_config['dropout'], 
                                  losses=losses, sgd=optimizer)
                
                # Evaluate on validation set
                if (epoch + 1) % self.ner_config['validation_frequency'] == 0:
                    val_metrics = self.evaluate(val_data)
                    self.logger.info(f"Validation - F1: {val_metrics['f1']:.4f}, "
                                   f"Precision: {val_metrics['precision']:.4f}, "
                                   f"Recall: {val_metrics['recall']:.4f}")
                    
                    # Save best model
                    if val_metrics['f1'] > self.best_f1:
                        self.best_f1 = val_metrics['f1']
                        self.save_checkpoint(epoch, val_metrics)
                        
                        # Early stopping check
                        improvement = val_metrics['f1'] - self.best_f1
                        if improvement < self.ner_config['early_stopping_threshold']:
                            self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                            break
        
        self.logger.info(f"Training completed. Best F1: {self.best_f1:.4f}")
    
    def evaluate(self, data):
        """Evaluate model on given data"""
        predictions = []
        ground_truth = []
        
        for text, annotations in data:
            doc = self.nlp(text)
            
            # Get predicted entities
            pred_entities = [(ent.start_char, ent.end_char, ent.label_) 
                           for ent in doc.ents]
            
            # Get ground truth entities
            true_entities = [(start, end, label) for start, end, label in annotations['entities']]
            
            predictions.append({"text": text, "entities": pred_entities})
            ground_truth.append({"text": text, "entities": true_entities})
        
        # Calculate metrics
        results = self.evaluator.evaluate_ner_model(predictions, ground_truth)
        return results['metrics']
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint_dir = self.model_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"ner_epoch_{epoch+1}.spacy"
        self.nlp.to_disk(checkpoint_path)
        
        # Save metrics
        metrics_file = checkpoint_dir / f"metrics_epoch_{epoch+1}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self, test_data):
        """Save final model and evaluate on test set"""
        self.logger.info("Evaluating on test set...")
        test_metrics = self.evaluate(test_data)
        
        # Save final model
        final_dir = self.model_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        self.nlp.to_disk(final_dir)
        
        # Save evaluation results
        self.evaluator.save_evaluation_results(
            {"test_metrics": test_metrics},
            "ner_model"
        )
        
        self.logger.info(f"Final model saved to {final_dir}")
        self.logger.info(f"Test metrics - F1: {test_metrics['f1']:.4f}")
        
        return test_metrics

def main():
    # Load configuration
    config = config_loader.load_all()
    
    # Initialize trainer
    trainer = STEMNERTrainer(config)
    
    # Load data
    train_data, val_data, test_data = trainer.load_data()
    
    # Train model
    trainer.train(train_data, val_data)
    
    # Evaluate on test set
    test_metrics = trainer.save_final_model(test_data)
    
    # Log experiment
    logger = setup_logger("experiment")
    experiment_results = {
        "model": "STEM_NER",
        "test_metrics": test_metrics,
        "best_f1": trainer.best_f1,
        "config": config['ner']
    }
    
    log_experiment(logger, config, experiment_results)
    
    print("\n" + "="*50)
    print("NER Training Complete!")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Model saved to: {trainer.model_dir / 'final'}")
    print("="*50)

if __name__ == "__main__":
    main()