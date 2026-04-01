#!/usr/bin/env python3
"""
Train Sentence-BERT model for similarity
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pickle
import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer, 
    losses, 
    evaluation, 
    models
)
from sentence_transformers.readers import InputExample
import numpy as np
from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger, log_experiment
from scripts.utils.evaluation import TextModelEvaluator
import json
from tqdm import tqdm

class SentenceBERTTrainer:
    def __init__(self, config):
        self.config = config
        self.sbert_config = config['sentence_bert']
        self.model_dir = Path(config['paths']['models']['embeddings'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("sbert_training")
        self.evaluator = TextModelEvaluator(config)
        
        # Initialize model
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize Sentence-BERT model"""
        self.logger.info("Initializing Sentence-BERT model...")
        
        word_embedding_model = models.Transformer(
            self.sbert_config['base_model'],
            max_seq_length=self.sbert_config['max_seq_length']
        )
        
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode=self.sbert_config['pooling_mode']
        )
        
        # Optional: Add dense layer
        if self.sbert_config.get('use_dense_layer', False):
            dense_model = models.Dense(
                in_features=pooling_model.get_sentence_embedding_dimension(),
                out_features=256,
                activation_function=torch.nn.Tanh()
            )
            modules = [word_embedding_model, pooling_model, dense_model]
        else:
            modules = [word_embedding_model, pooling_model]
        
        return SentenceTransformer(modules=modules)
    
    def load_data(self):
        """Load similarity dataset"""
        dataset_path = Path(self.config['paths']['data']['datasets']) / "similarity_dataset.pkl"
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        self.logger.info(f"Loaded similarity dataset: "
                       f"Train={len(dataset['train'])}, "
                       f"Val={len(dataset['val'])}, "
                       f"Test={len(dataset['test'])}")
        
        # Convert to InputExample format
        train_examples = [
            InputExample(texts=[pair['text1'], pair['text2']], label=pair['label'])
            for pair in dataset['train']
        ]
        
        val_examples = [
            InputExample(texts=[pair['text1'], pair['text2']], label=pair['label'])
            for pair in dataset['val']
        ]
        
        test_examples = [
            InputExample(texts=[pair['text1'], pair['text2']], label=pair['label'])
            for pair in dataset['test']
        ]
        
        return train_examples, val_examples, test_examples
    
    def train(self, train_examples, val_examples):
        """Train the Sentence-BERT model"""
        self.logger.info("Starting Sentence-BERT training...")
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.sbert_config['batch_size']
        )
        
        # Create evaluator
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
            val_examples,
            name='lecture-val',
            show_progress_bar=True
        )
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Training parameters
        warmup_steps = int(len(train_dataloader) * self.sbert_config['epochs'] * 0.1)
        
        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.sbert_config['epochs'],
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=str(self.model_dir / "checkpoints"),
            save_best_model=True,
            checkpoint_path=str(self.model_dir / "checkpoints"),
            checkpoint_save_steps=1000,
            optimizer_params={'lr': self.sbert_config['learning_rate']},
            show_progress_bar=True
        )
        
        self.logger.info("Training completed")
    
    def evaluate(self, test_examples):
        """Evaluate model on test set"""
        self.logger.info("Evaluating on test set...")
        
        # Create evaluator
        test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
            test_examples,
            name='lecture-test',
            show_progress_bar=True
        )
        
        # Evaluate
        test_score = test_evaluator(self.model)
        
        # Additional evaluation: clustering quality
        embeddings = self.model.encode(
            [ex.texts[0] for ex in test_examples],
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # Calculate similarity matrix
        similarity_matrix = torch.nn.functional.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        ).cpu().numpy()
        
        # True similarity from labels
        true_similarity = np.zeros_like(similarity_matrix)
        for i in range(len(test_examples)):
            for j in range(len(test_examples)):
                if i == j:
                    true_similarity[i, j] = 1.0
                else:
                    # Simple approximation: label similarity
                    true_similarity[i, j] = 1.0 if test_examples[i].label == test_examples[j].label else 0.0
        
        # Calculate metrics
        metrics = self.evaluator.evaluate_similarity_model(
            embeddings.cpu().numpy(),
            similarity_matrix,
            true_similarity
        )
        
        metrics['test_score'] = test_score
        
        return metrics
    
    def save_final_model(self, test_metrics):
        """Save final model"""
        final_dir = self.model_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(final_dir))
        
        # Save model info
        model_info = {
            'config': self.sbert_config,
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'max_seq_length': self.sbert_config['max_seq_length'],
            'test_metrics': test_metrics
        }
        
        with open(final_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Save evaluation results
        self.evaluator.save_evaluation_results(
            {'test_metrics': test_metrics},
            'sentence_bert'
        )
        
        self.logger.info(f"Final model saved to {final_dir}")
        return final_dir
    
    def demo_similarity(self):
        """Demonstrate similarity calculation"""
        sample_texts = [
            "The Schrödinger equation describes quantum systems",
            "Quantum mechanics uses wave functions",
            "Newton's laws describe classical motion",
            "F = ma is Newton's second law"
        ]
        
        embeddings = self.model.encode(sample_texts)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        print("\nSimilarity Matrix:")
        for i, text1 in enumerate(sample_texts):
            for j, text2 in enumerate(sample_texts):
                if i <= j:
                    similarity = similarity_matrix[i, j]
                    print(f"{text1[:30]}... vs {text2[:30]}...: {similarity:.3f}")

def main():
    # Load configuration
    config = config_loader.load_all()
    
    # Initialize trainer
    trainer = SentenceBERTTrainer(config)
    
    # Load data
    train_examples, val_examples, test_examples = trainer.load_data()
    
    # Train model
    trainer.train(train_examples, val_examples)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_examples)
    
    # Save final model
    final_dir = trainer.save_final_model(test_metrics)
    
    # Demonstrate similarity
    trainer.demo_similarity()
    
    # Log experiment
    logger = setup_logger("experiment")
    experiment_results = {
        'model': 'Sentence_BERT',
        'test_metrics': test_metrics,
        'config': config['sentence_bert']
    }
    
    log_experiment(logger, config, experiment_results)
    
    print("\n" + "="*50)
    print("Sentence-BERT Training Complete!")
    print(f"Test Score: {test_metrics.get('test_score', 0):.4f}")
    print(f"Model saved to: {final_dir}")
    print("="*50)

if __name__ == "__main__":
    main()