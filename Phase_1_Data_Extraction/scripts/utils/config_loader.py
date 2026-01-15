import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        
    def load_all(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            # Create default config if doesn't exist
            self._create_default_config()
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set default values if not present
        self._set_defaults()
        
        return self.config
    
    def _create_default_config(self):
        """Create default configuration"""
        default_config = {
            'project': {
                'name': 'video_summarizer',
                'random_seed': 42
            },
            'paths': {
                'data': {
                    'annotations': 'data/annotations',
                    'datasets': 'data/datasets'
                },
                'models': {
                    'ner': 'models/ner',
                    'topic': 'models/topic',
                    'embeddings': 'models/embeddings',
                    'text': 'models/text'
                },
                'results': 'results'
            },
            'data': {
                'test_ratio': 0.2,
                'val_ratio': 0.1,
                'stem_categories': ['EQUATION', 'CONCEPT', 'METHOD', 'THEOREM', 'FORMULA']
            },
            'ner': {
                'model_base': 'bert-base-uncased',
                'n_iter': 10,
                'validation_frequency': 2,
                'dropout': 0.3,
                'early_stopping_threshold': 0.001
            },
            'topic': {
                'embedding_model': 'all-MiniLM-L6-v2',
                'min_topic_size': 10,
                'n_gram_range': [1, 2],
                'nr_topics': 'auto',
                'top_n_words': 10,
                'diversity': 0.1
            },
            'sentence_bert': {
                'base_model': 'bert-base-uncased',
                'max_seq_length': 256,
                'pooling_mode': 'mean',
                'use_dense_layer': False,
                'epochs': 3,
                'batch_size': 16,
                'learning_rate': 2e-5,
                'warmup_ratio': 0.1
            }
        }
        
        # Create directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default configuration at {self.config_path}")
    
    def _set_defaults(self):
        """Set default values for missing keys"""
        defaults = {
            'project': {'random_seed': 42},
            'paths': {
                'data': {'annotations': 'data/annotations', 'datasets': 'data/datasets'},
                'models': {
                    'ner': 'models/ner',
                    'topic': 'models/topic',
                    'embeddings': 'models/embeddings',
                    'text': 'models/text'
                },
                'results': 'results'
            },
            'data': {
                'test_ratio': 0.2,
                'val_ratio': 0.1,
                'stem_categories': ['EQUATION', 'CONCEPT']
            },
            'ner': {
                'n_iter': 10,
                'validation_frequency': 2,
                'dropout': 0.3,
                'early_stopping_threshold': 0.001,
                'learning_rate': 2e-5,
                'batch_size': 8,
                'max_length': 128
            },
            'topic': {
                'min_topic_size': 10,
                'n_gram_range': [1, 2],
                'nr_topics': 'auto',
                'top_n_words': 10
            },
            'sentence_bert': {
                'epochs': 3,
                'batch_size': 16,
                'learning_rate': 2e-5
            }
        }
        
        # Merge defaults
        for section, section_defaults in defaults.items():
            if section not in self.config:
                self.config[section] = section_defaults
            else:
                for key, value in section_defaults.items():
                    if key not in self.config[section]:
                        self.config[section][key] = value

# Create singleton instance
config_loader = ConfigLoader()