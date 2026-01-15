#!/usr/bin/env python3
"""
Data preparation script for text models using NLTK
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import nltk
import json
import traceback

def setup_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        print("NLTK resources already downloaded.")
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("NLTK resources downloaded successfully.")

def main():
    # Setup NLTK first
    setup_nltk_resources()
    
    # Try to load configuration
    try:
        from scripts.utils.config_loader import config_loader
        config = config_loader.load_all()
        print("Configuration loaded successfully.")
    except ImportError as e:
        print(f"Configuration module not found: {e}")
        # Create minimal config
        config = {
            'project': {'random_seed': 42},
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
                }
            },
            'data': {
                'test_ratio': 0.2,
                'val_ratio': 0.1,
                'stem_categories': ['EQUATION', 'CONCEPT']
            }
        }
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Setup logger
    try:
        from scripts.utils.logger import setup_logger, log_experiment
        logger = setup_logger("data_preparation", level="INFO")
    except ImportError as e:
        # Create simple logger if module not found
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger("data_preparation")
    
    logger.info("Starting data preparation with NLTK...")
    
    # Initialize data loader
    try:
        from scripts.utils.data_loader import LectureDataLoader
        data_loader = LectureDataLoader(config)
    except ImportError as e:
        logger.error(f"Failed to import data_loader: {e}")
        traceback.print_exc()
        return
    except Exception as e:
        logger.error(f"Error initializing data loader: {e}")
        traceback.print_exc()
        return
    
    # Load annotations
    logger.info("Loading annotations...")
    try:
        annotations = data_loader.load_all_annotations()
    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        traceback.print_exc()
        return
    
    if not annotations:
        logger.error("No annotations found!")
        return
    
    # Extract transcript data using NLTK
    logger.info("Extracting transcript data using NLTK...")
    try:
        df = data_loader.extract_transcript_data(annotations)
    except Exception as e:
        logger.error(f"Error extracting transcript data: {e}")
        traceback.print_exc()
        return
    
    # Check if we got any data
    if df is None or len(df) == 0:
        logger.error("No data extracted!")
        return
    
    # Generate statistics
    stats = data_loader.get_data_statistics(df)
    logger.info(f"Dataset statistics: {stats}")
    
    # Create NER dataset using NLTK
    logger.info("Creating NER dataset using NLTK...")
    try:
        ner_dataset = data_loader.create_ner_dataset(df)
        logger.info(f"NER dataset stats: {ner_dataset['stats']}")
    except Exception as e:
        logger.error(f"Error creating NER dataset: {e}")
        traceback.print_exc()
        ner_dataset = {'stats': {'total': 0, 'train': 0, 'val': 0, 'test': 0}}
    
    # Create topic dataset
    logger.info("Creating topic dataset...")
    try:
        topic_dataset = data_loader.create_topic_dataset(df)
        logger.info(f"Topic dataset created for {len(topic_dataset)} lectures")
    except Exception as e:
        logger.error(f"Error creating topic dataset: {e}")
        traceback.print_exc()
        topic_dataset = {}
    
    # Create similarity dataset
    logger.info("Creating similarity dataset...")
    try:
        similarity_dataset = data_loader.create_similarity_dataset(df)
        logger.info(f"Similarity dataset created with {len(similarity_dataset.get('train', []))} train, "
                   f"{len(similarity_dataset.get('val', []))} val, {len(similarity_dataset.get('test', []))} test pairs")
    except Exception as e:
        logger.error(f"Error creating similarity dataset: {e}")
        traceback.print_exc()
        similarity_dataset = {'train': [], 'val': [], 'test': []}
    
    # Try to log experiment
    try:
        experiment_results = {
            "statistics": stats,
            "ner_dataset_size": ner_dataset.get('stats', {}),
            "topic_dataset_lectures": len(topic_dataset),
            "similarity_pairs": {
                "train": len(similarity_dataset.get('train', [])),
                "val": len(similarity_dataset.get('val', [])),
                "test": len(similarity_dataset.get('test', []))
            }
        }
        
        log_experiment(logger, config, experiment_results)
    except Exception as e:
        logger.warning(f"Could not log experiment: {e}")
    
    # Save summary
        # Save summary with proper type conversion
    summary_file = Path("results/text_models/data_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert stats to JSON-serializable format
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    summary = {
        "data_preparation": {
            "timestamp": pd.Timestamp.now().isoformat(),
            "text_processor": "NLTK",
            "nltk_version": nltk.__version__,
            "statistics": convert_for_json(stats),
            "datasets_created": [
                "ner_dataset.pkl",
                "topic_dataset.pkl",
                "similarity_dataset.pkl"
            ]
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Data preparation complete! Summary saved to {summary_file}")
    
    # Print final summary
    print("\n" + "="*60)
    print("DATA PREPARATION SUMMARY")
    print("="*60)
    print(f"Total lectures processed: {stats.get('total_lectures', 0)}")
    print(f"Total segments extracted: {stats.get('total_segments', 0)}")
    print(f"NER dataset size: {ner_dataset.get('stats', {}).get('total', 0)}")
    print(f"Topic dataset lectures: {len(topic_dataset)}")
    print(f"Similarity pairs: {len(similarity_dataset.get('train', [])) + len(similarity_dataset.get('val', [])) + len(similarity_dataset.get('test', []))}")
    print("="*60)

if __name__ == "__main__":
    main()