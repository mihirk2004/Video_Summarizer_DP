#!/usr/bin/env python3
"""
Data preparation script for text models
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.data_loader import LectureDataLoader
from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger, log_experiment
import pandas as pd

def main():
    # Setup
    config = config_loader.load_all()
    logger = setup_logger("data_preparation", level="INFO")
    
    logger.info("Starting data preparation...")
    
    # Initialize data loader
    data_loader = LectureDataLoader(config)
    
    # Load annotations
    logger.info("Loading annotations...")
    annotations = data_loader.load_all_annotations()
    
    if not annotations:
        logger.error("No annotations found!")
        return
    
    # Extract transcript data
    logger.info("Extracting transcript data...")
    df = data_loader.extract_transcript_data(annotations)
    
    # Generate statistics
    stats = data_loader.get_data_statistics(df)
    logger.info(f"Dataset statistics: {stats}")
    
    # Create NER dataset
    logger.info("Creating NER dataset...")
    ner_dataset = data_loader.create_ner_dataset(df)
    logger.info(f"NER dataset: {ner_dataset['stats']}")
    
    # Create topic dataset
    logger.info("Creating topic dataset...")
    topic_dataset = data_loader.create_topic_dataset(df)
    logger.info(f"Topic dataset created for {len(topic_dataset)} lectures")
    
    # Create similarity dataset
    logger.info("Creating similarity dataset...")
    similarity_dataset = data_loader.create_similarity_dataset(df)
    
    # Log experiment
    experiment_results = {
        "statistics": stats,
        "ner_dataset_size": ner_dataset['stats'],
        "topic_dataset_lectures": len(topic_dataset),
        "similarity_pairs": {
            "train": len(similarity_dataset['train']),
            "val": len(similarity_dataset['val']),
            "test": len(similarity_dataset['test'])
        }
    }
    
    log_experiment(logger, config, experiment_results)
    
    # Save summary
    summary_file = Path("results/text_models/data_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "data_preparation": {
            "timestamp": pd.Timestamp.now().isoformat(),
            "statistics": stats,
            "datasets_created": [
                "ner_dataset.pkl",
                "topic_dataset.pkl",
                "similarity_dataset.pkl"
            ]
        }
    }
    
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Data preparation complete! Summary saved to {summary_file}")

if __name__ == "__main__":
    main()