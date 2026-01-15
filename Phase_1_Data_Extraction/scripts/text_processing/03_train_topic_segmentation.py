#!/usr/bin/env python3
"""
Train BERTopic model for topic segmentation
"""
import sys
from pathlib import Path

# --- Dynamic Path Setup (Same as above) ---
## FOR GOOGLE COLAB 
def get_project_root():
    """Dynamically find the project root."""
    possible_roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path('/content/lecture_summarization'),
        Path('/content/drive/MyDrive/lecture_summarization')
    ]
    
    for root in possible_roots:
        scripts_dir = root / 'scripts' / 'text_processing'
        if scripts_dir.exists():
            print(f"Project root found at: {root}")
            return root
    
    print(f"Warning: Using current directory: {Path.cwd()}")
    return Path.cwd()

project_root = get_project_root()
sys.path.insert(0, str(project_root))


import pickle
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objects as go
from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger, log_experiment
import pandas as pd
import json

# class TopicSegmentationTrainer:
#     def __init__(self, config):
#         self.config = config
#         self.topic_config = config['topic']
#         self.model_dir = Path(config['paths']['models']['topic'])
#         self.model_dir.mkdir(parents=True, exist_ok=True)
        
#         self.logger = setup_logger("topic_training")
        
#         # Initialize models
#         self.topic_model = None
#         self.embedding_model = None
        
#     def load_data(self):
#         """Load topic modeling data"""
#         dataset_path = Path(self.config['paths']['data']['datasets']) / "topic_dataset.pkl"
        
#         with open(dataset_path, 'rb') as f:
#             dataset = pickle.load(f)
        
#         self.logger.info(f"Loaded topic dataset with {len(dataset)} lectures")
        
#         # Prepare data for BERTopic
#         all_documents = []
#         self.lecture_mapping = []
        
#         for lecture_id, segments in dataset.items():
#             for segment in segments:
#                 all_documents.append(segment['text'])
#                 self.lecture_mapping.append({
#                     'lecture_id': lecture_id,
#                     'start': segment['start'],
#                     'end': segment['end'],
#                     'has_equation': segment['visual_cues']['has_equation'],
#                     'has_diagram': segment['visual_cues']['has_diagram']
#                 })
        
#         return all_documents


class TopicSegmentationTrainer:
    def __init__(self, config):
        self.config = config
        self.topic_config = config['topic']
        
        # --- Colab-Compatible Paths ---
        if 'COLAB_GPU' in os.environ or 'colab' in str(project_root).lower():
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            drive_root = '/content/drive/MyDrive'
            self.model_dir = Path(drive_root) / 'lecture_summarization' / 'models' / 'topic'
            print(f"Colab mode: Models will be saved to Drive: {self.model_dir}")
        else:
            self.model_dir = Path(config['paths']['models']['topic'])
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("topic_training")
        self.topic_model = None
        self.embedding_model = None
        
    def load_data(self):
        """Load topic modeling data with dynamic path resolution"""
        # Find the dataset
        possible_dirs = [
            project_root / 'data' / 'datasets',
            Path('data/datasets'),
            Path('/content/data/datasets'),
            Path('/content/drive/MyDrive/lecture_summarization/data/datasets')
        ]
        
        dataset_path = None
        for data_dir in possible_dirs:
            temp_path = data_dir / "topic_dataset.pkl"
            if temp_path.exists():
                dataset_path = temp_path
                break
        
        if dataset_path is None:
            raise FileNotFoundError("Topic dataset not found.")
        
        self.logger.info(f"Loading topic dataset from: {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        self.logger.info(f"Loaded topic dataset with {len(dataset)} lectures")
        
        # Prepare data for BERTopic
        all_documents = []
        self.lecture_mapping = []
        
        for lecture_id, segments in dataset.items():
            for segment in segments:
                all_documents.append(segment['text'])
                self.lecture_mapping.append({
                    'lecture_id': lecture_id,
                    'start': segment['start'],
                    'end': segment['end'],
                    'has_equation': segment['visual_cues']['has_equation'],
                    'has_diagram': segment['visual_cues']['has_diagram']
                })
        
        return all_documents
    def create_custom_stop_words(self):
        """Create lecture-specific stop words"""
        base_stop_words = [
            'okay', 'alright', 'uh', 'um', 'so', 'right', 'now',
            'today', 'lecture', 'chapter', 'section', 'slide',
            "let's", "we're", "we'll", 'going', 'talk', 'like',
            'know', 'want', 'really', 'actually'
        ]
        
        return base_stop_words
    
    def train(self, documents):
        """Train BERTopic model"""
        self.logger.info("Training BERTopic model...")
        
        # Step 1: Create embeddings
        self.logger.info("Creating document embeddings...")
        self.embedding_model = SentenceTransformer(self.topic_config['embedding_model'])
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # Step 2: Configure BERTopic components
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=self.config['project']['random_seed']
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.topic_config['min_topic_size'],
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            min_samples=1
        )
        
        vectorizer_model = CountVectorizer(
            stop_words=self.create_custom_stop_words(),
            ngram_range=tuple(self.topic_config['n_gram_range']),
            max_features=5000
        )
        
        # Step 3: Initialize and train BERTopic
        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=self.topic_config.get('nr_topics', None),  # Use None for automatic
            top_n_words=self.topic_config['top_n_words'],
            diversity=self.topic_config.get('diversity', 0.1),
            calculate_probabilities=True,
            verbose=True
        )
        
        topics, probabilities = self.topic_model.fit_transform(documents, embeddings)
        
        self.logger.info(f"Found {len(set(topics)) - 1} topics (excluding -1)")
        
        return topics, probabilities
    
    def detect_topic_boundaries(self, topics):
        """Detect topic boundaries in lectures"""
        boundaries = {}
        
        # Group by lecture
        lecture_groups = {}
        for idx, mapping in enumerate(self.lecture_mapping):
            lecture_id = mapping['lecture_id']
            if lecture_id not in lecture_groups:
                lecture_groups[lecture_id] = []
            lecture_groups[lecture_id].append((idx, mapping))
        
        # Detect boundaries for each lecture
        for lecture_id, segments in lecture_groups.items():
            lecture_boundaries = []
            segments = sorted(segments, key=lambda x: x[1]['start'])
            
            current_topic = topics[segments[0][0]]
            start_idx = 0
            
            for i in range(1, len(segments)):
                seg_idx, mapping = segments[i]
                
                if topics[seg_idx] != current_topic:
                    # Found a boundary
                    prev_seg = segments[i-1][1]
                    
                    boundary_info = {
                        'boundary_index': i,
                        'timestamp': (prev_seg['end'] + mapping['start']) / 2,
                        'previous_topic': int(current_topic),
                        'next_topic': int(topics[seg_idx]),
                        'confidence': self._calculate_boundary_confidence(
                            topics, [s[0] for s in segments], i
                        ),
                        'visual_context': {
                            'has_equation': mapping['has_equation'],
                            'has_diagram': mapping['has_diagram']
                        }
                    }
                    
                    lecture_boundaries.append(boundary_info)
                    current_topic = topics[seg_idx]
            
            boundaries[lecture_id] = lecture_boundaries
        
        return boundaries
    
    def _calculate_boundary_confidence(self, topics, segment_indices, boundary_idx):
        """Calculate confidence score for a boundary"""
        window_size = 3
        start = max(0, boundary_idx - window_size)
        end = min(len(segment_indices), boundary_idx + window_size)
        
        # Get topics in window
        window_topics = [topics[segment_indices[i]] for i in range(start, end)]
        
        # Calculate topic consistency before and after boundary
        before_topics = set(window_topics[:boundary_idx - start])
        after_topics = set(window_topics[boundary_idx - start:])
        
        if len(before_topics.union(after_topics)) > 0:
            overlap = len(before_topics.intersection(after_topics))
            confidence = 1 - (overlap / len(before_topics.union(after_topics)))
        else:
            confidence = 1.0
        
        return confidence
    
    def visualize_topics(self, topics, probabilities):
        """Create topic visualizations"""
        viz_dir = self.model_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Topic distribution
            topic_counts = pd.Series(topics).value_counts()
            fig1 = px.bar(
                x=topic_counts.index.astype(str),
                y=topic_counts.values,
                title="Topic Distribution",
                labels={'x': 'Topic ID', 'y': 'Number of Segments'}
            )
            fig1.write_html(str(viz_dir / "topic_distribution.html"))
            
            # 2. Hierarchical clustering
            hierarchical_topics = self.topic_model.hierarchical_topics(
                [self.lecture_mapping[i]['lecture_id'] for i in range(len(topics))]
            )
            fig2 = self.topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
            fig2.write_html(str(viz_dir / "topic_hierarchy.html"))
            
            # 3. Topic similarity
            fig3 = self.topic_model.visualize_heatmap()
            fig3.write_html(str(viz_dir / "topic_similarity.html"))
            
            # 4. Document clustering (simplified version)
            self.logger.info("Creating document clustering visualization...")
            
            self.logger.info(f"Visualizations saved to {viz_dir}")
        except Exception as e:
            self.logger.warning(f"Could not create some visualizations: {e}")
    
    def save_model(self, topics, boundaries):
        """Save model and results"""
        results_dir = self.model_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save BERTopic model
        self.topic_model.save(str(results_dir / "topic_model"))
        
        # Save embedding model
        self.embedding_model.save(str(results_dir / "embedding_model"))
        
        # Save results
        results = {
            'topics': topics.tolist() if hasattr(topics, 'tolist') else list(topics),
            'topic_info': self.topic_model.get_topic_info().to_dict('records'),
            'boundaries': boundaries,
            'lecture_mapping': self.lecture_mapping,
            'config': self.topic_config
        }
        
        with open(results_dir / "segmentation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save topic representations
        topic_representations = {}
        for topic_id in set(topics):
            if topic_id != -1:
                try:
                    topic_words = self.topic_model.get_topic(topic_id)
                    topic_representations[topic_id] = [
                        {'word': word, 'score': score} for word, score in topic_words
                    ]
                except:
                    continue
        
        with open(results_dir / "topic_representations.json", 'w') as f:
            json.dump(topic_representations, f, indent=2)
        
        self.logger.info(f"Model and results saved to {results_dir}")
        
        return results_dir

def main():
    config = config_loader.load_all()
    trainer = TopicSegmentationTrainer(config)
    
    try:
        documents = trainer.load_data()
        topics, probabilities = trainer.train(documents)
        boundaries = trainer.detect_topic_boundaries(topics)
        trainer.visualize_topics(topics, probabilities)
        results_dir = trainer.save_model(topics, boundaries)
        
        # Log experiment
        logger = setup_logger("experiment")
        unique_topics = set(topics)
        if -1 in unique_topics:
            unique_topics.remove(-1)
        
        topic_stats = {
            'total_topics': len(unique_topics),
            'outlier_segments': list(topics).count(-1),
            'avg_segments_per_topic': len(topics) / len(unique_topics) if len(unique_topics) > 0 else 0,
            'total_boundaries': sum(len(b) for b in boundaries.values())
        }
        
        experiment_results = {
            'model': 'BERTopic_Segmentation_Colab',
            'statistics': topic_stats,
            'boundaries_detected': topic_stats['total_boundaries'],
            'config': config['topic']
        }
        
        log_experiment(logger, config, experiment_results)
        
        print("\n" + "="*50)
        print("Topic Segmentation Complete!")
        print(f"Topics found: {topic_stats['total_topics']}")
        print(f"Total boundaries detected: {topic_stats['total_boundaries']}")
        print(f"Results saved to: {results_dir}")
        print("="*50)
        
    except Exception as e:
        print(f"Error during topic segmentation training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()