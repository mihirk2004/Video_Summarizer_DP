import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
from tqdm import tqdm
import yaml

class LectureDataLoader:
    """Loader for lecture annotation data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['paths']['data']['annotations'])
        self.dataset_dir = Path(config['paths']['data']['datasets'])
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize spaCy for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def load_all_annotations(self) -> List[Dict]:
        """Load all annotation files"""
        annotation_files = list(self.data_dir.glob("*.json"))
        print(f"Found {len(annotation_files)} annotation files")
        
        all_annotations = []
        for file_path in tqdm(annotation_files, desc="Loading annotations"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_annotations.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return all_annotations
    
    def extract_transcript_data(self, annotations: List[Dict]) -> pd.DataFrame:
        """Extract transcript data into DataFrame"""
        records = []
        
        for lecture in annotations:
            lecture_id = lecture.get("video_id", "unknown")
            
            # Process transcript segments
            transcript_segments = lecture.get("processing", {}).get("transcript", [])
            frames = lecture.get("processing", {}).get("frames", [])
            
            for segment in transcript_segments:
                # Find corresponding frames
                segment_frames = [
                    frame for frame in frames
                    if segment["start"] <= frame.get("timestamp", 0) <= segment["end"]
                ]
                
                # Extract visual information
                has_equation = any(
                    frame.get("detections", {}).get("equations")
                    for frame in segment_frames
                )
                has_diagram = any(
                    frame.get("detections", {}).get("diagrams")
                    for frame in segment_frames
                )
                has_gesture = any(
                    frame.get("detections", {}).get("gesture") != "none"
                    for frame in segment_frames
                )
                
                # Create record
                record = {
                    "lecture_id": lecture_id,
                    "segment_id": f"{lecture_id}_{segment['start']}_{segment['end']}",
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment["text"],
                    "speaker": segment.get("speaker", "unknown"),
                    "has_equation": has_equation,
                    "has_diagram": has_diagram,
                    "has_gesture": has_gesture,
                    "frame_count": len(segment_frames),
                    "duration": segment["end"] - segment["start"],
                    "word_count": len(segment["text"].split()),
                    "sentence_count": len(list(self.nlp(segment["text"]).sents))
                }
                
                records.append(record)
        
        df = pd.DataFrame(records)
        print(f"Extracted {len(df)} transcript segments")
        print(f"Statistics: {df['has_equation'].sum()} with equations, "
              f"{df['has_diagram'].sum()} with diagrams")
        
        return df
    
    def create_ner_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create dataset for NER training"""
        # Filter segments with visual content for NER training
        ner_df = df[df['has_equation'] | df['has_diagram']].copy()
        
        # Generate training data format for spaCy
        train_data = []
        
        for _, row in tqdm(ner_df.iterrows(), desc="Creating NER data", total=len(ner_df)):
            text = row['text']
            doc = self.nlp(text)
            
            entities = self._extract_stem_entities(doc, row)
            
            if entities:
                train_data.append({
                    "text": text,
                    "entities": entities,
                    "metadata": {
                        "lecture_id": row['lecture_id'],
                        "has_equation": row['has_equation'],
                        "has_diagram": row['has_diagram'],
                        "timestamp": row['start_time']
                    }
                })
        
        # Split dataset
        train_data, test_data = train_test_split(
            train_data,
            test_size=self.config['data']['test_ratio'],
            random_state=self.config['project']['random_seed']
        )
        
        train_data, val_data = train_test_split(
            train_data,
            test_size=self.config['data']['val_ratio'] / 
                     (1 - self.config['data']['test_ratio']),
            random_state=self.config['project']['random_seed']
        )
        
        dataset = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "stats": {
                "total": len(train_data) + len(val_data) + len(test_data),
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data)
            }
        }
        
        # Save dataset
        dataset_path = self.dataset_dir / "ner_dataset.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"NER dataset saved to {dataset_path}")
        return dataset
    
    def _extract_stem_entities(self, doc, row) -> List[Tuple[int, int, str]]:
        """Extract STEM entities using rule-based approach"""
        entities = []
        
        # Rule 1: Mathematical expressions
        math_keywords = ['equation', 'formula', 'derive', 'calculate', 'solve']
        for token in doc:
            if token.text.lower() in math_keywords:
                # Look for equation in context
                for i in range(max(0, token.i-3), min(len(doc), token.i+4)):
                    if doc[i].ent_type_ == '' and doc[i].is_alpha:
                        entities.append((doc[i].idx, doc[i].idx + len(doc[i]), "EQUATION"))
        
        # Rule 2: Scientific concepts
        concept_keywords = ['theory', 'principle', 'law', 'concept', 'model']
        for token in doc:
            if token.text.lower() in concept_keywords:
                # Look for concept name
                for i in range(max(0, token.i-2), min(len(doc), token.i+3)):
                    if doc[i].ent_type_ == '' and doc[i].is_alpha and doc[i].is_title:
                        entities.append((doc[i].idx, doc[i].idx + len(doc[i]), "CONCEPT"))
        
        return list(set(entities))  # Remove duplicates
    
    def create_topic_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create dataset for topic modeling"""
        # Group by lecture
        lectures = {}
        
        for lecture_id in df['lecture_id'].unique():
            lecture_df = df[df['lecture_id'] == lecture_id].sort_values('start_time')
            
            segments = []
            for _, row in lecture_df.iterrows():
                segments.append({
                    "text": row['text'],
                    "start": row['start_time'],
                    "end": row['end_time'],
                    "visual_cues": {
                        "has_equation": row['has_equation'],
                        "has_diagram": row['has_diagram']
                    }
                })
            
            lectures[lecture_id] = segments
        
        # Save dataset
        dataset_path = self.dataset_dir / "topic_dataset.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(lectures, f)
        
        print(f"Topic dataset saved to {dataset_path}")
        return lectures
    
    def create_similarity_dataset(self, df: pd.DataFrame) -> List[Dict]:
        """Create dataset for similarity training"""
        similarity_pairs = []
        
        for lecture_id in df['lecture_id'].unique():
            lecture_df = df[df['lecture_id'] == lecture_id].sort_values('start_time')
            segments = lecture_df['text'].tolist()
            
            # Create positive pairs (adjacent segments)
            for i in range(len(segments) - 1):
                similarity_pairs.append({
                    "text1": segments[i],
                    "text2": segments[i+1],
                    "label": 1.0,  # Positive pair
                    "lecture_id": lecture_id,
                    "time_gap": lecture_df.iloc[i+1]['start_time'] - lecture_df.iloc[i]['end_time']
                })
            
            # Create negative pairs (distant segments)
            if len(segments) > 3:
                for i in range(len(segments) - 3):
                    similarity_pairs.append({
                        "text1": segments[i],
                        "text2": segments[i+3],
                        "label": 0.0,  # Negative pair
                        "lecture_id": lecture_id,
                        "time_gap": lecture_df.iloc[i+3]['start_time'] - lecture_df.iloc[i]['end_time']
                    })
        
        # Split dataset
        train_pairs, test_pairs = train_test_split(
            similarity_pairs,
            test_size=self.config['data']['test_ratio'],
            random_state=self.config['project']['random_seed']
        )
        
        train_pairs, val_pairs = train_test_split(
            train_pairs,
            test_size=self.config['data']['val_ratio'] / 
                     (1 - self.config['data']['test_ratio']),
            random_state=self.config['project']['random_seed']
        )
        
        dataset = {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs
        }
        
        # Save dataset
        dataset_path = self.dataset_dir / "similarity_dataset.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Similarity dataset saved to {dataset_path}")
        return dataset
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate statistics about the dataset"""
        stats = {
            "total_lectures": df['lecture_id'].nunique(),
            "total_segments": len(df),
            "avg_segments_per_lecture": len(df) / df['lecture_id'].nunique(),
            "avg_segment_duration": df['duration'].mean(),
            "avg_word_count": df['word_count'].mean(),
            "segments_with_equations": df['has_equation'].sum(),
            "segments_with_diagrams": df['has_diagram'].sum(),
            "segments_with_gestures": df['has_gesture'].sum(),
            "equation_percentage": (df['has_equation'].sum() / len(df)) * 100,
            "diagram_percentage": (df['has_diagram'].sum() / len(df)) * 100,
            "word_distribution": {
                "min": df['word_count'].min(),
                "max": df['word_count'].max(),
                "mean": df['word_count'].mean(),
                "std": df['word_count'].std()
            }
        }
        
        return stats