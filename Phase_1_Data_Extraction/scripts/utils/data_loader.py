import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import re

class LectureDataLoader:
    """Loader for lecture annotation data using NLTK for text processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['paths']['data']['annotations'])
        self.dataset_dir = Path(config['paths']['data']['datasets'])
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLTK and download required resources
        self._setup_nltk()
        
        # Define concept categories
        self.concept_categories = {
            'EQUATION': ['Equations', 'Math', 'Formula', 'Algebra', 'Calculus'],
            'DIAGRAM': ['Diagrams', 'Graphs_Charts', 'Graph/Chart', 'Chart', 'Graph'],
            'CODE': ['Code', 'Programming', 'Algorithm'],
            'REACTION': ['Molecular_Reactions', 'Chemistry', 'Reaction'],
            'GESTURE': ['Instructor_Pointing', 'Instructor_Writing', 'Instructor'],
            'QNA': ['Questions', 'Answers']
        }
    
    def _setup_nltk(self):
        """Download required NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("Downloading NLTK resources...")
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt_tab', quiet=True)
    
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
        """Extract transcript data into DataFrame using NLTK for text processing"""
        records = []
        
        for lecture_idx, lecture in enumerate(annotations):
            lecture_id = lecture.get("video_id", f"lecture_{lecture_idx:03d}")
            
            # Process transcript data - it's a dictionary with 'text' and 'segments'
            processing = lecture.get("processing", {})
            transcript_data = processing.get("transcript", {})
            
            # Get the full transcript text
            full_text = transcript_data.get("text", "")
            
            # Check if we have segments or just full text
            segments = transcript_data.get("segments", [])
            
            # Get annotation frames
            annotations_data = lecture.get("annotations", {})
            annotation_frames = annotations_data.get("frames", [])
            
            # Get auto_annotations for additional context
            auto_annotations = annotations_data.get("auto_annotations", {})
            detected_concepts = auto_annotations.get("detected_concepts", [])
            
            # If segments array is empty, create segments from full text
            if not segments or len(segments) == 0:
                if full_text:
                    # Split full text into sentences and create segments
                    try:
                        sentences = sent_tokenize(full_text)
                    except:
                        sentences = full_text.split('. ')
                    
                    # Create segments from sentences (group 2-3 sentences together)
                    segment_size = 3  # Sentences per segment
                    for i in range(0, len(sentences), segment_size):
                        segment_text = ' '.join(sentences[i:i+segment_size])
                        if segment_text.strip():
                            segments.append({
                                "start": i * 5,  # Approximate: 5 seconds per sentence
                                "end": (i + segment_size) * 5,
                                "text": segment_text,
                                "speaker": "unknown"
                            })
                else:
                    # Skip lectures without text
                    continue
            
            # Process each segment
            for segment_idx, segment in enumerate(segments):
                # Process text with NLTK
                text = segment.get("text", "")
                
                # If text is empty but we have full text, use full text (for first segment)
                if not text and full_text and segment_idx == 0:
                    text = full_text
                
                # Skip empty segments
                if not text or not text.strip():
                    continue
                
                # Tokenize text
                try:
                    sentences = sent_tokenize(text) if text else []
                    words = word_tokenize(text) if text else []
                except:
                    sentences = [text] if text else []
                    words = text.split() if text else []
                
                # Get segment time boundaries
                segment_start = segment.get("start", segment_idx * 30)  # 30 seconds per segment
                segment_end = segment.get("end", segment_start + 30)
                
                # Extract visual concepts from annotation frames
                visual_info = self._extract_visual_concepts_from_frames(
                    annotation_frames, segment_start, segment_end
                )
                
                # Extract additional info from notes
                notes_info = self._extract_info_from_notes(
                    annotation_frames, segment_start, segment_end
                )
                
                # Combine with detected concepts from auto_annotations
                if detected_concepts:
                    for concept in detected_concepts:
                        concept_lower = str(concept).lower()
                        if any(math_term in concept_lower for math_term in ['equation', 'formula', 'math']):
                            visual_info['has_equation'] = True
                        if any(diagram_term in concept_lower for diagram_term in ['diagram', 'graph', 'chart']):
                            visual_info['has_diagram'] = True
                
                # Create record
                record = {
                    "lecture_id": lecture_id,
                    "segment_id": f"{lecture_id}_seg_{segment_idx:03d}",
                    "start_time": segment_start,
                    "end_time": segment_end,
                    "text": text,
                    "speaker": segment.get("speaker", "unknown"),
                    "has_equation": visual_info['has_equation'],
                    "has_diagram": visual_info['has_diagram'],
                    "has_code": visual_info['has_code'],
                    "has_reaction": visual_info['has_reaction'],
                    "has_gesture": visual_info['has_gesture'],
                    "has_question": visual_info['has_question'],
                    "has_answer": visual_info['has_answer'],
                    "notes_summary": notes_info['summary'],
                    "frame_count": visual_info['frame_count'],
                    "duration": segment_end - segment_start,
                    "word_count": len(words),
                    "sentence_count": len(sentences)
                }
                
                records.append(record)
        
        df = pd.DataFrame(records)
        print(f"Extracted {len(df)} transcript segments")
        
        if len(df) > 0:
            print(f"Statistics:")
            print(f"  Segments with equations: {df['has_equation'].sum()}")
            print(f"  Segments with diagrams: {df['has_diagram'].sum()}")
            print(f"  Segments with code: {df['has_code'].sum()}")
            print(f"  Segments with reactions: {df['has_reaction'].sum()}")
            print(f"  Segments with gestures: {df['has_gesture'].sum()}")
            print(f"  Segments with questions: {df['has_question'].sum()}")
            print(f"  Segments with answers: {df['has_answer'].sum()}")
            print(f"  Total words: {df['word_count'].sum()}")
            print(f"  Average words per segment: {df['word_count'].mean():.2f}")
            print(f"  Unique lectures: {df['lecture_id'].nunique()}")
        else:
            print("Warning: No transcript segments extracted!")
        
        return df
    
    def _extract_visual_concepts_from_frames(self, frames: List[Dict], 
                                           segment_start: float, segment_end: float) -> Dict:
        """Extract visual concepts from annotation frames"""
        visual_info = {
            'has_equation': False,
            'has_diagram': False,
            'has_code': False,
            'has_reaction': False,
            'has_gesture': False,
            'has_question': False,
            'has_answer': False,
            'frame_count': 0
        }
        
        for frame in frames:
            try:
                timestamp = frame.get("timestamp", 0)
                
                # Check if frame is within segment
                if segment_start <= timestamp <= segment_end:
                    visual_info['frame_count'] += 1
                    
                    # Extract concepts from frame
                    concepts = frame.get("concepts", [])
                    
                    # Check each concept
                    for concept in concepts:
                        if isinstance(concept, str):
                            concept_lower = concept.lower()
                            
                            # Check for equations
                            if any(term in concept_lower for term in 
                                   ['equation', 'math', 'formula', 'algebra', 'calculus']):
                                visual_info['has_equation'] = True
                            
                            # Check for diagrams/graphs/charts
                            if any(term in concept_lower for term in 
                                   ['diagram', 'graph', 'chart', 'plot', 'figure']):
                                visual_info['has_diagram'] = True
                            
                            # Check for code
                            if any(term in concept_lower for term in 
                                   ['code', 'programming', 'algorithm', 'function']):
                                visual_info['has_code'] = True
                            
                            # Check for reactions
                            if any(term in concept_lower for term in 
                                   ['reaction', 'molecular', 'chemistry', 'chemical']):
                                visual_info['has_reaction'] = True
                            
                            # Check for gestures
                            if any(term in concept_lower for term in 
                                   ['instructor', 'pointing', 'writing', 'gesture']):
                                visual_info['has_gesture'] = True
                            
                            # Check for questions
                            if any(term in concept_lower for term in 
                                   ['question', 'ask', 'problem']):
                                visual_info['has_question'] = True
                            
                            # Check for answers
                            if any(term in concept_lower for term in 
                                   ['answer', 'solution', 'explanation']):
                                visual_info['has_answer'] = True
                    
                    # Also check quality field for additional info
                    quality = frame.get("quality", "").lower()
                    if quality in ['good', 'excellent'] and visual_info['frame_count'] == 1:
                        # High quality frames might have clearer visual content
                        pass
                        
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                continue
        
        return visual_info
    
    def _extract_entities_from_text(self, text: str) -> List[Tuple[int, int, str]]:
        """Extract entities from text using more aggressive patterns"""
        entities = []
        
        # Pattern 1: Capitalized phrases (potential concepts/theorems)
        pattern1 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        for match in re.finditer(pattern1, text):
            phrase = match.group(1)
            # Skip common words
            if len(phrase.split()) <= 3 and phrase not in ['The', 'This', 'That', 'These', 'Those']:
                entities.append((match.start(1), match.end(1), "CONCEPT"))
        
        # Pattern 2: Mathematical patterns
        math_patterns = [
            (r'\b([A-Z][a-z]+\'s\s+[A-Za-z]+)\b', "THEOREM"),  # Newton's Law
            (r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', "CONCEPT"),  # Quantum Mechanics
            (r'\b([A-Z][a-z]+-[A-Z][a-z]+)\b', "CONCEPT"),  # Proof-of-Concept
        ]
        
        for pattern, label in math_patterns:
            for match in re.finditer(pattern, text):
                entities.append((match.start(1), match.end(1), label))
        
        return list(set(entities))
    
    def _extract_info_from_notes(self, frames: List[Dict], 
                               segment_start: float, segment_end: float) -> Dict:
        """Extract information from frame notes"""
        notes_summary = []
        
        for frame in frames:
            try:
                timestamp = frame.get("timestamp", 0)
                
                # Check if frame is within segment
                if segment_start <= timestamp <= segment_end:
                    notes = frame.get("notes", "")
                    if notes and notes.strip():
                        # Clean and add note
                        clean_note = notes.strip()
                        if clean_note and len(clean_note) > 2:  # Avoid very short notes
                            notes_summary.append(clean_note)
            except:
                continue
        
        # Create a summary of notes
        if notes_summary:
            # Get unique notes and join them
            unique_notes = list(set(notes_summary))
            summary = "; ".join(unique_notes[:3])  # Take up to 3 notes
            if len(unique_notes) > 3:
                summary += f" ... (+{len(unique_notes)-3} more)"
        else:
            summary = ""
        
        return {'summary': summary, 'count': len(notes_summary)}
    
    def create_ner_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create dataset for NER training using NLTK"""
        print(f"Creating NER dataset from {len(df)} segments")
        
        # Use ALL segments for NER training (not just those with visual content)
        # This will give us more training data
        ner_df = df.copy()
        
        # But prioritize segments with visual content by oversampling them
        visual_columns = ['has_equation', 'has_diagram', 'has_code', 'has_reaction']
        has_visual = df[visual_columns].any(axis=1)
        visual_df = df[has_visual]
        
        print(f"Found {len(visual_df)} segments with visual content (will be prioritized)")
        
        
        # Generate training data format for NER
        train_data = []

        # First, process ALL visual segments
        visual_data = []
        for _, row in tqdm(visual_df.iterrows(), desc="Processing visual segments", total=len(visual_df)):
            text = row['text']
            if not text or not text.strip():
                continue
                
            entities = self._extract_stem_entities_with_context(text, row)
            if entities:
                visual_data.append({
                    "text": text,
                    "entities": entities,
                    "metadata": {
                        "lecture_id": row['lecture_id'],
                        "has_equation": bool(row['has_equation']),
                        "has_diagram": bool(row['has_diagram']),
                        "has_code": bool(row['has_code']),
                        "has_reaction": bool(row['has_reaction']),
                        "notes": row.get('notes_summary', ''),
                        "timestamp": float(row['start_time'])
                    }
                })
        
        # Then, sample some non-visual segments for diversity
        non_visual_df = df[~has_visual]
        non_visual_sample = non_visual_df.sample(
            min(1000, len(non_visual_df)),  # Sample up to 1000 non-visual segments
            random_state=self.config['project']['random_seed']
        )
        
        non_visual_data = []
        
        for _, row in tqdm(non_visual_sample.iterrows(), desc="Processing non-visual segments", total=len(non_visual_sample)):
            text = row['text']
            if not text or not text.strip():
                continue
                
            # Use a more aggressive entity extraction for non-visual segments
            entities = self._extract_entities_from_text(text)
            if entities:
                non_visual_data.append({
                    "text": text,
                    "entities": entities,
                    "metadata": {
                        "lecture_id": row['lecture_id'],
                        "has_equation": False,
                        "has_diagram": False,
                        "has_code": False,
                        "has_reaction": False,
                        "notes": row.get('notes_summary', ''),
                        "timestamp": float(row['start_time'])
                    }
                })
        # Combine both datasets
        train_data = visual_data + non_visual_data

        print(f"Total NER samples: {len(train_data)} (visual: {len(visual_data)}, non-visual: {len(non_visual_data)})")

        for _, row in tqdm(ner_df.iterrows(), desc="Creating NER data", total=len(ner_df)):
            text = row['text']
            
            # Skip empty text
            if not text or not text.strip():
                continue
                
            # Extract entities using NLTK-based method with visual context
            entities = self._extract_stem_entities_with_context(text, row)
            
            if entities:
                train_data.append({
                    "text": text,
                    "entities": entities,
                    "metadata": {
                        "lecture_id": row['lecture_id'],
                        "has_equation": bool(row['has_equation']),
                        "has_diagram": bool(row['has_diagram']),
                        "has_code": bool(row['has_code']),
                        "has_reaction": bool(row['has_reaction']),
                        "notes": row.get('notes_summary', ''),
                        "timestamp": float(row['start_time'])
                    }
                })
        
        # Check if we have enough data
        if len(train_data) == 0:
            print("Warning: No NER training data created!")
            return {
                "train": [],
                "val": [],
                "test": [],
                "stats": {
                    "total": 0,
                    "train": 0,
                    "val": 0,
                    "test": 0
                }
            }
        
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
    
    def _extract_stem_entities_with_context(self, text: str, row: pd.Series) -> List[Tuple[int, int, str]]:
        """Extract STEM entities using NLTK and visual context"""
        entities = []
        
        # Skip empty text
        if not text or not text.strip():
            return entities
        
        try:
            # Tokenize and get word positions
            tokens = word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            # Get character positions for tokens
            char_pos = 0
            token_positions = []
            
            for token in tokens:
                idx = text.find(token, char_pos)
                if idx != -1:
                    token_positions.append((idx, idx + len(token)))
                    char_pos = idx + len(token)
                else:
                    token_positions.append((char_pos, char_pos + len(token)))
                    char_pos += len(token) + 1
            
            # Use visual context to guide entity extraction
            if row['has_equation']:
                # Look for mathematical entities
                entities.extend(self._extract_mathematical_entities(tokens, pos_tags, token_positions))
            
            if row['has_diagram']:
                # Look for diagram-related entities
                entities.extend(self._extract_diagram_entities(tokens, pos_tags, token_positions))
            
            if row['has_code']:
                # Look for programming entities
                entities.extend(self._extract_code_entities(tokens, pos_tags, token_positions))
            
            if row['has_reaction']:
                # Look for chemical entities
                entities.extend(self._extract_chemical_entities(tokens, pos_tags, token_positions))
            
            # Always look for general STEM concepts
            entities.extend(self._extract_general_stem_entities(tokens, pos_tags, token_positions))
            
        except Exception as e:
            print(f"Error extracting entities from text: {e}")
            return []
        
        return list(set(entities))
    
    def _extract_mathematical_entities(self, tokens, pos_tags, token_positions):
        """Extract mathematical entities"""
        entities = []
        math_keywords = ['equation', 'formula', 'theorem', 'lemma', 'corollary', 
                        'integral', 'derivative', 'matrix', 'vector', 'tensor',
                        'function', 'variable', 'constant', 'parameter']
        
        for i, (token, pos) in enumerate(pos_tags):
            token_lower = token.lower()
            
            if token_lower in math_keywords:
                # Look for mathematical terms in context
                context_window = 3
                start_idx = max(0, i - context_window)
                end_idx = min(len(tokens), i + context_window + 1)
                
                for j in range(start_idx, end_idx):
                    if j != i and j < len(token_positions):
                        context_token = tokens[j]
                        # Mathematical entities often have specific patterns
                        if (context_token[0].isupper() or 
                            re.match(r'^[A-Z][a-z]+$', context_token) or
                            re.match(r'^[a-z]+_[0-9]+$', context_token) or
                            re.match(r'^[A-Za-z]+\d+$', context_token)):
                            start, end = token_positions[j]
                            entities.append((start, end, "MATH_TERM"))
        
        # Look for equation numbers
        equation_patterns = [
            (r'(Equation|Eq\.?)\s+([0-9]+[a-z]*)', 'EQUATION'),
            (r'(Theorem|Lemma|Corollary)\s+([0-9]+[a-z]*)', 'THEOREM'),
            (r'(Figure|Fig\.?)\s+([0-9]+[a-z]*)', 'FIGURE'),
        ]
        
        text = ' '.join(tokens)
        for pattern, label in equation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.lastindex >= 2:
                    start = match.start(2)
                    end = match.end(2)
                    entities.append((start, end, label))
        
        return entities
    
    def _extract_diagram_entities(self, tokens, pos_tags, token_positions):
        """Extract diagram-related entities"""
        entities = []
        diagram_keywords = ['diagram', 'graph', 'chart', 'figure', 'plot',
                          'axis', 'coordinate', 'point', 'line', 'curve']
        
        for i, (token, pos) in enumerate(pos_tags):
            token_lower = token.lower()
            
            if token_lower in diagram_keywords:
                # Look for diagram elements in context
                context_window = 2
                start_idx = max(0, i - context_window)
                end_idx = min(len(tokens), i + context_window + 1)
                
                for j in range(start_idx, end_idx):
                    if j != i and j < len(token_positions):
                        context_token = tokens[j]
                        if (context_token[0].isupper() or 
                            re.match(r'^[A-Z][a-z]+$', context_token)):
                            start, end = token_positions[j]
                            entities.append((start, end, "DIAGRAM_ELEMENT"))
        
        return entities
    
    def _extract_code_entities(self, tokens, pos_tags, token_positions):
        """Extract programming/code entities"""
        entities = []
        code_keywords = ['function', 'method', 'class', 'object', 'variable',
                        'loop', 'array', 'list', 'dictionary', 'algorithm']
        
        for i, (token, pos) in enumerate(pos_tags):
            token_lower = token.lower()
            
            if token_lower in code_keywords:
                # Look for code elements in context
                context_window = 2
                start_idx = max(0, i - context_window)
                end_idx = min(len(tokens), i + context_window + 1)
                
                for j in range(start_idx, end_idx):
                    if j != i and j < len(token_positions):
                        context_token = tokens[j]
                        # Programming terms often have specific patterns
                        if (re.match(r'^[a-z]+[A-Z][a-z]+$', context_token) or  # camelCase
                            re.match(r'^[a-z]+_[a-z]+$', context_token) or      # snake_case
                            context_token in ['if', 'else', 'for', 'while', 'return']):
                            start, end = token_positions[j]
                            entities.append((start, end, "CODE_TERM"))
        
        return entities
    
    def _extract_chemical_entities(self, tokens, pos_tags, token_positions):
        """Extract chemical entities"""
        entities = []
        chem_keywords = ['reaction', 'molecule', 'atom', 'bond', 'chemical',
                        'compound', 'element', 'solution', 'mixture']
        
        for i, (token, pos) in enumerate(pos_tags):
            token_lower = token.lower()
            
            if token_lower in chem_keywords:
                # Look for chemical terms in context
                context_window = 3
                start_idx = max(0, i - context_window)
                end_idx = min(len(tokens), i + context_window + 1)
                
                for j in range(start_idx, end_idx):
                    if j != i and j < len(token_positions):
                        context_token = tokens[j]
                        # Chemical formulas often have numbers or specific patterns
                        if (re.match(r'^[A-Z][a-z]?\d*$', context_token) or  # H2O, NaCl
                            re.match(r'^[A-Z][a-z]+[0-9]*$', context_token)):  # Methane, Ethanol
                            start, end = token_positions[j]
                            entities.append((start, end, "CHEMICAL"))
        
        return entities
    
    def _extract_general_stem_entities(self, tokens, pos_tags, token_positions):
        """Extract general STEM entities"""
        entities = []
        stem_keywords = ['theory', 'principle', 'law', 'concept', 'model',
                        'hypothesis', 'experiment', 'observation', 'analysis']
        
        for i, (token, pos) in enumerate(pos_tags):
            token_lower = token.lower()
            
            if token_lower in stem_keywords:
                # Look for STEM concepts in context
                context_window = 2
                start_idx = max(0, i - context_window)
                end_idx = min(len(tokens), i + context_window + 1)
                
                for j in range(start_idx, end_idx):
                    if j != i and j < len(token_positions):
                        context_token = tokens[j]
                        if (pos_tags[j][1] in ['NNP', 'NNPS'] or  # Proper nouns
                            (context_token[0].isupper() and len(context_token) > 1)):
                            start, end = token_positions[j]
                            entities.append((start, end, "STEM_CONCEPT"))
        
        return entities
    
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
                    "start": float(row['start_time']),
                    "end": float(row['end_time']),
                    "visual_cues": {
                        "has_equation": bool(row['has_equation']),
                        "has_diagram": bool(row['has_diagram']),
                        "has_code": bool(row['has_code']),
                        "has_reaction": bool(row['has_reaction']),
                        "has_gesture": bool(row['has_gesture'])
                    },
                    "notes": row.get('notes_summary', '')
                })
            
            lectures[lecture_id] = segments
        
        # Save dataset
        dataset_path = self.dataset_dir / "topic_dataset.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(lectures, f)
        
        print(f"Topic dataset saved to {dataset_path}")
        print(f"Created topic dataset for {len(lectures)} lectures with total {sum(len(s) for s in lectures.values())} segments")
        return lectures
    
    def create_similarity_dataset(self, df: pd.DataFrame) -> List[Dict]:
        """Create dataset for similarity training with visual context"""
        similarity_pairs = []
        
        for lecture_id in df['lecture_id'].unique():
            lecture_df = df[df['lecture_id'] == lecture_id].sort_values('start_time')
            
            # Skip lectures with too few segments
            if len(lecture_df) < 2:
                continue
            
            # Create pairs with visual context similarity
            segments = lecture_df.to_dict('records')
            
            for i in range(len(segments)):
                for j in range(i + 1, min(i + 4, len(segments))):  # Look at nearby segments
                    seg1 = segments[i]
                    seg2 = segments[j]
                    
                    # Calculate visual similarity score
                    visual_similarity = self._calculate_visual_similarity(seg1, seg2)
                    
                    # Time gap
                    time_gap = seg2['start_time'] - seg1['end_time']
                    
                    # Label: 1.0 for adjacent segments with similar visual content, 0.0 otherwise
                    label = 1.0 if (j == i + 1 and visual_similarity > 0.5) else 0.0
                    
                    similarity_pairs.append({
                        "text1": seg1['text'],
                        "text2": seg2['text'],
                        "label": label,
                        "lecture_id": lecture_id,
                        "time_gap": float(time_gap),
                        "visual_similarity": float(visual_similarity),
                        "visual_context1": {
                            "has_equation": bool(seg1['has_equation']),
                            "has_diagram": bool(seg1['has_diagram']),
                            "has_code": bool(seg1['has_code'])
                        },
                        "visual_context2": {
                            "has_equation": bool(seg2['has_equation']),
                            "has_diagram": bool(seg2['has_diagram']),
                            "has_code": bool(seg2['has_code'])
                        }
                    })
        
        # Check if we have enough data
        if len(similarity_pairs) == 0:
            print("Warning: No similarity pairs created!")
            return {
                "train": [],
                "val": [],
                "test": []
            }
        
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
        print(f"Created {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test pairs")
        return dataset
    
    def _calculate_visual_similarity(self, seg1: Dict, seg2: Dict) -> float:
        """Calculate similarity based on visual content"""
        visual_features1 = [
            seg1['has_equation'],
            seg1['has_diagram'],
            seg1['has_code'],
            seg1['has_reaction'],
            seg1['has_gesture']
        ]
        
        visual_features2 = [
            seg2['has_equation'],
            seg2['has_diagram'],
            seg2['has_code'],
            seg2['has_reaction'],
            seg2['has_gesture']
        ]
        
        # Calculate Jaccard similarity
        intersection = sum(1 for f1, f2 in zip(visual_features1, visual_features2) if f1 and f2)
        union = sum(1 for f1, f2 in zip(visual_features1, visual_features2) if f1 or f2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate statistics about the dataset"""
        if len(df) == 0:
            return {"error": "No data extracted"}
        
        # Convert all values to Python native types
        stats = {
            "total_lectures": int(df['lecture_id'].nunique()),
            "total_segments": int(len(df)),
            "avg_segments_per_lecture": float(df['lecture_id'].value_counts().mean()),
            "avg_segment_duration": float(df['duration'].mean()),
            "avg_word_count": float(df['word_count'].mean()),
            "avg_sentence_count": float(df['sentence_count'].mean()),
            "segments_with_equations": int(df['has_equation'].sum()),
            "segments_with_diagrams": int(df['has_diagram'].sum()),
            "segments_with_code": int(df['has_code'].sum()),
            "segments_with_reactions": int(df['has_reaction'].sum()),
            "segments_with_gestures": int(df['has_gesture'].sum()),
            "segments_with_questions": int(df['has_question'].sum()),
            "segments_with_answers": int(df['has_answer'].sum()),
            "segments_with_notes": int(df[df['notes_summary'] != ''].shape[0]),
            "equation_percentage": float((df['has_equation'].sum() / len(df)) * 100) if len(df) > 0 else 0.0,
            "diagram_percentage": float((df['has_diagram'].sum() / len(df)) * 100) if len(df) > 0 else 0.0,
            "code_percentage": float((df['has_code'].sum() / len(df)) * 100) if len(df) > 0 else 0.0,
            "word_distribution": {
                "min": int(df['word_count'].min()),
                "max": int(df['word_count'].max()),
                "mean": float(df['word_count'].mean()),
                "std": float(df['word_count'].std())
            },
            "sentence_distribution": {
                "min": int(df['sentence_count'].min()),
                "max": int(df['sentence_count'].max()),
                "mean": float(df['sentence_count'].mean()),
                "std": float(df['sentence_count'].std())
            }
        }
        
        return stats