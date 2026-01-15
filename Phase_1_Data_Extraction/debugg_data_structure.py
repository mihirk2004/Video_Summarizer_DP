#!/usr/bin/env python3
"""
Debug script to check annotation data structure
"""
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader

def main():
    config = config_loader.load_all()
    data_dir = Path(config['paths']['data']['annotations'])
    
    # Load first annotation file
    annotation_files = list(data_dir.glob("*.json"))
    if not annotation_files:
        print("No annotation files found!")
        return
    
    print(f"Found {len(annotation_files)} annotation files")
    
    # Check first 3 files
    for i in range(min(3, len(annotation_files))):
        file_path = annotation_files[i]
        print(f"\n{'='*60}")
        print(f"File {i+1}: {file_path.name}")
        print('='*60)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check structure
        print(f"Keys in file: {list(data.keys())}")
        
        if 'processing' in data:
            processing = data['processing']
            print(f"Keys in processing: {list(processing.keys())}")
            
            if 'transcript' in processing:
                transcript = processing['transcript']
                print(f"Transcript type: {type(transcript)}")
                print(f"Transcript length: {len(transcript) if isinstance(transcript, list) else 'N/A'}")
                
                if transcript and isinstance(transcript, list) and len(transcript) > 0:
                    print(f"First transcript segment keys: {list(transcript[0].keys())}")
                    print(f"Sample text: {transcript[0].get('text', 'No text')[:100]}...")
                else:
                    print("Transcript is empty or not a list!")
            
            if 'frames' in processing:
                frames = processing['frames']
                print(f"Frames type: {type(frames)}")
                print(f"Frames length: {len(frames) if isinstance(frames, list) else 'N/A'}")
                
                if frames and isinstance(frames, list) and len(frames) > 0:
                    print(f"Type of first frame: {type(frames[0])}")
                    if isinstance(frames[0], dict):
                        print(f"First frame keys: {list(frames[0].keys())}")
                        if 'detections' in frames[0]:
                            detections = frames[0]['detections']
                            if isinstance(detections, dict):
                                print(f"Detection keys: {list(detections.keys())}")
                            else:
                                print(f"Detections type: {type(detections)}")
                    elif isinstance(frames[0], (int, float, str)):
                        print(f"First frame value: {frames[0]}")
                else:
                    print("Frames are empty or not a list!")
        
        # Check if there are annotations
        if 'annotations' in data:
            annotations = data['annotations']
            print(f"Annotations type: {type(annotations)}")
            if isinstance(annotations, list):
                print(f"Annotations count: {len(annotations)}")
        
        print(f"{'='*60}")

if __name__ == "__main__":
    main()