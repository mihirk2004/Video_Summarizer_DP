#!/usr/bin/env python3
"""
Check what concepts are in the annotation files
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
    
    # Collect all unique concepts from first 5 files
    all_concepts = set()
    
    for i in range(min(5, len(annotation_files))):
        file_path = annotation_files[i]
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'processing' in data:
            processing = data['processing']
            frames = processing.get('frames', [])
            
            for frame in frames:
                if isinstance(frame, dict):
                    concepts = frame.get('concepts', [])
                    if isinstance(concepts, list):
                        for concept in concepts:
                            if concept:  # Skip empty strings
                                all_concepts.add(concept)
    
    print("\nUnique concepts found:")
    print("-" * 40)
    for concept in sorted(all_concepts):
        print(concept)
    print("-" * 40)
    print(f"Total unique concepts: {len(all_concepts)}")

if __name__ == "__main__":
    main()