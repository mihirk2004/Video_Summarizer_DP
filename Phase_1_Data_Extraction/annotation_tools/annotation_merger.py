# annotation_tools/annotation_merger.py
import json
import os
from pathlib import Path

def merge_human_annotations(original_json_path: str, 
                           annotation_json_path: str,
                           output_json_path: str = None):
    """
    Merge human annotations back into the original processed JSON
    
    Args:
        original_json_path: Path to the auto-processed JSON
        annotation_json_path: Path to human annotations (from Label Studio or GUI)
        output_json_path: Output path (default: adds '_final' suffix)
    """
    
    # Load original data
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)
    
    # Load human annotations
    with open(annotation_json_path, 'r') as f:
        annotation_data = json.load(f)
    
    # Create segments from annotations
    segments = []
    current_segment = None
    
    # Group consecutive frames with similar concepts into segments
    frames = original_data['processing']['frames']
    
    for i, frame in enumerate(frames):
        # Get human annotation for this frame if exists
        frame_annotation = None
        for anno in annotation_data.get('frames', []):
            if anno.get('frame_index') == i:
                frame_annotation = anno
                break
        
        if frame_annotation and frame_annotation.get('concepts'):
            # Start new segment or extend existing
            if current_segment is None:
                current_segment = {
                    'start': frame['timestamp'],
                    'end': frame['timestamp'],
                    'concepts': frame_annotation['concepts'],
                    'key_frames': [i],
                    'summary': '',
                    'title': ' | '.join(frame_annotation['concepts'][:3])
                }
            else:
                # Check if concepts are similar enough to extend segment
                common_concepts = set(current_segment['concepts']) & set(frame_annotation['concepts'])
                if len(common_concepts) >= 1:
                    current_segment['end'] = frame['timestamp']
                    current_segment['key_frames'].append(i)
                else:
                    # Close current segment and start new one
                    segments.append(current_segment)
                    current_segment = {
                        'start': frame['timestamp'],
                        'end': frame['timestamp'],
                        'concepts': frame_annotation['concepts'],
                        'key_frames': [i],
                        'summary': '',
                        'title': ' | '.join(frame_annotation['concepts'][:3])
                    }
        else:
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None
    
    # Add final segment if exists
    if current_segment is not None:
        segments.append(current_segment)
    
    # Add transcript-based segments
    transcript_segments = original_data['processing']['transcript']['segments']
    
    # Merge transcript segments with visual segments
    merged_segments = []
    
    for i, ts in enumerate(transcript_segments):
        # Find overlapping visual segments
        overlapping_visual = []
        for vs in segments:
            if (ts['start'] <= vs['end'] and ts['end'] >= vs['start']):
                overlapping_visual.append(vs)
        
        if overlapping_visual:
            # Merge with visual information
            merged_segment = {
                'start': ts['start'],
                'end': ts['end'],
                'type': 'multimodal',
                'transcript_text': ts['text'],
                'visual_concepts': overlapping_visual[0]['concepts'] if overlapping_visual else [],
                'key_frames': overlapping_visual[0]['key_frames'] if overlapping_visual else [],
                'title': f"Segment {i+1}",
                'summary': ts['text'][:150] + "..." if len(ts['text']) > 150 else ts['text']
            }
        else:
            # Audio-only segment
            merged_segment = {
                'start': ts['start'],
                'end': ts['end'],
                'type': 'audio_only',
                'transcript_text': ts['text'],
                'visual_concepts': [],
                'key_frames': [],
                'title': f"Segment {i+1}",
                'summary': ts['text'][:150] + "..." if len(ts['text']) > 150 else ts['text']
            }
        
        merged_segments.append(merged_segment)
    
    # Update original data with merged annotations
    original_data['annotations']['segments'] = merged_segments
    original_data['annotations']['human_verified'] = True
    original_data['annotations']['verification_date'] = os.path.getmtime(annotation_json_path)
    
    # Calculate quality score
    total_frames = len(frames)
    annotated_frames = len(annotation_data.get('frames', []))
    quality_score = annotated_frames / total_frames if total_frames > 0 else 0
    
    original_data['annotations']['quality_score'] = round(quality_score, 3)
    
    # Save final JSON
    if output_json_path is None:
        output_json_path = original_json_path.replace('.json', '_final.json')
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Merged annotations saved to: {output_json_path}")
    print(f"📊 Stats: {len(merged_segments)} segments, Quality score: {quality_score:.2%}")
    
    return original_data

def batch_merge_annotations(processed_dir: str, annotations_dir: str):
    """Merge annotations for all processed videos"""
    
    processed_path = Path(processed_dir)
    annotations_path = Path(annotations_dir)
    
    for video_dir in processed_path.iterdir():
        if video_dir.is_dir():
            json_file = video_dir / f"{video_dir.name}.json"
            annotation_file = annotations_path / f"{video_dir.name}_annotated.json"
            
            if json_file.exists() and annotation_file.exists():
                print(f"\nMerging: {video_dir.name}")
                merge_human_annotations(str(json_file), str(annotation_file))

if __name__ == "__main__":
    # Example usage for single file
    original = "data/processed/lecture_001/lecture_001.json"
    annotations = "data/annotations/lecture_001_annotated.json"
    
    if os.path.exists(original) and os.path.exists(annotations):
        merge_human_annotations(original, annotations)
    
    # Or batch process
    # batch_merge_annotations("data/processed", "data/annotations")