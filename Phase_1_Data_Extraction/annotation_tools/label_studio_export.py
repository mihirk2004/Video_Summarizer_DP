# annotation_tools/label_studio_export.py
import json
import os
from pathlib import Path

def export_to_label_studio(json_path: str, output_path: str):
    """Convert processed JSON to Label Studio import format"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tasks = []
    
    # Create tasks for each frame that needs annotation
    for i, frame in enumerate(data['processing']['frames']):
        # Only annotate frames with instructor (to reduce workload)
        if frame.get('instructor', False):
            task = {
                "data": {
                    "image": f"/data/local-files/?d={frame['path']}",
                    "timestamp": frame['timestamp'],
                    "video_id": data['video_id'],
                    "frame_index": i
                },
                "predictions": [{
                    "model_version": "auto_annotation_v1",
                    "result": []
                }]
            }
            
            # Add auto-annotation predictions
            for concept in frame.get('concepts', []):
                if concept['confidence'] > 0.3:
                    task["predictions"][0]["result"].append({
                        "from_name": "concept",
                        "to_name": "image",
                        "type": "choices",
                        "value": {
                            "choices": [concept['concept']]
                        }
                    })
            
            tasks.append(task)
    
    # Save Label Studio tasks
    with open(output_path, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Exported {len(tasks)} tasks to {output_path}")
    return tasks

# Label Studio configuration template
LABEL_STUDIO_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true"/>
  
  <Header value="Frame Information"/>
  <Text name="timestamp" value="Timestamp: ${{timestamp}}s"/>
  <Text name="video_id" value="Video: ${{video_id}}"/>
  
  <Header value="Annotation Tasks"/>
  
  <Choices name="concept" toName="image" choice="multiple" showInLine="true">
    <Choice value="Mathematical Equation"/>
    <Choice value="Scientific Diagram"/>
    <Choice value="Computer Code"/>
    <Choice value="Instructor Pointing"/>
    <Choice value="Instructor Writing"/>
    <Choice value="Slide Presentation"/>
    <Choice value="Graph/Chart"/>
    <Choice value="Whiteboard Content"/>
  </Choices>
  
  <TextArea name="notes" toName="image" 
            rows="3" placeholder="Additional notes..."
            maxSubmissions="1"/>
  
  <Choices name="quality" toName="image" choice="single">
    <Choice value="Clear"/>
    <Choice value="Blurry"/>
    <Choice value="Too Dark"/>
    <Choice value="Too Bright"/>
  </Choices>
</View>
"""

if __name__ == "__main__":
    # Export all processed videos
    processed_dir = "data/processed"
    for video_dir in Path(processed_dir).iterdir():
        if video_dir.is_dir():
            json_file = video_dir / f"{video_dir.name}.json"
            if json_file.exists():
                output_file = f"data/annotations/label_studio_{video_dir.name}.json"
                export_to_label_studio(json_file, output_file)