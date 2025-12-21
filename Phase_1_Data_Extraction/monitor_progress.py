# monitor_progress.py
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

def generate_progress_report(processed_dir: str = "data/processed"):
    """Generate a progress report for all processed videos"""
    
    report_data = []
    
    for video_dir in Path(processed_dir).iterdir():
        if video_dir.is_dir():
            json_file = video_dir / f"{video_dir.name}.json"
            
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                report_data.append({
                    'Video ID': data['video_id'],
                    'Duration (s)': data['metadata']['duration'],
                    'Frames': len(data['processing']['frames']),
                    'Transcript Segments': len(data['processing']['transcript']['segments']),
                    'Instructor Frames': data['processing']['processing_stats']['frames_with_instructor'],
                    'Annotation Status': 'Yes' if data['annotations'].get('segments') else 'No',
                    'Quality Score': data['annotations'].get('quality_score', 'N/A'),
                    'Processing Date': data['metadata']['processing_date']
                })
    
    if report_data:
        df = pd.DataFrame(report_data)
        report_file = f"progress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(report_file, index=False)
        
        print("\n📊 Processing Progress Report")
        print("=" * 60)
        print(df.to_string())
        print(f"\n✅ Report saved to: {report_file}")
        
        # Summary stats
        print(f"\n📈 Summary:")
        print(f"Total videos processed: {len(df)}")
        print(f"Total frames: {df['Frames'].sum()}")
        print(f"Average duration: {df['Duration (s)'].mean():.1f}s")
        print(f"Annotation completion: {(df['Annotation Status'] == 'Yes').sum()}/{len(df)}")
        
        return df
    else:
        print("No processed videos found.")
        return None

if __name__ == "__main__":
    generate_progress_report()