# run_pipeline.py
#!/usr/bin/env python3
"""
Complete pipeline runner - from video download to final annotation
"""

import os
import subprocess
import time
from pathlib import Path

def download_youtube_videos(channel_urls: list, output_dir: str, max_videos: int = 50):
    """Download videos from YouTube using yt-dlp"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, url in enumerate(channel_urls[:max_videos]):
        print(f"\nDownloading video {i+1}/{max_videos}...")
        
        cmd = [
            'yt-dlp',
            '--format', 'mp4',
            '--write-auto-sub',
            '--sub-lang', 'en',
            '--output', f'{output_dir}/video_%(id)s.%(ext)s',
            '--quiet',
            url
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Downloaded: {url}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download {url}: {e}")

def main():
    """Main pipeline execution"""
    
    print("🎬 Lecture Video Processing Pipeline")
    print("=" * 50)
    
    # Step 1: Download videos (optional)
    youtube_channels = [
        "https://youtube.com/playlist?list=PLSQl0a2vh4HD2AZcy7Pe8mHAiO5UG2WKt&si=Loj7_tyEA6DrZscm"
        #"https://youtu.be/Je_f4RimfKI?si=Gs7xUm3Sd8J-YzUN"
        # "https://www.youtube.com/watch?v=example1",
        # "https://www.youtube.com/watch?v=example2",
        # Add your YouTube URLs here
    ]
    
    download_choice = input("Download YouTube videos? (y/n): ")
    if download_choice.lower() == 'y':
        print("\nStep 1: Downloading videos...")
        download_youtube_videos(youtube_channels, "data/raw_videos", max_videos=50)
    
    # Step 2: Process videos
    print("\nStep 2: Processing videos...")
    os.system("python process_video.py")
    
    # Step 3: Export for annotation
    print("\nStep 3: Preparing for annotation...")
    os.system("python annotation_tools/label_studio_export.py")
    
    print("\n" + "=" * 50)
    print("📋 Next Steps:")
    print("1. Open Label Studio at http://localhost:8080")
    print("2. Import the tasks from data/annotations/label_studio_*.json")
    print("3. Annotate the frames")
    print("4. Export annotations from Label Studio")
    print("5. Run: python annotation_tools/annotation_merger.py")
    print("=" * 50)
    
    # Alternatively, run GUI annotation tool
    gui_choice = input("\nRun GUI annotation tool instead? (y/n): ")
    if gui_choice.lower() == 'y':
        print("Launching GUI annotation tool...")
        os.system("python annotation_tools/annotation_gui.py")

if __name__ == "__main__":
    main()