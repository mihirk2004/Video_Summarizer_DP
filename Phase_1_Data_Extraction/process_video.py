#!/usr/bin/env python3
"""
Complete Lecture Video Processing Pipeline
Processes videos: Audio extraction → Transcription → Frame sampling → Auto-annotation
Output: JSON with all metadata
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import whisper
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import clip
from typing import Dict, List, Any
from skimage.metrics import structural_similarity as ssim


# Configuration
class Config:
    # Paths
    RAW_VIDEO_DIR = "data/raw_videos"
    PROCESSED_DIR = "data/processed"
    FRAMES_DIR = "data/processed/frames"
    
    # Processing settings
    FRAME_RATE = 1  # Extract 1 frame per second
    WHISPER_MODEL = "small"  # or "base", "medium", "large"
    YOLO_MODEL = "yolov8n-face.pt"  # Face detection model
    
    # CLIP settings for concept detection
    CLIP_MODELS = [
        "mathematical equation on whiteboard",
        "scientific diagram",
        "computer code on screen",
        "lecturer pointing at board",
        "lecturer writing on board",
        "slide presentation",
        "graph or chart"
    ]

def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract audio from video using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-q:a', '0',  # Best quality
            '-map', 'a',
            '-ac', '1',   # Mono audio
            '-ar', '16000',  # 16kHz sample rate
            audio_path,
            '-y'  # Overwrite output
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False

def transcribe_audio(audio_path: str, model_name: str = "small") -> Dict:
    """Transcribe audio using Whisper with timestamps"""
    try:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        
        # Transcribe with detailed timestamps
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            language='en',
            task='transcribe'
        )
        
        # Format segments
        formatted_segments = []
        for segment in result['segments']:
            formatted_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'words': [
                    {
                        'word': word['word'],
                        'start': word['start'],
                        'end': word['end']
                    }
                    for word in segment.get('words', [])
                ]
            })
        
        return {
            'text': result['text'],
            'segments': formatted_segments,
            'language': result['language']
        }
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def extract_frames(video_path: str, output_dir: str, fps: int = 1, ssim_thresh: float = 0.88) -> List[Dict]:
    """
    Extract frames at 1 FPS and apply SSIM-based frame sampling
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frames_data = []
    prev_gray = None
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (600, 300))

            save_frame = False

            if prev_gray is None:
                save_frame = True
            else:
                score = ssim(prev_gray, gray)
                if score < ssim_thresh:
                    save_frame = True

            if save_frame:
                filename = f"frame_{saved_idx:06d}.jpg"
                path = os.path.join(output_dir, filename)

                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                frames_data.append({
                    "index": saved_idx,
                    "timestamp": round(timestamp, 2),
                    "path": path,
                    "ssim_kept": True
                })

                prev_gray = gray
                saved_idx += 1

        frame_idx += 1

    cap.release()
    return frames_data


def detect_objects(frame_path: str, model) -> Dict:
    """Detect instructor and other objects in frame"""
    frame = cv2.imread(frame_path)
    if frame is None:
        return {'instructor': False, 'faces': []}
    
    # Run YOLO face detection
    results = model(frame)
    
    faces = []
    for box in results[0].boxes:
        if box.cls == 0:  # Assuming class 0 is face
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            
            faces.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class': 'face'
            })
    
    return {
        'instructor': len(faces) > 0,
        'faces': faces,
        'face_count': len(faces)
    }

def detect_concepts_clip(frame_path: str, clip_model, preprocess, text_features) -> Dict:
    """Use CLIP to detect concepts in frames"""
    try:
        import torch
        
        # Load and preprocess image
        image = cv2.imread(frame_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess for CLIP
        image_input = preprocess(image_rgb).unsqueeze(0)
        
        with torch.no_grad():
            # Extract features
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get top 3 concepts
            values, indices = similarity[0].topk(3)
            
            concepts = []
            for value, idx in zip(values, indices):
                concepts.append({
                    'concept': Config.CLIP_MODELS[idx],
                    'confidence': value.item()
                })
        
        return {'concepts': concepts}
        
    except Exception as e:
        print(f"CLIP detection error: {e}")
        return {'concepts': []}

def process_single_video(video_path: str, video_id: str) -> Dict:
    """Process a single video end-to-end"""
    print(f"\n{'='*60}")
    print(f"Processing: {video_path}")
    print(f"Video ID: {video_id}")
    print(f"{'='*60}")
    
    # Create output directories
    video_output_dir = os.path.join(Config.PROCESSED_DIR, video_id)
    frames_dir = os.path.join(video_output_dir, "frames")
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    # 1. Extract audio
    print("\n[1/5] Extracting audio...")
    audio_path = os.path.join(video_output_dir, "audio.wav")
    if not extract_audio(video_path, audio_path):
        print("Failed to extract audio")
        return None
    
    # 2. Transcribe audio
    print("[2/5] Transcribing audio...")
    transcript = transcribe_audio(audio_path, Config.WHISPER_MODEL)
    if transcript is None:
        print("Failed to transcribe audio")
        return None
    
    # 3. Extract frames
    print("[3/5] Extracting frames...")
    frames_data = extract_frames(video_path, frames_dir, Config.FRAME_RATE)
    
    # 4. Initialize detection models
    print("[4/5] Initializing detection models...")
    
    # Face detection model
    face_model = YOLO('yolov8n-face.pt')
    
    # CLIP model for concept detection
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=clip_device)
    
    # Prepare text features for CLIP
    text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in Config.CLIP_MODELS]).to(clip_device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 5. Process frames with detection
    print("[5/5] Running object/concept detection on frames...")
    for i, frame_info in enumerate(frames_data):
        if i % 10 == 0:
            print(f"  Processing frame {i+1}/{len(frames_data)}...")
        
        # Face detection
        detections = detect_objects(frame_info['path'], face_model)
        frame_info.update(detections)
        
        # Concept detection (every 5th frame for speed)
        if i % 5 == 0:
            concepts = detect_concepts_clip(
                frame_info['path'], 
                clip_model, 
                preprocess, 
                text_features
            )
            frame_info.update(concepts)
    
    # 6. Compile final JSON
    print("Compiling JSON output...")
    
    # Get video metadata
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Build the complete JSON structure
    output_json = {
        "video_id": video_id,
        "metadata": {
            "original_path": video_path,
            "duration": round(duration, 2),
            "fps": fps,
            "resolution": f"{width}x{height}",
            "processing_date": datetime.now().isoformat(),
            "subject": "STEM",  # Can be updated manually
            "institution": "Unknown",  # Can be updated manually
            "source": "YouTube"
        },
        "processing": {
            "audio_path": audio_path,
            "transcript": transcript,
            "frames": frames_data,
            "processing_stats": {
                "total_frames": len(frames_data),
                "frames_with_instructor": sum(1 for f in frames_data if f.get('instructor', False)),
                "avg_face_confidence": np.mean([f.get('face_confidence', 0) for f in frames_data if f.get('instructor', False)] or [0])
            }
        },
        "annotations": {
            "segments": [],  # To be filled by human annotation
            "quality_score": None,
            "auto_annotations": {
                "instructor_presence": [f['instructor'] for f in frames_data],
                "detected_concepts": [f.get('concepts', []) for f in frames_data]
            }
        }
    }
    
    # Save JSON file
    json_path = os.path.join(video_output_dir, f"{video_id}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Processing complete!")
    print(f"📁 Output saved to: {json_path}")
    print(f"📊 Stats:")
    print(f"   - Duration: {duration:.1f}s")
    print(f"   - Frames extracted: {len(frames_data)}")
    print(f"   - Transcript segments: {len(transcript['segments'])}")
    
    return output_json

def batch_process_videos(video_dir: str, max_videos: int = None, start_index: int = 1):
    """Process all videos in a directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    if max_videos:
        video_files = video_files[:max_videos]
    
    print(f"Found {len(video_files)} videos to process")
    
    for i, video_path in enumerate(video_files):
        lecture_number = start_index + i
        video_id = f"lecture_{lecture_number:03d}"
        
        print(f"\n📹 Processing video {lecture_number}: {video_path.name}")
        
        try:
            process_single_video(str(video_path.resolve()), video_id)
        except Exception as e:
            print(f"❌ Failed to process {video_path}: {e}")
            continue

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        # Process single video
        video_path = sys.argv[1]
        video_id = sys.argv[2] if len(sys.argv) > 2 else Path(video_path).stem
        process_single_video(video_path, video_id)
    else:
        # Batch process all videos in raw directory
        batch_process_videos(
    Config.RAW_VIDEO_DIR,
    max_videos=None,
    start_index=217
)
