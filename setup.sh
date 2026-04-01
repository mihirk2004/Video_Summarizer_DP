#!/bin/bash
# setup.sh - Complete environment setup

echo "Setting up Lecture Video Processing Pipeline..."
echo "=============================================="

# # Create directory structure
# mkdir -p data/{raw_videos,processed,annotations}
# mkdir -p annotation_tools
# mkdir -p utils

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip

# Core packages
pip install opencv-python pillow numpy pandas scikit-learn
pip install whisper openai-whisper
pip install ultralytics torch torchvision torchaudio
pip install transformers datasets
pip install git+https://github.com/openai/CLIP.git
pip install yt-dlp moviepy

# Annotation tools
pip install label-studio

# Install ffmpeg (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y ffmpeg

# Download YOLO face model
echo "Downloading YOLO face detection model..."
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n-face.pt')
"

echo "✅ Setup complete!"
echo ""
echo "Quick start commands:"
echo "1. Download videos: python run_pipeline.py"
echo "2. Process single video: python process_video.py path/to/video.mp4"
echo "3. Start annotation: python annotation_tools/annotation_gui.py"
echo "4. Batch process: python process_video.py (no arguments)"