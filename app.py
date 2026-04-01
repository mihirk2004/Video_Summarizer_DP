#!/usr/bin/env python3
"""
Flask Web Application — Multimodal Lecture Video Summarizer
Dark-themed creative UI for the full 4-phase pipeline.

Routes:
    GET  /                          → Home page (video input)
    POST /api/upload                → Upload video file
    POST /api/process               → Start pipeline processing
    GET  /api/progress/<job_id>     → SSE progress stream
    GET  /api/status/<job_id>       → JSON status check
    GET  /document/<job_id>         → Document viewer page
    GET  /api/document/<job_id>/<fmt> → Download generated document
"""

import os
import sys
import json
import uuid
import time
import threading
from pathlib import Path
from queue import Queue

from flask import (
    Flask, render_template, request, jsonify,
    Response, send_file, redirect, url_for
)

# Project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250MB max upload
app.config['UPLOAD_FOLDER'] = str(PROJECT_ROOT / 'uploads')
app.config['OUTPUT_FOLDER'] = str(PROJECT_ROOT / 'output')
app.secret_key = os.environ.get('SECRET_KEY', 'multimodal-lecture-summarizer-dev-key')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ──────────────────────────────────────────────
#  Job Tracking
# ──────────────────────────────────────────────

# In-memory job store. In production, use Redis or a database.
jobs = {}  # job_id -> { status, progress_queue, result, ... }

ALLOWED_EXTENSIONS = {'mp4', 'mkv', 'avi', 'mov', 'webm', 'flv', 'wmv'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────
#  Routes — Pages
# ──────────────────────────────────────────────

@app.route('/')
def index():
    """Home page — video input (upload or YouTube URL)"""
    return render_template('index.html')


@app.route('/document/<job_id>')
def document_viewer(job_id):
    """Document viewer page"""
    job = jobs.get(job_id)
    if not job:
        return render_template('index.html', error="Job not found."), 404

    if job.get('status') != 'completed':
        return redirect(url_for('index'))

    # Load document data
    doc_data_path = Path(app.config['OUTPUT_FOLDER']) / job_id / 'document_data.json'
    doc_data = {}
    if doc_data_path.exists():
        with open(str(doc_data_path), 'r', encoding='utf-8') as f:
            doc_data = json.load(f)

    return render_template('document.html', job_id=job_id, doc=doc_data)


# ──────────────────────────────────────────────
#  Routes — API
# ──────────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Save file with unique name
    job_id = str(uuid.uuid4())[:8]
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{job_id}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return jsonify({
        "job_id": job_id,
        "video_path": filepath,
        "filename": file.filename,
        "message": "Upload successful"
    })


@app.route('/api/process', methods=['POST'])
def start_processing():
    """Start pipeline processing"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    video_path = data.get('video_path')
    youtube_url = data.get('youtube_url')
    job_id = data.get('job_id') or str(uuid.uuid4())[:8]

    # Handle YouTube URL — download first
    if youtube_url and not video_path:
        try:
            video_path = _download_youtube(youtube_url, job_id)
        except Exception as e:
            return jsonify({"error": f"YouTube download failed: {str(e)}"}), 400

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 400

    # Initialize job
    progress_queue = Queue()
    jobs[job_id] = {
        "status": "processing",
        "progress_queue": progress_queue,
        "video_path": video_path,
        "result": None,
        "started_at": time.time(),
    }

    # Start pipeline in background thread
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, video_path, progress_queue),
        daemon=True
    )
    thread.start()

    return jsonify({
        "job_id": job_id,
        "message": "Processing started",
        "status": "processing"
    })


@app.route('/api/progress/<job_id>')
def progress_stream(job_id):
    """SSE endpoint for real-time progress updates"""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        queue = job['progress_queue']
        while True:
            try:
                # Wait for progress update (timeout after 30s to send keepalive)
                try:
                    data = queue.get(timeout=30)
                except Exception:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                    continue

                yield f"data: {json.dumps(data)}\n\n"

                # If completed or failed, close stream
                if data.get('status') in ('completed', 'failed'):
                    break

            except GeneratorExit:
                break

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


@app.route('/api/status/<job_id>')
def job_status(job_id):
    """Get current job status as JSON"""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "job_id": job_id,
        "status": job.get("status"),
        "result": job.get("result"),
    })


@app.route('/api/document/<job_id>/<fmt>')
def download_document(job_id, fmt):
    """Download generated document in specified format"""
    job = jobs.get(job_id)
    if not job or job.get('status') != 'completed':
        return jsonify({"error": "Document not ready"}), 404

    result = job.get('result', {})
    doc_paths = result.get('document_paths', {})

    format_map = {
        'html': ('html', 'text/html'),
        'markdown': ('markdown', 'text/markdown'),
        'md': ('markdown', 'text/markdown'),
        'pdf': ('pdf', 'application/pdf'),
        'json': ('json', 'application/json'),
    }

    if fmt not in format_map:
        return jsonify({"error": f"Invalid format. Available: {list(format_map.keys())}"}), 400

    key, mimetype = format_map[fmt]
    file_path = doc_paths.get(key)

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": f"{fmt.upper()} document not available"}), 404

    return send_file(
        file_path,
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"lecture_summary_{job_id}.{fmt}"
    )

@app.route('/api/frames/<job_id>/<filename>')
def serve_frame(job_id, filename):
    """Serve frame images from the job output directory"""
    import re
    # Sanitize inputs to prevent directory traversal
    if not re.match(r'^[a-zA-Z0-9_-]+$', job_id):
        return jsonify({"error": "Invalid job ID"}), 400
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({"error": "Invalid filename"}), 400

    frames_dir = Path(app.config['OUTPUT_FOLDER']) / job_id / 'frames'
    file_path = frames_dir / filename

    if not file_path.exists():
        return jsonify({"error": "Frame not found"}), 404

    return send_file(str(file_path), mimetype='image/jpeg')


# ──────────────────────────────────────────────
#  Background Workers
# ──────────────────────────────────────────────

def _run_pipeline(job_id: str, video_path: str, progress_queue: Queue):
    """Run the full pipeline in a background thread"""
    from pipeline import LecturePipeline

    def progress_callback(data):
        progress_queue.put(data)

    try:
        pipeline = LecturePipeline(
            job_id=job_id,
            video_path=video_path,
            output_dir=app.config['OUTPUT_FOLDER'],
            progress_callback=progress_callback,
        )
        result = pipeline.run()

        jobs[job_id]['status'] = 'completed' if 'error' not in result else 'failed'
        jobs[job_id]['result'] = result

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['result'] = {"error": error_msg}
        progress_queue.put({
            "status": "failed",
            "error": error_msg,
            "job_id": job_id,
        })


def _download_youtube(url: str, job_id: str) -> str:
    """Download YouTube video using yt-dlp subprocess with robust fallbacks."""
    import subprocess
    import shutil

    upload_dir = app.config['UPLOAD_FOLDER']
    output_template = os.path.join(upload_dir, f"{job_id}.%(ext)s")

    # Check if ffmpeg is available (needed for merging)
    ffmpeg_available = shutil.which('ffmpeg') is not None

    # Define fallback strategies as tuples of (format_spec, merge_output_format, description)
    strategies = [
        (None, None, "Default (yt-dlp chooses best format)"),
        ('best', None, "Explicit 'best' format"),
        ('bestvideo+bestaudio', 'mp4', "bestvideo+bestaudio merged to mp4 (requires ffmpeg)"),
    ]

    last_error = None

    for fmt, merge_fmt, desc in strategies:
        # Skip merge strategy if ffmpeg missing
        if merge_fmt and not ffmpeg_available:
            print(f"Skipping {desc}: ffmpeg not installed.")
            continue

        # Build command
        cmd = ['yt-dlp', '--no-playlist', '--no-check-certificate', '--socket-timeout', '30',
               '--retries', '3', '--fragment-retries', '3', '-o', output_template]
        if fmt:
            cmd.extend(['-f', fmt])
        if merge_fmt:
            cmd.extend(['--merge-output-format', merge_fmt])

        print(f"Trying strategy: {desc}")
        result = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Find the downloaded file
            for f in sorted(os.listdir(upload_dir), key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)), reverse=True):
                if f.startswith(job_id):
                    full_path = os.path.join(upload_dir, f)
                    if os.path.getsize(full_path) > 0:
                        print(f"✓ YouTube video downloaded: {full_path}")
                        return full_path
                    else:
                        os.remove(full_path)
                        raise RuntimeError("Downloaded file is empty")
            # If we get here, no file found – but download succeeded? Possibly naming mismatch
            print("Download completed but file not found. Listing uploads:")
            print(os.listdir(upload_dir))
            raise FileNotFoundError("Downloaded file not found after successful download")

        else:
            # Log error and continue to next strategy
            print(f"Strategy failed (exit {result.returncode}): {result.stderr.strip()}")
            last_error = result.stderr.strip()
            # Clean up any partial files
            for f in os.listdir(upload_dir):
                if f.startswith(job_id):
                    try:
                        os.remove(os.path.join(upload_dir, f))
                    except OSError:
                        pass

    # If we exit the loop, all strategies failed
    raise RuntimeError(
        f"All download strategies failed. Last error: {last_error}\n"
        "Possible causes:\n"
        "  - The video is age‑restricted or requires login.\n"
        "  - Your yt‑dlp version is outdated. Try: pip install --upgrade yt-dlp\n"
        "  - ffmpeg is missing and the video needs merging. Install ffmpeg.\n"
        "  - The video URL may be invalid or private.\n"
        "  - Try downloading manually to verify: yt-dlp -f best <URL>"
    )


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Multimodal Lecture Video Summarizer")
    print("  http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
