#!/usr/bin/env python3
"""
Unified Pipeline Coordinator for the Multimodal Lecture Video Summarizer.
Orchestrates all 4 phases: Video Processing → Frame Classification →
Summarization → Document Generation, with real-time progress callbacks.
"""

import sys
import os
import gc
import json
import time
import uuid
import traceback
from pathlib import Path
from typing import Dict, Callable, Optional, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class PipelineProgress:
    """Track and communicate pipeline progress"""

    PHASES = {
        1: "Video Processing",
        2: "Frame Classification",
        3: "Text Summarization",
        4: "Document Generation",
    }

    def __init__(self, job_id: str, callback: Optional[Callable] = None):
        self.job_id = job_id
        self.callback = callback
        self.current_phase = 0
        self.total_phases = 4
        self.status = "initialized"
        self.message = ""
        self.progress = 0.0  # 0-100
        self.error = None
        self.result = None

    def update(self, phase: int, message: str, progress: float = None):
        """Update progress and notify callback"""
        self.current_phase = phase
        self.status = "processing"
        self.message = message
        if progress is not None:
            self.progress = progress
        phase_name = self.PHASES.get(phase, f"Phase {phase}")
        if self.callback:
            self.callback({
                "job_id": self.job_id,
                "phase": phase,
                "phase_name": phase_name,
                "total_phases": self.total_phases,
                "message": message,
                "progress": self.progress,
                "status": self.status,
            })

    def complete(self, result: Dict):
        """Mark pipeline as completed"""
        self.status = "completed"
        self.progress = 100.0
        self.result = result
        if self.callback:
            self.callback({
                "job_id": self.job_id,
                "phase": self.total_phases,
                "phase_name": "Complete",
                "total_phases": self.total_phases,
                "message": "Pipeline completed successfully!",
                "progress": 100.0,
                "status": "completed",
                "result": result,
            })

    def fail(self, error: str):
        """Mark pipeline as failed"""
        self.status = "failed"
        self.error = error
        if self.callback:
            self.callback({
                "job_id": self.job_id,
                "phase": self.current_phase,
                "phase_name": self.PHASES.get(self.current_phase, "Unknown"),
                "total_phases": self.total_phases,
                "message": f"Error: {error}",
                "progress": self.progress,
                "status": "failed",
                "error": error,
            })


class LecturePipeline:
    """
    Orchestrates: Video Processing → Classification → Summarization → Document Generation
    """

    def __init__(
        self,
        job_id: str,
        video_path: str,
        output_dir: str = "output",
        progress_callback: Callable = None,
        use_gpu: bool = True,
    ):
        self.job_id = job_id
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu and self._check_gpu()
        self.progress = PipelineProgress(job_id, progress_callback)

        # Intermediate data
        self.video_metadata = {}
        self.frame_classifications = {}
        self.segments = []
        self.summaries = []
        self.document_paths = {}

    def _check_gpu(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def run(self) -> Dict:
        """Execute the full 4-phase pipeline"""
        try:
            start_time = time.time()

            # Phase 1: Video Processing
            self.step_1_process_video()

            # Phase 2: Frame Classification
            self.step_2_classify_frames()

            # Phase 3: Text Summarization
            self.step_3_summarize()

            # Phase 4: Document Generation
            self.step_4_generate_document()

            elapsed = time.time() - start_time
            result = {
                "job_id": self.job_id,
                "video_path": str(self.video_path),
                "output_dir": str(self.output_dir),
                "document_paths": self.document_paths,
                "elapsed_seconds": round(elapsed, 1),
                "total_segments": len(self.segments),
                "gpu_used": self.use_gpu,
            }

            self.progress.complete(result)
            return result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.progress.fail(error_msg)
            traceback.print_exc()
            return {"error": error_msg, "job_id": self.job_id}

    # ──────────────────────────────────────────────
    #  Phase 1: Video Processing
    # ──────────────────────────────────────────────
    def step_1_process_video(self):
        """Extract audio, transcribe, extract frames"""
        self.progress.update(1, "Starting video processing...", 0)

        from process_video import (
            extract_audio, transcribe_audio,
            extract_frames, detect_objects,
        )

        video_str = str(self.video_path)
        video_name = self.video_path.stem
        lecture_id = f"lecture_{video_name}"

        # Create working directories
        audio_dir = self.output_dir / "audio"
        frames_dir = self.output_dir / "frames"
        audio_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Step 1a: Extract audio
        self.progress.update(1, "Extracting audio from video...", 5)
        audio_path = str(audio_dir / f"{video_name}.wav")
        extract_audio(video_str, audio_path)

        # Step 1b: Transcribe with Whisper
        self.progress.update(1, "Transcribing audio with Whisper...", 15)
        transcript_data = transcribe_audio(audio_path)

        # Step 1c: Extract frames
        self.progress.update(1, "Extracting keyframes from video...", 30)
        frame_data = extract_frames(video_str, str(frames_dir))

        # Step 1d: Object + concept detection on frames
        self.progress.update(1, "Running object detection on frames...", 40)
        frame_paths = [
            str(frames_dir / f)
            for f in os.listdir(str(frames_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        detection_results = {}
        for fp in frame_paths:
            try:
                det = detect_objects(fp)
                detection_results[fp] = det
            except Exception:
                detection_results[fp] = {"faces": [], "objects": []}

        # Store metadata
        self.video_metadata = {
            "video_id": lecture_id,
            "video_path": video_str,
            "audio_path": audio_path,
            "transcript": transcript_data,
            "frames_dir": str(frames_dir),
            "frame_paths": sorted(frame_paths),
            "frame_count": len(frame_paths),
            "detections": detection_results,
        }

        # Save metadata
        meta_path = self.output_dir / "video_metadata.json"
        with open(str(meta_path), 'w', encoding='utf-8') as f:
            json.dump(self.video_metadata, f, indent=2, default=str)

        self.progress.update(1, f"Video processing complete. {len(frame_paths)} frames extracted.", 50)

    # ──────────────────────────────────────────────
    #  Phase 2: Frame Classification
    # ──────────────────────────────────────────────
    def step_2_classify_frames(self):
        """Classify extracted frames using hybrid ResNet50 + CLIP"""
        self.progress.update(2, "Loading visual classification models...", 50)

        from visual_classifier import HybridVisualClassifier

        classifier = HybridVisualClassifier(
            model_dir="models/visual",
            strategy="max_confidence",
            threshold=0.7,
        )

        frame_paths = self.video_metadata.get("frame_paths", [])
        if not frame_paths:
            self.progress.update(2, "No frames to classify.", 60)
            self.frame_classifications = {}
            return

        self.progress.update(2, f"Classifying {len(frame_paths)} frames...", 52)

        def _cls_progress(done, total):
            pct = 52 + (done / total) * 8
            self.progress.update(2, f"Classifying frames: {done}/{total}", pct)

        results = classifier.classify_frames_batch(frame_paths, _cls_progress)

        # Build classification map: path -> category
        self.frame_classifications = {}
        for r in results:
            self.frame_classifications[r['path']] = {
                'category': r['category'],
                'confidence': r['confidence'],
                'decided_by': r.get('decided_by', 'unknown'),
            }

        # Unload visual models
        classifier.unload_models()

        # Save classification results
        cls_path = self.output_dir / "frame_classifications.json"
        with open(str(cls_path), 'w', encoding='utf-8') as f:
            json.dump(self.frame_classifications, f, indent=2)

        self.progress.update(2, f"Frame classification complete. {len(results)} frames classified.", 60)

    # ──────────────────────────────────────────────
    #  Phase 3: Text Summarization
    # ──────────────────────────────────────────────
    def step_3_summarize(self):
        """Segment transcript and run BART + LLM hybrid summarization"""
        self.progress.update(3, "Preparing text segments...", 60)

        import re
        import torch

        transcript = self.video_metadata.get("transcript", {})
        whisper_segments = transcript.get("segments", [])

        if not whisper_segments:
            self.progress.update(3, "No transcript segments found.", 75)
            return

        # Build text segments (400-word chunking fallback)
        self.progress.update(3, "Segmenting transcript...", 62)
        self.segments = self._segment_transcript(whisper_segments)

        if not self.segments:
            self.progress.update(3, "No valid segments created.", 75)
            return

        # Match frames to segments
        self.progress.update(3, "Matching frames to segments...", 65)
        self._match_frames_to_segments()

        # Copy matched frame images to job output for web serving
        self.progress.update(3, "Preparing frame images...", 66)
        self._copy_frames_to_output()

        # Run summarization
        self.progress.update(3, "Loading summarization model...", 68)

        try:
            from scripts.inference.lecture_summarizer import LectureSummarizer
            
            # We will use either 'hybrid' or 'bart' for purely text segments
            mode = 'hybrid' if self.use_gpu else 'bart'
            text_summarizer = LectureSummarizer(mode=mode)
            
            # Attempt to load multimodal model
            fusion_model = None
            fusion_tokenizer = None
            vis_classifier = None
            fusion_meta = None
            device = torch.device('cuda' if self.use_gpu else 'cpu')
            fusion_dir = Path("models/multimodal/clip_bart_fusion/best_model")
            
            if fusion_dir.exists():
                self.progress.update(3, "Loading Multimodal Fusion model...", 69)
                try:
                    import importlib
                    from transformers import AutoTokenizer, BartForConditionalGeneration
                    from peft import PeftModel
                    _fusion_module = importlib.import_module("scripts.text_processing.14_train_clip_bart_fusion")
                    CLIPBARTFusion = _fusion_module.CLIPBARTFusion
                    
                    with open(fusion_dir / "fusion_meta.json", 'r') as f:
                        fusion_meta = json.load(f)
                    
                    bart_lora_dir = fusion_dir / "bart_lora"
                    fusion_tokenizer = AutoTokenizer.from_pretrained(str(bart_lora_dir))
                    base_bart = BartForConditionalGeneration.from_pretrained(fusion_meta.get('bart_base', 'facebook/bart-large-cnn'))
                    base_bart.resize_token_embeddings(len(fusion_tokenizer))
                    
                    bart_model = PeftModel.from_pretrained(base_bart, str(bart_lora_dir), is_trainable=False)
                    fusion_model = CLIPBARTFusion(
                        bart_model=bart_model,
                        clip_dim=fusion_meta['clip_dim'],
                        bart_dim=fusion_meta['bart_dim'],
                        hidden_dim=fusion_meta['hidden_dim'],
                        max_images=fusion_meta['max_images'],
                    )
                    
                    proj_state = torch.load(str(fusion_dir / "projection.pt"), map_location='cpu', weights_only=True)
                    fusion_model.visual_projection.load_state_dict(proj_state['visual_projection'])
                    fusion_model.visual_gate.data = proj_state['visual_gate']
                    fusion_model.to(device)
                    fusion_model.eval()
                    
                    from visual_classifier import HybridVisualClassifier
                    vis_classifier = HybridVisualClassifier(model_dir="models/visual", device=device)
                    self.progress.update(3, "Multimodal Fusion model loaded.", 70)
                except Exception as e:
                    print(f"Failed to load fusion model: {e}")
                    fusion_model = None

            total_segs = len(self.segments)
            for i, seg in enumerate(self.segments):
                pct = 70 + ((i + 1) / total_segs) * 8
                self.progress.update(3, f"Summarizing segment {i+1}/{total_segs}...", pct)

                t0 = time.time()
                try:
                    # 1. Check if we have visuals & fusion model exists
                    if fusion_model and seg.get('matched_frames'):
                        # CLIP Inference
                        max_img = fusion_meta['max_images']
                        clip_emb = torch.zeros(max_img, fusion_meta['clip_dim']).to(device)
                        img_mask = torch.zeros(max_img).to(device)
                        
                        from PIL import Image
                        n_filled = 0
                        for f_path in seg['matched_frames'][:max_img]:
                            if os.path.exists(f_path):
                                img = Image.open(f_path).convert("RGB")
                                emb = vis_classifier.get_clip_embedding(img).to(device)
                                clip_emb[n_filled] = emb
                                img_mask[n_filled] = 1.0
                                n_filled += 1
                        
                        # Generate text tokens
                        text = seg['raw_text']
                        text_enc = fusion_tokenizer(text, max_length=1024, truncation=True, padding=True, return_tensors="pt").to(device)
                        
                        # Forward pass
                        with torch.no_grad():
                            gen_ids = fusion_model.generate(
                                input_ids=text_enc['input_ids'],
                                attention_mask=text_enc['attention_mask'],
                                clip_embeddings=clip_emb.unsqueeze(0),
                                image_mask=img_mask.unsqueeze(0),
                                max_new_tokens=350,
                                min_new_tokens=10,
                                num_beams=4,
                                no_repeat_ngram_size=3,
                                repetition_penalty=1.2,
                                length_penalty=2.0,
                                early_stopping=True,
                            )
                        summary = fusion_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                        seg['summary'] = summary
                        seg['bart_draft'] = summary
                        seg['mode'] = 'fusion'
                        seg['summarize_time'] = round(time.time() - t0, 2)
                    
                    else:
                        # 2. Text-only fallback
                        result = text_summarizer.summarize(seg['raw_text'])
                        seg['summary'] = result.get('summary', '')
                        seg['bart_draft'] = result.get('bart_draft', '')
                        seg['mode'] = result.get('mode', 'bart')
                        seg['summarize_time'] = result.get('time_seconds', 0.0)
                        
                except Exception as e:
                    seg['summary'] = f"[Summarization error: {str(e)}]"
                    seg['summarize_time'] = 0.0

            if vis_classifier:
                vis_classifier.unload_models()
            if fusion_model:
                del fusion_model
                del fusion_tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Compute quality scores using SBERT similarity
            self.progress.update(3, "Scoring summary quality...", 78)
            self._score_summaries()

            # Generate Hierarchical Map-Reduce Summary
            self.progress.update(3, "Generating hierarchical overall summary...", 79)
            self._generate_overall_summary(text_summarizer)

            # Unload standard summarizer
            text_summarizer.unload()

        except ImportError as e:
            print(f"Summarizer error: {e}")
            self.progress.update(3, "Summarizer not available. Using fallback.", 75)
            for seg in self.segments:
                text = seg['raw_text']
                sentences = re.split(r'(?<=[.!?])\s+', text)
                seg['summary'] = ' '.join(sentences[:2]) if sentences else text[:200]
                seg['quality_score'] = 0.0
            self.overall_summary = ' '.join(
                s.get('summary', '') for s in self.segments if s.get('summary')
            )

        # Save segments
        seg_path = self.output_dir / "segments.json"
        with open(str(seg_path), 'w', encoding='utf-8') as f:
            json.dump(self.segments, f, indent=2)

        self.progress.update(3, f"Summarization complete. {len(self.segments)} segments processed.", 80)

    # ──────────────────────────────────────────────
    #  Phase 4: Document Generation
    # ──────────────────────────────────────────────
    def step_4_generate_document(self):
        """Generate HTML, Markdown, and PDF documents"""
        self.progress.update(4, "Generating documents...", 80)

        import re

        video_name = self.video_path.stem
        lecture_id = self.video_metadata.get("video_id", f"lecture_{video_name}")

        # Use proper overall summary (generated in Phase 3)
        overall_summary = getattr(self, 'overall_summary', None)
        if not overall_summary:
            all_summaries = [s.get('summary', '') for s in self.segments if s.get('summary')]
            overall_summary = ' '.join(all_summaries) if all_summaries else "No summary available."

        # Build document data structure
        doc_data = {
            "lecture_id": lecture_id,
            "title": video_name.replace('_', ' ').replace('-', ' ').title(),
            "overall_summary": overall_summary,
            "total_segments": len(self.segments),
            "segments": [],
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        for seg in self.segments:
            seg_data = {
                "segment_id": seg.get('segment_id', ''),
                "start": seg.get('start', 0),
                "end": seg.get('end', 0),
                "summary": seg.get('summary', ''),
                "quality_score": seg.get('quality_score', 0.0),
                "visual_tags": seg.get('visual_tags', []),
                "images": [],
            }

            # Add matched frame images with web-accessible paths
            for frame_path in seg.get('matched_frames', []):
                cls_data = self.frame_classifications.get(frame_path, {})
                # Use the web-accessible path from output directory
                web_path = seg.get('web_frame_paths', {}).get(frame_path, '')
                seg_data["images"].append({
                    "path": frame_path,
                    "web_path": web_path,
                    "filename": Path(frame_path).name,
                    "category": cls_data.get('category', 'Unknown'),
                    "confidence": cls_data.get('confidence', 0.0),
                })

            doc_data["segments"].append(seg_data)

        # Generate HTML document
        self.progress.update(4, "Generating HTML document...", 85)
        html_path = self._generate_html(doc_data)

        # Generate Markdown document
        self.progress.update(4, "Generating Markdown document...", 90)
        md_path = self._generate_markdown(doc_data)

        # Save full data as JSON
        json_path = self.output_dir / "document_data.json"
        with open(str(json_path), 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, indent=2, default=str)

        self.document_paths = {
            "html": str(html_path),
            "markdown": str(md_path),
            "json": str(json_path),
        }

        # Try generating PDF
        self.progress.update(4, "Generating PDF document...", 95)
        try:
            pdf_path = self._generate_pdf(doc_data)
            self.document_paths["pdf"] = str(pdf_path)
        except Exception as e:
            print(f"PDF generation failed (fpdf2 error): {e}")

        self.progress.update(4, "Document generation complete!", 100)

    # ──────────────────────────────────────────────
    #  Helper Methods
    # ──────────────────────────────────────────────

    def _copy_frames_to_output(self):
        """Copy matched frame images into job output for web serving"""
        import shutil

        frames_out_dir = self.output_dir / "frames"
        frames_out_dir.mkdir(parents=True, exist_ok=True)

        for seg in self.segments:
            web_paths = {}
            for fp in seg.get('matched_frames', []):
                src = Path(fp)
                if src.exists():
                    dst = frames_out_dir / src.name
                    if not dst.exists():
                        shutil.copy2(str(src), str(dst))
                    # Web path relative to /api/frames/<job_id>/
                    web_paths[fp] = f"/api/frames/{self.job_id}/{src.name}"
            seg['web_frame_paths'] = web_paths

    def _score_summaries(self):
        """Score summary quality using SBERT cosine similarity (source vs summary)"""
        try:
            from sentence_transformers import SentenceTransformer, util

            sbert = SentenceTransformer('all-MiniLM-L6-v2')

            for seg in self.segments:
                summary = seg.get('summary', '')
                source = seg.get('raw_text', '')

                if not summary or summary.startswith('['):
                    seg['quality_score'] = 0.0
                    continue

                # SBERT cosine similarity as quality proxy
                emb_source = sbert.encode(source[:1000], convert_to_tensor=True)
                emb_summary = sbert.encode(summary, convert_to_tensor=True)
                score = float(util.cos_sim(emb_source, emb_summary)[0][0])
                seg['quality_score'] = max(0.0, min(1.0, score))

            del sbert
            gc.collect()

        except ImportError:
            # Fallback: length-ratio heuristic
            for seg in self.segments:
                summary = seg.get('summary', '')
                source = seg.get('raw_text', '')
                if not summary or summary.startswith('['):
                    seg['quality_score'] = 0.0
                else:
                    ratio = len(summary.split()) / max(len(source.split()), 1)
                    # Good summaries are ~10-30% of source length
                    seg['quality_score'] = min(1.0, ratio * 5) if 0.05 < ratio < 0.5 else 0.3

    def _generate_overall_summary(self, text_summarizer):
        """Map-Reduce Hierarchical Summary: Group by blocks -> section summaries -> final summary"""
        all_summaries = [
            s.get('summary', '') for s in self.segments
            if s.get('summary') and not s['summary'].startswith('[')
        ]

        if not all_summaries:
            self.overall_summary = "No summary available."
            return

        # Map-Reduce logic
        block_size = max(3, len(all_summaries) // 5) # Group into ~5 blocks
        section_summaries = []

        # Map phase: Generate Section Summaries
        try:
            for i in range(0, len(all_summaries), block_size):
                block = ' '.join(all_summaries[i:i + block_size])
                if len(block.split()) > 40:
                    res = text_summarizer.summarize(block)
                    section_summaries.append(res.get('summary', block[:300]))
                else:
                    section_summaries.append(block)

            # Reduce phase: Combine Sections into Final
            combined_sections = ' '.join(section_summaries)
            if len(combined_sections.split()) <= 150:
                 self.overall_summary = combined_sections
            else:
                 res = text_summarizer.summarize(combined_sections)
                 condensed = res.get('summary', '')
                 if len(condensed.split()) > 10:
                      self.overall_summary = condensed
                 else:
                      self.overall_summary = combined_sections[:500]

        except Exception as e:
            print(f"Hierarchical summary failed: {e}")
            self.overall_summary = ' '.join(all_summaries[:3])

    def _segment_transcript(self, whisper_segments: List[Dict]) -> List[Dict]:
        """Segment transcript by 400-word chunks with timestamp mapping"""
        import re

        min_words = 50
        max_words = 800
        chunk_size = 400

        # Build sentences with timestamps
        sentences = []
        current_sent = []
        current_start = None
        current_end = None

        for ws in whisper_segments:
            txt = ws.get('text', '').strip()
            if not txt:
                continue
            start = ws.get('start')
            end = ws.get('end')
            if start is None or end is None:
                continue

            parts = re.split(r'(?<=[.!?])\s+', txt)
            for part in parts:
                if not part:
                    continue
                if current_start is None:
                    current_start = start
                    current_end = end
                current_sent.append(part)
                if part[-1] in '.!?':
                    full_sent = ' '.join(current_sent)
                    sentences.append((current_start, current_end, full_sent))
                    current_sent = []
                    current_start = None
                    current_end = None
                else:
                    current_end = end

        if current_sent:
            full_sent = ' '.join(current_sent)
            sentences.append((current_start, current_end, full_sent))

        # Group into chunks
        chunks = []
        current_chunk_sents = []
        word_count = 0
        chunk_start = None
        chunk_end = None
        lecture_id = self.video_metadata.get('video_id', 'lecture')

        for sent_start, sent_end, sent_text in sentences:
            if sent_start is None or sent_end is None:
                continue
            wc = len(sent_text.split())
            if word_count + wc > chunk_size and current_chunk_sents:
                if chunk_start is not None and chunk_end is not None:
                    combined = ' '.join(current_chunk_sents)
                    if len(combined.split()) >= min_words:
                        chunks.append({
                            'segment_id': f"{lecture_id}_chunk_{len(chunks)}",
                            'lecture_id': lecture_id,
                            'start': round(chunk_start, 2),
                            'end': round(chunk_end, 2),
                            'raw_text': combined,
                            'word_count': len(combined.split()),
                        })
                current_chunk_sents = []
                word_count = 0
                chunk_start = None
                chunk_end = None

            current_chunk_sents.append(sent_text)
            word_count += wc
            if chunk_start is None:
                chunk_start = sent_start
            chunk_end = sent_end

        # Last chunk
        if current_chunk_sents and chunk_start is not None and chunk_end is not None:
            combined = ' '.join(current_chunk_sents)
            if len(combined.split()) >= min_words:
                chunks.append({
                    'segment_id': f"{lecture_id}_chunk_{len(chunks)}",
                    'lecture_id': lecture_id,
                    'start': round(chunk_start, 2),
                    'end': round(chunk_end, 2),
                    'raw_text': combined,
                    'word_count': len(combined.split()),
                })

        return chunks

    def _match_frames_to_segments(self):
        """Match classified frames to segments by timestamp proximity"""
        frame_paths = self.video_metadata.get('frame_paths', [])
        if not frame_paths:
            return

        # Try to extract timestamp from frame filename (frame_000005.jpg => frame 5)
        # Then estimate timestamp based on video duration and frame spacing
        transcript = self.video_metadata.get('transcript', {})
        total_duration = 0
        whisper_segs = transcript.get('segments', [])
        if whisper_segs:
            total_duration = whisper_segs[-1].get('end', 0)

        frame_count = len(frame_paths)
        buffer = 5.0  # seconds buffer

        for seg in self.segments:
            seg_start = seg.get('start', 0) - buffer
            seg_end = seg.get('end', 0) + buffer
            matched = []

            for i, fp in enumerate(frame_paths):
                # Estimate frame timestamp (evenly distributed)
                if frame_count > 1 and total_duration > 0:
                    frame_ts = (i / (frame_count - 1)) * total_duration
                else:
                    frame_ts = 0

                if seg_start <= frame_ts <= seg_end:
                    matched.append(fp)

            seg['matched_frames'] = matched[:5]  # Max 5 frames per segment
            seg['visual_tags'] = list(set(
                self.frame_classifications.get(fp, {}).get('category', 'Unknown')
                for fp in matched[:5]
            ))

    def _generate_html(self, doc_data: Dict) -> Path:
        """Generate HTML document from data"""
        html_path = self.output_dir / "document.html"

        segments_html = ""
        for i, seg in enumerate(doc_data.get('segments', []), 1):
            start_min = int(seg['start'] // 60)
            start_sec = int(seg['start'] % 60)
            end_min = int(seg['end'] // 60)
            end_sec = int(seg['end'] % 60)

            visual_badges = ""
            for tag in seg.get('visual_tags', []):
                visual_badges += f'<span class="badge">{tag}</span> '

            images_html = ""
            for img in seg.get('images', []):
                img_name = Path(img['path']).name
                images_html += f'''
                <div class="frame-card">
                    <div class="frame-category">{img['category']}</div>
                    <div class="frame-info">Confidence: {img['confidence']:.1%}</div>
                </div>'''

            quality_pct = seg.get('quality_score', 0) * 100

            segments_html += f'''
            <div class="segment-card" id="segment-{i}">
                <div class="segment-header">
                    <h3>Segment {i}</h3>
                    <span class="timestamp">{start_min}:{start_sec:02d} – {end_min}:{end_sec:02d}</span>
                </div>
                <div class="visual-tags">{visual_badges}</div>
                <div class="segment-summary">{seg.get('summary', 'No summary available.')}</div>
                {f'<div class="segment-frames">{images_html}</div>' if images_html else ''}
                <div class="quality-bar">
                    <div class="quality-fill" style="width: {quality_pct}%"></div>
                    <span class="quality-label">Quality: {quality_pct:.0f}%</span>
                </div>
            </div>'''

        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{doc_data.get('title', 'Lecture Summary')}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-indigo: #818cf8;
            --accent-green: #34d399;
            --border-color: rgba(148, 163, 184, 0.1);
            --gradient: linear-gradient(135deg, #38bdf8, #818cf8);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            line-height: 1.7;
            padding: 2rem;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .doc-header {{
            text-align: center;
            padding: 3rem 2rem;
            margin-bottom: 2rem;
            background: var(--bg-secondary);
            border-radius: 16px;
            border: 1px solid var(--border-color);
        }}
        .doc-header h1 {{
            font-size: 2rem;
            font-weight: 700;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }}
        .doc-header .meta {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        .overall-summary {{
            padding: 1.5rem 2rem;
            background: var(--bg-secondary);
            border-radius: 12px;
            border-left: 4px solid var(--accent-cyan);
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }}
        .segment-card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border-color);
            transition: border-color 0.3s;
        }}
        .segment-card:hover {{ border-color: var(--accent-cyan); }}
        .segment-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }}
        .segment-header h3 {{ font-size: 1.1rem; font-weight: 600; }}
        .timestamp {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--accent-cyan);
            background: rgba(56, 189, 248, 0.1);
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
        }}
        .badge {{
            display: inline-block;
            font-size: 0.75rem;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            background: rgba(129, 140, 248, 0.15);
            color: var(--accent-indigo);
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }}
        .segment-summary {{ color: var(--text-secondary); font-size: 0.92rem; margin: 0.75rem 0; }}
        .quality-bar {{
            position: relative;
            height: 6px;
            background: rgba(148, 163, 184, 0.1);
            border-radius: 3px;
            margin-top: 1rem;
            overflow: hidden;
        }}
        .quality-fill {{
            height: 100%;
            background: var(--gradient);
            border-radius: 3px;
            transition: width 0.5s;
        }}
        .quality-label {{
            position: absolute;
            right: 0;
            top: -18px;
            font-size: 0.7rem;
            color: var(--text-secondary);
        }}
        .frame-card {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.05);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            margin: 0.25rem;
            font-size: 0.8rem;
        }}
        .frame-category {{ color: var(--accent-green); font-weight: 500; }}
        .frame-info {{ color: var(--text-secondary); font-size: 0.75rem; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="doc-header">
            <h1>{doc_data.get('title', 'Lecture Summary')}</h1>
            <p class="meta">{doc_data.get('total_segments', 0)} segments • Generated {doc_data.get('generated_at', '')}</p>
        </div>
        <div class="overall-summary">
            <strong>Overall Summary</strong><br>
            {doc_data.get('overall_summary', 'No summary available.')}
        </div>
        {segments_html}
    </div>
</body>
</html>'''

        with open(str(html_path), 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path

    def _generate_markdown(self, doc_data: Dict) -> Path:
        """Generate Markdown document"""
        md_path = self.output_dir / "document.md"

        lines = [
            f"# {doc_data.get('title', 'Lecture Summary')}",
            "",
            f"*Generated: {doc_data.get('generated_at', '')}*",
            f"*Segments: {doc_data.get('total_segments', 0)}*",
            "",
            "## Overall Summary",
            "",
            doc_data.get('overall_summary', 'No summary.'),
            "",
            "---",
            "",
        ]

        for i, seg in enumerate(doc_data.get('segments', []), 1):
            start_min = int(seg['start'] // 60)
            start_sec = int(seg['start'] % 60)
            end_min = int(seg['end'] // 60)
            end_sec = int(seg['end'] % 60)

            lines.append(f"### Segment {i} [{start_min}:{start_sec:02d} – {end_min}:{end_sec:02d}]")
            lines.append("")

            tags = seg.get('visual_tags', [])
            if tags:
                lines.append(f"**Visual Elements:** {', '.join(tags)}")
                lines.append("")

            lines.append(seg.get('summary', 'No summary.'))
            lines.append("")

            quality = seg.get('quality_score', 0) * 100
            lines.append(f"*Quality Score: {quality:.0f}%*")
            lines.append("")
            lines.append("---")
            lines.append("")

        with open(str(md_path), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return md_path

    def _generate_pdf(self, doc_data: Dict) -> Path:
        """Generate PDF using fpdf2 natively"""
        import fpdf
        # fpdf2 uses the `fpdf` namespace
        from fpdf import FPDF
        
        class LecturePDF(FPDF):
            def header(self):
                self.set_font("helvetica", 'B', 10)
                self.cell(0, 10, 'Lecture Summary Document', border=False, align='R')
                self.ln(10)
                
            def footer(self):
                self.set_y(-15)
                self.set_font("helvetica", 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', align='C')

        pdf = LecturePDF()
        pdf.add_page()
        
        # Replace non-latin1 characters with safe equivalents
        def safe_text(txt):
            return txt.encode('latin-1', 'replace').decode('latin-1').replace('?', '-')
        
        # Title
        pdf.set_font("helvetica", "B", 16)
        pdf.multi_cell(0, 10, safe_text(doc_data.get('title', 'Lecture Summary')), align="C")
        
        # Meta
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(100, 100, 100)
        meta_txt = f"{doc_data.get('total_segments', 0)} segments   -   Generated {doc_data.get('generated_at', '')}"
        pdf.cell(0, 10, safe_text(meta_txt), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)
        
        # Overall Summary
        pdf.set_font("helvetica", "B", 13)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "Overall Summary", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("helvetica", "", 11)
        pdf.multi_cell(0, 6, safe_text(doc_data.get('overall_summary', 'No summary.')))
        pdf.ln(10)
        
        # Loop over segments
        for i, seg in enumerate(doc_data.get('segments', []), 1):
            start_min = int(seg['start'] // 60)
            start_sec = int(seg['start'] % 60)
            end_min = int(seg['end'] // 60)
            end_sec = int(seg['end'] % 60)
            
            # Segment Title
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 8, f"Segment {i} [{start_min}:{start_sec:02d} - {end_min}:{end_sec:02d}]", new_x="LMARGIN", new_y="NEXT")
            
            # Tags
            tags = seg.get('visual_tags', [])
            if tags:
                pdf.set_font("helvetica", "I", 10)
                pdf.set_text_color(50, 50, 200)
                pdf.cell(0, 6, safe_text(f"Visual Elements: {', '.join(tags)}"), new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)
            
            # Text
            pdf.set_font("helvetica", "", 11)
            pdf.multi_cell(0, 6, safe_text(seg.get('summary', 'No summary.')))
            
            # Quality score
            pdf.set_font("helvetica", "I", 9)
            pdf.set_text_color(120, 120, 120)
            qs = seg.get('quality_score', 0) * 100
            pdf.cell(0, 6, f"Quality Score: {qs:.0f}%", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            
            pdf.ln(6)
        
        pdf_path = self.output_dir / "document.pdf"
        pdf.output(str(pdf_path))
        return pdf_path
