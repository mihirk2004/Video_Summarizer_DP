#!/usr/bin/env python3
"""
Phase 4 — Step 1: Prepare Multimodal Dataset
Maps classified frames → annotation timestamps → BART segments.
Generates visually-aware target summaries using Mistral LLM.

Input:
    - data/frames_reorganized/{category}/{split}/ (classified frames)
    - data/annotations/lecture_XXX_annotated.json  (frame timestamps)
    - data/processed/bart_dataset/dataset_with_summaries.json (text segments)

Output:
    - data/processed/multimodal_dataset/multimodal_segments.json

Usage:
    python scripts/text_processing/11_prepare_multimodal_data.py              # Full run
    python scripts/text_processing/11_prepare_multimodal_data.py --test       # 5 lectures
    python scripts/text_processing/11_prepare_multimodal_data.py --skip-llm   # Skip LLM summaries
"""
import sys
import gc
import json
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


class MultimodalDataPreparator:
    """Build multimodal dataset: segments + image paths + timestamps + visual summaries"""

    def __init__(self, config: Dict, test_mode: bool = False, skip_llm: bool = False, resume: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.skip_llm = skip_llm
        self.resume = resume
        self.logger = setup_logger("multimodal_data_prep")

        # Paths
        self.annotations_dir = Path(config['paths']['data']['annotations'])
        self.raw_frames_dir = Path(config['paths']['data']['raw_frames'])
        self.frames_reorg_dir = Path("data/frames_reorganized")
        self.bart_dataset_path = Path(config['paths']['data'].get(
            'bart_dataset', 'data/processed/bart_dataset')) / "dataset_with_summaries.json"
        self.output_dir = Path(config['paths'].get('multimodal', {}).get(
            'dataset', 'data/processed/multimodal_dataset'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Config values with sensible defaults
        mm_cfg = config.get('multimodal_data', {})
        self.timestamp_buffer = mm_cfg.get('timestamp_buffer', 5.0)
        self.max_frames_per_seg = mm_cfg.get('max_frames_per_segment', 5)
        self.skip_categories = set(mm_cfg.get('skip_categories', ['Instructor_Writing']))
        self.caption_categories = set(mm_cfg.get('caption_categories', [
            'Equation', 'Diagrams', 'Computer_Code', 'Graph_Chart',
            'Table', 'Flow_diagram', 'Slide_Presentation', 'Question'
        ]))

        # Data prep config
        prep = config.get('data_prep', {})
        self.min_words = prep.get('min_segment_words', 50)
        self.max_words = prep.get('max_segment_words', 800)
        self.chunk_size = prep.get('fallback_chunk_size', 400)

        # LLM model (lazy-loaded)
        self._llm_model = None
        self._llm_tokenizer = None

        # Metadata for output
        self.metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_mode': test_mode,
            'skip_llm': skip_llm,
            'min_words': self.min_words,
            'max_words': self.max_words,
            'timestamp_buffer': self.timestamp_buffer,
            'max_frames_per_seg': self.max_frames_per_seg,
        }

    # ------------------------------------------------------------------
    # Step A: Build Frame Category Map from frames_reorganized
    # ------------------------------------------------------------------
    def build_frame_category_map(self) -> Dict[str, Dict[int, str]]:
        """
        Parse filenames in frames_reorganized to create:
        {lecture_id: {frame_index: category, ...}, ...}

        Filenames are like: lecture_018_frame_000005.jpg
        """
        self.logger.info("Step A: Building frame category map from frames_reorganized/ ...")
        category_map = defaultdict(dict)  # lecture_id -> {frame_idx -> category}
        total_frames = 0

        for category_dir in self.frames_reorg_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
            category = category_dir.name

            if category in self.skip_categories:
                self.logger.info(f"  Skipping category: {category}")
                continue

            for split in ['train', 'val', 'test']:
                split_dir = category_dir / split
                if not split_dir.exists():
                    continue

                for img_file in split_dir.iterdir():
                    if not img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        continue

                    # Parse filename: lecture_018_frame_000005.jpg or lecture_018_frame_000005_1.jpg
                    match = re.match(
                        r'(lecture_\d+)_frame_(\d+)(?:_\d+)?\.(?:jpg|jpeg|png)',
                        img_file.name, re.IGNORECASE
                    )
                    if match:
                        lecture_id = match.group(1)
                        frame_idx = int(match.group(2))
                        # If frame already has a category, keep both (multi-label possible)
                        # But for simplicity, keep the first one found
                        if frame_idx not in category_map[lecture_id]:
                            category_map[lecture_id][frame_idx] = category
                        total_frames += 1

        self.logger.info(f"  Mapped {total_frames} frames across {len(category_map)} lectures")

        # Log category distribution
        cat_counts = defaultdict(int)
        for lec_frames in category_map.values():
            for cat in lec_frames.values():
                cat_counts[cat] += 1
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            self.logger.info(f"    {cat}: {count} frames")

        return dict(category_map)

    # ------------------------------------------------------------------
    # Step B: Build Timestamp Lookup from Annotation JSONs
    # ------------------------------------------------------------------
    def build_timestamp_lookup(self, lecture_ids: set) -> Dict[str, Dict]:
        """
        From annotation JSONs, build:
        {lecture_id: {
            frame_timestamps: {frame_index: timestamp},
            transcript_segments: [{start, end, text}, ...],
            full_text: "..."
        }}
        """
        self.logger.info("Step B: Building timestamp lookup from annotations ...")
        lookup = {}

        for ann_file in sorted(self.annotations_dir.glob("*_annotated.json")):
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                self.logger.warning(f"  Error loading {ann_file.name}: {e}")
                continue

            lecture_id = data.get('video_id', '')
            if lecture_id not in lecture_ids:
                continue

            processing = data.get('processing', {})

            # Frame timestamps
            frame_ts = {}
            for frame in processing.get('frames', []):
                idx = frame.get('index', frame.get('frame_index', -1))
                ts = frame.get('timestamp', -1)
                if idx >= 0 and ts >= 0:
                    frame_ts[idx] = ts

            # Transcript segments (Whisper-level)
            whisper_segs = processing.get('transcript', {}).get('segments', [])
            full_text = processing.get('transcript', {}).get('text', '')

            lookup[lecture_id] = {
                'frame_timestamps': frame_ts,
                'whisper_segments': whisper_segs,
                'full_text': full_text,
                'total_duration': whisper_segs[-1]['end'] if whisper_segs else 0,
            }

        self.logger.info(f"  Built timestamp lookup for {len(lookup)} lectures")
        return lookup

    # ------------------------------------------------------------------
    # Step C: Create Multimodal Segments
    # ------------------------------------------------------------------
    def create_multimodal_segments(self, category_map, timestamp_lookup):
        self.logger.info("Step C: Creating multimodal segments ...")

        # Checkpoint file
        checkpoint_path = self.output_dir / "multimodal_segments_checkpoint.json"

        # Load existing data if resuming
        processed_lectures = set()
        all_segments = []
        if self.resume and checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                processed_lectures = set(checkpoint.get('processed_lectures', []))
                all_segments = checkpoint.get('segments', [])
                self.logger.info(f"Resuming from checkpoint: {len(all_segments)} segments already, "
                                f"{len(processed_lectures)} lectures processed")
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")

        # Topic boundaries (unchanged)
        topic_seg_path = Path(self.config['paths']['models']['topic']) / "segmentation_results.json"
        topic_boundaries = None
        if topic_seg_path.exists():
            with open(topic_seg_path, 'r', encoding='utf-8') as f:
                topic_boundaries = json.load(f)
            self.logger.info("  Loaded topic segmentation boundaries")
        else:
            self.logger.info("  No topic boundaries found — using word-chunking fallback")

        lectures_to_process = sorted(
            set(category_map.keys()) & set(timestamp_lookup.keys())
        )
        if self.test_mode:
            lectures_to_process = lectures_to_process[:5]

        if self.resume:
            # Remove already processed lectures
            lectures_to_process = [lec for lec in lectures_to_process if lec not in processed_lectures]

        self.logger.info(f"  Processing {len(lectures_to_process)} lectures with both frames and timestamps")

        stats = defaultdict(int)

        for lecture_id in tqdm(lectures_to_process, desc="Building segments"):
            lec_data = timestamp_lookup[lecture_id]
            lec_frames = category_map.get(lecture_id, {})
            whisper_segs = lec_data['whisper_segments']

            if not whisper_segs:
                stats['skipped_no_transcript'] += 1
                continue

            boundaries = self._get_boundaries(topic_boundaries, lecture_id)
            segments = self._build_segments(lecture_id, whisper_segs, boundaries)

            for seg in segments:
                matched_frames = self._match_frames_to_segment(
                    seg, lec_frames, lec_data['frame_timestamps']
                )
                seg['image_paths'] = [f['path'] for f in matched_frames]
                seg['image_categories'] = [f['category'] for f in matched_frames]
                seg['image_timestamps'] = [f['timestamp'] for f in matched_frames]
                seg['image_frame_indices'] = [f['frame_index'] for f in matched_frames]
                seg['visual_tags'] = sorted(set(seg['image_categories']))
                seg['has_visuals'] = len(matched_frames) > 0

                if matched_frames:
                    stats['segments_with_images'] += 1
                    stats['total_image_pairs'] += len(matched_frames)
                stats['total_segments'] += 1

            all_segments.extend(segments)
            processed_lectures.add(lecture_id)

            # Save checkpoint after each lecture
            try:
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'processed_lectures': list(processed_lectures),
                        'segments': all_segments
                    }, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to save checkpoint: {e}")

        # Clean up checkpoint if everything succeeded
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            self.logger.info("Checkpoint file removed after successful completion.")

        self.logger.info(f"  Created {stats['total_segments']} segments")
        self.logger.info(f"  Segments with images: {stats['segments_with_images']}")
        self.logger.info(f"  Total text-image pairs: {stats['total_image_pairs']}")

        return all_segments

    def _get_boundaries(self, topic_boundaries: Optional[Dict], lecture_id: str) -> List[float]:
        """Get sorted boundary timestamps for a lecture"""
        if topic_boundaries is None:
            return []
        boundaries = topic_boundaries.get('boundaries', {}).get(lecture_id, [])
        return sorted(b['timestamp'] for b in boundaries)

    def _build_segments(
        self,
        lecture_id: str,
        whisper_segs: List[Dict],
        boundaries: List[float]
    ) -> List[Dict]:
        """Build text segments from Whisper segments using topic boundaries or chunking"""
        if boundaries:
            return self._segments_from_boundaries(lecture_id, whisper_segs, boundaries)
        else:
            return self._segments_by_chunking(lecture_id, whisper_segs)

    def _segments_from_boundaries(
        self,
        lecture_id: str,
        whisper_segs: List[Dict],
        boundaries: List[float]
    ) -> List[Dict]:
        """Group Whisper segments between topic boundary timestamps"""
        last_end = max(s.get('end', 0) for s in whisper_segs)
        starts = [0.0] + boundaries
        ends = boundaries + [last_end + 1.0]

        segments = []
        for idx, (seg_s, seg_e) in enumerate(zip(starts, ends)):
            texts, actual_s, actual_e = [], None, None
            for ws in whisper_segs:
                mid = (ws['start'] + ws['end']) / 2
                txt = ws.get('text', '').strip()
                if seg_s <= mid < seg_e and txt:
                    texts.append(txt)
                    if actual_s is None:
                        actual_s = ws['start']
                    actual_e = ws['end']

            if not texts:
                continue

            combined = " ".join(texts)
            wc = len(combined.split())
            if wc < self.min_words:
                continue
            if wc > self.max_words:
                # Truncate at sentence boundary
                combined = self._truncate_by_sentence(combined, self.max_words)
                wc = len(combined.split())

            segments.append({
                'segment_id': f"{lecture_id}_seg_{idx}",
                'lecture_id': lecture_id,
                'start': round(actual_s or seg_s, 2),
                'end': round(actual_e or seg_e, 2),
                'raw_text': combined,
                'word_count': wc,
                'topic_id': idx,
            })
        return segments

    def _segments_by_chunking(self, lecture_id: str, whisper_segs: List[Dict]) -> List[Dict]:
      """Fallback: chunk by sentences and map timestamps, with safe timestamp handling."""
      # Build sentences with timestamps
      sentences = []
      current_sent = []
      current_start = None
      current_end = None

      for ws in whisper_segs:
          txt = ws.get('text', '').strip()
          if not txt:
              continue
          start = ws.get('start')
          end = ws.get('end')
          if start is None or end is None:
              self.logger.warning(f"Whisper segment missing timestamp in {lecture_id}, skipping")
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

      # Group sentences into chunks
      chunks = []
      current_chunk_sentences = []
      current_word_count = 0
      chunk_start = None
      chunk_end = None

      for sent_start, sent_end, sent_text in sentences:
          # Skip any sentence with missing timestamps
          if sent_start is None or sent_end is None:
              self.logger.warning(f"Skipping sentence with missing timestamp in {lecture_id}")
              continue

          wc = len(sent_text.split())
          # If adding this sentence would exceed chunk size and we already have something,
          # finish the current chunk.
          if current_word_count + wc > self.chunk_size and current_chunk_sentences:
              # Only create chunk if we have valid timestamps
              if chunk_start is not None and chunk_end is not None:
                  combined = ' '.join(current_chunk_sentences)
                  if len(combined.split()) >= self.min_words:
                      chunks.append({
                          'segment_id': f"{lecture_id}_chunk_{len(chunks)}",
                          'lecture_id': lecture_id,
                          'start': round(chunk_start, 2),
                          'end': round(chunk_end, 2),
                          'raw_text': combined,
                          'word_count': len(combined.split()),
                          'topic_id': -1,
                      })
              # Reset chunk accumulators
              current_chunk_sentences = []
              current_word_count = 0
              chunk_start = None
              chunk_end = None

          # Add the current sentence to the chunk
          current_chunk_sentences.append(sent_text)
          current_word_count += wc
          if chunk_start is None:
              chunk_start = sent_start
          chunk_end = sent_end

      # Handle the last chunk
      if current_chunk_sentences:
          if chunk_start is not None and chunk_end is not None:
              combined = ' '.join(current_chunk_sentences)
              if len(combined.split()) >= self.min_words:
                  chunks.append({
                      'segment_id': f"{lecture_id}_chunk_{len(chunks)}",
                      'lecture_id': lecture_id,
                      'start': round(chunk_start, 2),
                      'end': round(chunk_end, 2),
                      'raw_text': combined,
                      'word_count': len(combined.split()),
                      'topic_id': -1,
                  })
          else:
              self.logger.warning(f"Skipping final chunk in lecture {lecture_id} due to missing timestamps")

      return chunks

    @staticmethod
    def _truncate_by_sentence(text: str, max_words: int) -> str:
        """Truncate text to at most max_words, cutting at last sentence boundary."""
        words = text.split()
        if len(words) <= max_words:
            return text
        # Find last sentence boundary within max_words
        # Simple: split into sentences (by .!? followed by space or end)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        truncated = []
        total_words = 0
        for sent in sentences:
            sent_words = len(sent.split())
            if total_words + sent_words <= max_words:
                truncated.append(sent)
                total_words += sent_words
            else:
                # If first sentence already exceeds, just take prefix
                if not truncated:
                    truncated.append(' '.join(words[:max_words]))
                break
        return ' '.join(truncated).strip()

    def _match_frames_to_segment(
        self,
        segment: Dict,
        lec_frames: Dict[int, str],  # frame_index -> category
        frame_timestamps: Dict[int, float],  # frame_index -> timestamp
    ) -> List[Dict]:
        """Find classified frames that fall within segment's time window"""
        seg_start = segment['start'] - self.timestamp_buffer
        seg_end = segment['end'] + self.timestamp_buffer
        lecture_id = segment['lecture_id']

        matched = []
        for frame_idx, category in lec_frames.items():
            ts = frame_timestamps.get(frame_idx, -1)
            if ts < 0:
                continue

            if seg_start <= ts <= seg_end:
                # Build the raw_frames path and check existence
                frame_path = self.raw_frames_dir / lecture_id / f"frame_{frame_idx:06d}.jpg"
                if not frame_path.exists():
                    continue
                matched.append({
                    'frame_index': frame_idx,
                    'timestamp': ts,
                    'category': category,
                    'path': str(frame_path),
                })

        # Sort by timestamp, take top-K with diverse selection
        matched.sort(key=lambda x: x['timestamp'])
        if len(matched) > self.max_frames_per_seg:
            matched = self._diverse_select(matched, self.max_frames_per_seg)

        return matched

    @staticmethod
    def _diverse_select(frames: List[Dict], k: int) -> List[Dict]:
        """Select K frames preferring category diversity and temporal spread."""
        if len(frames) <= k:
            return frames

        selected = []
        seen_categories = set()

        # First round: pick one per category
        for f in frames:
            if f['category'] not in seen_categories and len(selected) < k:
                selected.append(f)
                seen_categories.add(f['category'])

        # If we still need more, fill with the farthest frames in time
        if len(selected) < k:
            remaining = [f for f in frames if f not in selected]
            # To maximize spread, we could pick frames that are farthest apart,
            # but for simplicity, just take the first few remaining
            selected.extend(remaining[:k - len(selected)])

        # Sort back by timestamp
        selected.sort(key=lambda x: x['timestamp'])
        return selected

    # ------------------------------------------------------------------
    # Step D: Generate Visually-Aware Target Summaries (Mistral LLM)
    # ------------------------------------------------------------------
    def generate_visual_summaries(self, segments: List[Dict], resume: bool = False) -> List[Dict]:
        """Use Mistral to generate summaries, with checkpointing."""
        if self.skip_llm:
            self.logger.info("Step D: Skipping LLM summary generation (--skip-llm)")
            for seg in segments:
                seg['target_summary'] = ""
                seg['formatted_input'] = self._format_input(seg)
            return segments

        self.logger.info("Step D: Generating visually-aware target summaries with Mistral ...")

        checkpoint_path = self.output_dir / "summary_checkpoint.json"
        processed = {}

        # Load existing summaries if resuming
        if resume and checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    processed = json.load(f)
                self.logger.info(f"Loaded summaries for {len(processed)} segments from checkpoint.")
            except Exception as e:
                self.logger.warning(f"Could not load summary checkpoint: {e}")

        # Separate segments that already have summaries
        need_summary = []
        for seg in segments:
            seg_id = seg['segment_id']
            if seg_id in processed:
                seg['target_summary'] = processed[seg_id]['target_summary']
                seg['formatted_input'] = processed[seg_id]['formatted_input']
            else:
                need_summary.append(seg)

        if not need_summary:
            self.logger.info("All summaries already generated.")
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            return segments

        self.logger.info(f"Generating summaries for {len(need_summary)} segments...")

        # Load model
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            self.logger.error("transformers not installed.")
            return segments

        self.logger.info("  Loading Mistral-7B-Instruct (4-bit) ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left'   # crucial for decoder-only models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        self.logger.info("  Mistral loaded ✓")

        batch_size = 6   # adjust to 6 if GPU memory allows
        total = len(need_summary)

        try:
            for i in tqdm(range(0, total, batch_size), desc="LLM Summaries"):
                batch = need_summary[i:i + batch_size]
                prompts = [self._build_visual_summary_prompt(seg) for seg in batch]

                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1200,
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        min_new_tokens=20,
                        do_sample=False,
                        num_beams=2,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.2,
                        length_penalty=1.2,
                        early_stopping=True,
                    )

                prompt_len = inputs['input_ids'].shape[1]
                decoded = tokenizer.batch_decode(
                    outputs[:, prompt_len:], skip_special_tokens=True
                )

                # Update segments and checkpoint
                for j, seg in enumerate(batch):
                    summary = decoded[j].replace("</s>", "").strip()
                    seg['target_summary'] = summary
                    seg['formatted_input'] = self._format_input(seg)
                    processed[seg['segment_id']] = {
                        'target_summary': summary,
                        'formatted_input': seg['formatted_input']
                    }

                # Save checkpoint after each batch
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(processed, f, indent=2)

        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Clean up checkpoint on success
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            self.logger.info("Summary checkpoint removed.")

        self.logger.info("  LLM summary generation complete ✓")
        return segments

    def _build_visual_summary_prompt(self, segment: Dict) -> str:
        """Build a Mistral prompt that encourages visual-aware summaries"""
        text = segment['raw_text']
        words = text.split()
        if len(words) > 400:
            text = " ".join(words[:400])

        # Build visual context string only if visuals exist
        visual_context = ""
        if segment.get('has_visuals') and segment.get('image_categories'):
            visual_items = []
            for cat, ts in zip(segment['image_categories'], segment['image_timestamps']):
                mins = int(ts // 60)
                secs = int(ts % 60)
                visual_items.append(f"{cat.replace('_', ' ')} at {mins}:{secs:02d}")
            visual_context = f"\nVisual elements present in this segment: [{', '.join(visual_items)}]."
            visual_context += "\nYou MUST reference these visual elements naturally in your summary."

        prompt = (
            f"<s>[INST] You are an expert teaching assistant creating lecture notes. "
            f"Summarize this lecture segment in 2-4 concise sentences. "
            f"Focus on main concepts and key technical terms."
            f"{visual_context}\n\n"
            f"TRANSCRIPT:\n{text}\n\n"
            f"SUMMARY:\n[/INST]"
        )
        return prompt

    @staticmethod
    def _format_input(segment: Dict) -> str:
        """Build formatted model input with special tokens"""
        text = segment['raw_text'].strip()
        out = f"[TRANSCRIPT] {text}"

        if segment.get('visual_tags'):
            out += f" [VISUAL] {', '.join(segment['visual_tags'])}"

            # Add per-category tokens
            for cat in segment['visual_tags']:
                token_map = {
                    'Equation': '[EQUATION]',
                    'Diagrams': '[DIAGRAM]',
                    'Computer_Code': '[CODE]',
                    'Graph_Chart': '[GRAPH]',
                    'Slide_Presentation': '[SLIDE]',
                    'Table': '[TABLE]',
                    'Flow_diagram': '[DIAGRAM]',
                    'Question': '[QUESTION]',
                }
                token = token_map.get(cat, '')
                if token:
                    out += f" {token}"
        return out

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------
    def prepare(self) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info("Multimodal Data Preparation Pipeline")
        self.logger.info("=" * 60)

        t_start = time.time()

        # Step A: Build frame category map
        category_map = self.build_frame_category_map()

        # Step B: Build timestamp lookup
        all_lecture_ids = set(category_map.keys())
        timestamp_lookup = self.build_timestamp_lookup(all_lecture_ids)

        # Step C: Create multimodal segments
        segments = self.create_multimodal_segments(category_map, timestamp_lookup)

        # Safety: ensure segments is a list
        if segments is None:
            segments = []
            self.logger.warning("create_multimodal_segments returned None, using empty list.")

        if not segments:
            self.logger.error("No segments created! Check your data.")
            return {}

        # Step D: Generate visually-aware target summaries
        segments = self.generate_visual_summaries(segments, resume=self.resume)

        elapsed = time.time() - t_start

        # Statistics
        seg_with_imgs = sum(1 for s in segments if s.get('has_visuals'))
        total_pairs = sum(len(s.get('image_paths', [])) for s in segments)
        cat_dist = defaultdict(int)
        for s in segments:
            for cat in s.get('image_categories', []):
                cat_dist[cat] += 1
        summaries_generated = sum(1 for s in segments if s.get('target_summary'))

        statistics = {
            'total_segments': len(segments),
            'segments_with_images': seg_with_imgs,
            'segments_text_only': len(segments) - seg_with_imgs,
            'total_image_pairs': total_pairs,
            'category_distribution': dict(cat_dist),
            'summaries_generated': summaries_generated,
            'lectures_processed': len(set(s['lecture_id'] for s in segments)),
            'avg_words_per_segment': round(
                sum(s['word_count'] for s in segments) / len(segments), 1
            ),
            'elapsed_seconds': round(elapsed, 1),
        }

        # Save output with metadata
        output = {
            'metadata': self.metadata,
            'statistics': statistics,
            'segments': segments,
        }

        out_path = self.output_dir / "multimodal_segments.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"Multimodal Dataset Saved → {out_path}")
        self.logger.info(f"  Total segments:       {statistics['total_segments']}")
        self.logger.info(f"  With images:          {statistics['segments_with_images']}")
        self.logger.info(f"  Text-only:            {statistics['segments_text_only']}")
        self.logger.info(f"  Total image pairs:    {statistics['total_image_pairs']}")
        self.logger.info(f"  Summaries generated:  {statistics['summaries_generated']}")
        self.logger.info(f"  Time: {statistics['elapsed_seconds']}s")
        self.logger.info(f"{'=' * 50}")

        return output


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Prepare multimodal dataset (frames + timestamps + summaries)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode — process 5 lectures only")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM summary generation (use for debugging)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    config = config_loader.load_all()
    preparator = MultimodalDataPreparator(
        config, test_mode=args.test, skip_llm=args.skip_llm, resume=args.resume)
    result = preparator.prepare()

    if result:
        stats = result['statistics']
        print(f"\n{'=' * 50}")
        print("Multimodal Data Preparation Complete!")
        print(f"  Segments:       {stats['total_segments']}")
        print(f"  With images:    {stats['segments_with_images']}")
        print(f"  Image pairs:    {stats['total_image_pairs']}")
        print(f"  Summaries:      {stats['summaries_generated']}")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    main()