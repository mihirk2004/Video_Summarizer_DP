#!/usr/bin/env python3
"""
Step 1: Prepare data for BART fine-tuning
Loads annotated JSONs + topic segmentation boundaries → creates topical segments with visual tags

Usage:
    python scripts/text_processing/05_prepare_bart_data.py          # Full run
    python scripts/text_processing/05_prepare_bart_data.py --test   # Test mode (3 lectures)
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


class BARTDataPreparator:
    """Prepare lecture data for BART summarization training"""

    def __init__(self, config: Dict, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.logger = setup_logger("bart_data_prep")

        # Paths
        self.annotations_dir = Path(config['paths']['data']['annotations'])
        self.output_dir = Path(config['paths']['data'].get('bart_dataset', 'data/processed/bart_dataset'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Topic segmentation
        self.topic_seg_dir = Path(config['paths']['models']['topic'])

        # Data prep config
        prep = config.get('data_prep', {})
        self.min_words = prep.get('min_segment_words', 50)
        self.max_words = prep.get('max_segment_words', 800)
        self.chunk_size = prep.get('fallback_chunk_size', 400)
        self.visual_tags_list = prep.get('visual_tags',
            ['equation', 'diagram', 'code', 'slide', 'graph'])

        self.logger.info(f"BARTDataPreparator initialized (test_mode={test_mode})")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_annotations(self) -> List[Dict]:
        """Load all annotation JSON files"""
        files = sorted(self.annotations_dir.glob("*_annotated.json"))
        if self.test_mode:
            files = files[:3]
        self.logger.info(f"Loading {len(files)} annotation files ...")

        annotations = []
        for fp in tqdm(files, desc="Loading annotations"):
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    annotations.append(json.load(f))
            except Exception as e:
                self.logger.warning(f"Error loading {fp.name}: {e}")

        self.logger.info(f"Successfully loaded {len(annotations)} annotations")
        return annotations

    def load_topic_boundaries(self) -> Optional[Dict]:
        """Load topic segmentation results from Phase 2"""
        seg_path = self.topic_seg_dir / "segmentation_results.json"
        if not seg_path.exists():
            self.logger.warning(f"segmentation_results.json not found – using fallback chunking")
            return None

        self.logger.info("Loading topic segmentation results …")
        with open(seg_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        n_lectures = len(results.get('boundaries', {}))
        self.logger.info(f"Loaded boundaries for {n_lectures} lectures")
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _boundary_timestamps(self, seg_results: Optional[Dict], lecture_id: str) -> List[float]:
        """Return sorted boundary timestamps for a lecture"""
        if seg_results is None:
            return []
        boundaries = seg_results.get('boundaries', {}).get(lecture_id, [])
        return sorted(b['timestamp'] for b in boundaries)

    def _extract_visual_tags(self, lecture: Dict, start: float, end: float) -> List[str]:
        """Get visual content tags for a time range from frame / annotation data"""
        tags = set()

        # Collect frames from both locations
        frames = []

        # Location 1: processing.frames (video processing output)
        proc_frames = lecture.get("processing", {}).get("frames", [])
        if isinstance(proc_frames, list):
            frames.extend(proc_frames)

        # Location 2: annotations.frames (annotation GUI output — annotations is a DICT)
        ann = lecture.get("annotations", {})
        if isinstance(ann, dict):
            ann_frames = ann.get("frames", [])
            if isinstance(ann_frames, list):
                frames.extend(ann_frames)

        for frame in frames:
            ts = frame.get("timestamp", -1)
            if not (start <= ts <= end):
                continue

            # concepts is a list of label strings like "Equations", "Computer_Code", etc.
            for concept in frame.get("concepts", []):
                if isinstance(concept, str) and concept in self.visual_tags_list:
                    tags.add(concept)

        return sorted(tags)

    @staticmethod
    def _format_input(text: str, visual_tags: List[str]) -> str:
        """Build formatted model input with special tokens"""
        out = f"[TRANSCRIPT] {text.strip()}"
        if visual_tags:
            out += f" [VISUAL] {', '.join(visual_tags)}"
        return out

    # ------------------------------------------------------------------
    # Segment creation
    # ------------------------------------------------------------------
    def _segments_from_boundaries(self, lecture: Dict, boundaries: List[float]) -> List[Dict]:
        """Group Whisper segments between topic boundary timestamps"""
        transcript = lecture.get("processing", {}).get("transcript", {})
        whisper_segs = transcript.get("segments", [])
        if not whisper_segs:
            return []

        lecture_id = lecture.get("video_id", "unknown")
        last_end = max(s.get("end", 0) for s in whisper_segs)

        # Intervals: [0, b1), [b1, b2), …, [bN, last_end]
        starts = [0.0] + boundaries
        ends = boundaries + [last_end + 1.0]

        segments = []
        for idx, (seg_s, seg_e) in enumerate(zip(starts, ends)):
            texts, actual_s, actual_e = [], None, None
            for ws in whisper_segs:
                mid = (ws["start"] + ws["end"]) / 2
                txt = ws.get("text", "").strip()
                if seg_s <= mid < seg_e and txt:
                    texts.append(txt)
                    if actual_s is None:
                        actual_s = ws["start"]
                    actual_e = ws["end"]

            if not texts:
                continue

            combined = " ".join(texts)
            wc = len(combined.split())
            if wc < self.min_words:
                continue
            if wc > self.max_words:
                combined = " ".join(combined.split()[:self.max_words])
                wc = self.max_words

            vis = self._extract_visual_tags(lecture, actual_s or seg_s, actual_e or seg_e)
            segments.append({
                "segment_id": f"{lecture_id}_seg_{idx}",
                "lecture_id": lecture_id,
                "start": round(actual_s or seg_s, 2),
                "end": round(actual_e or seg_e, 2),
                "raw_text": combined,
                "formatted_input": self._format_input(combined, vis),
                "visual_tags": vis,
                "word_count": wc,
                "topic_id": idx,
            })
        return segments

    def _segments_by_chunking(self, lecture: Dict) -> List[Dict]:
        """Fallback: split transcript into fixed-size word chunks"""
        transcript = lecture.get("processing", {}).get("transcript", {})
        full_text = transcript.get("text", "").strip()
        if not full_text or len(full_text.split()) < self.min_words:
            return []

        lecture_id = lecture.get("video_id", "unknown")
        vis = self._extract_visual_tags(lecture, 0, 99999)
        words = full_text.split()
        segments = []

        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            wc = len(chunk.split())
            if wc < self.min_words:
                continue
            segments.append({
                "segment_id": f"{lecture_id}_chunk_{i // self.chunk_size}",
                "lecture_id": lecture_id,
                "start": 0.0,
                "end": 0.0,
                "raw_text": chunk,
                "formatted_input": self._format_input(chunk, vis),
                "visual_tags": vis,
                "word_count": wc,
                "topic_id": -1,
            })
        return segments

    def _full_transcript_entry(self, lecture: Dict) -> Optional[Dict]:
        """Create a full-lecture entry (used later for lecture-level inference)"""
        transcript = lecture.get("processing", {}).get("transcript", {})
        full_text = transcript.get("text", "").strip()
        if not full_text:
            return None

        lecture_id = lecture.get("video_id", "unknown")
        segs = transcript.get("segments", [])
        start = segs[0]["start"] if segs else 0
        end = segs[-1]["end"] if segs else 0
        vis = self._extract_visual_tags(lecture, start, end)

        return {
            "lecture_id": lecture_id,
            "text": full_text,
            "formatted_input": self._format_input(full_text, vis),
            "visual_tags": vis,
            "word_count": len(full_text.split()),
            "duration": round(end - start, 2),
        }

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def prepare(self) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info("Starting BART Data Preparation")
        self.logger.info("=" * 60)

        annotations = self.load_annotations()
        seg_results = self.load_topic_boundaries()

        all_segments: List[Dict] = []
        full_transcripts: List[Dict] = []
        skipped = 0

        for lecture in tqdm(annotations, desc="Building segments"):
            lid = lecture.get("video_id", "unknown")
            boundaries = self._boundary_timestamps(seg_results, lid)

            if boundaries:
                segs = self._segments_from_boundaries(lecture, boundaries)
            else:
                segs = self._segments_by_chunking(lecture)

            if segs:
                all_segments.extend(segs)
            else:
                skipped += 1

            entry = self._full_transcript_entry(lecture)
            if entry:
                full_transcripts.append(entry)

        # Statistics
        wc = [s['word_count'] for s in all_segments] if all_segments else [0]
        stats = {
            "total_lectures": len(annotations),
            "total_segments": len(all_segments),
            "skipped_lectures": skipped,
            "full_transcripts": len(full_transcripts),
            "avg_words_per_segment": round(sum(wc) / max(len(wc), 1), 1),
            "min_words": min(wc),
            "max_words": max(wc),
        }

        self.logger.info(f"Segments created: {stats['total_segments']}")
        self.logger.info(f"Avg words/segment: {stats['avg_words_per_segment']}")
        self.logger.info(f"Skipped lectures: {skipped}")

        output = {
            "segments": all_segments,
            "full_transcripts": full_transcripts,
            "statistics": stats,
        }

        out_path = self.output_dir / "prepared_segments.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved → {out_path}")

        # Sanity checks
        assert len(all_segments) > 0, "No segments created! Check your annotation data."
        self.logger.info("✅ Data preparation complete!")
        return output


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare data for BART fine-tuning")
    parser.add_argument("--test", action="store_true", help="Test mode – process 3 lectures only")
    args = parser.parse_args()

    config = config_loader.load_all()
    preparator = BARTDataPreparator(config, test_mode=args.test)
    result = preparator.prepare()

    print(f"\n{'=' * 50}")
    print("Data Preparation Complete!")
    print(f"  Segments:          {result['statistics']['total_segments']}")
    print(f"  Full transcripts:  {result['statistics']['full_transcripts']}")
    print(f"  Avg words/segment: {result['statistics']['avg_words_per_segment']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
