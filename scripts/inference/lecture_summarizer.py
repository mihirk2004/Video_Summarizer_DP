#!/usr/bin/env python3
"""
Production Lecture Summarizer — clean API for generating summaries
Integrates BART (fast draft) + Mistral-7B LLM (quality enhancement)

Usage as module:
    from scripts.inference.lecture_summarizer import LectureSummarizer

    summarizer = LectureSummarizer(mode="hybrid")     # hybrid | bart | llm
    result = summarizer.summarize("Lecture transcript here...")
    print(result['summary'])

    # Batch
    results = summarizer.summarize_segments(segments)

    # From annotation JSON
    results = summarizer.summarize_lecture("data/annotations/lecture_013_annotated.json")

Usage as CLI:
    python scripts/inference/lecture_summarizer.py --mode hybrid --input "transcript text"
    python scripts/inference/lecture_summarizer.py --mode hybrid --lecture data/annotations/lecture_013_annotated.json
    python scripts/inference/lecture_summarizer.py --mode bart --batch data/processed/bart_dataset/dataset_with_summaries.json
"""
import sys
import gc
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


class LectureSummarizer:
    """
    Production-ready lecture summarizer with 3 modes:
      - 'bart':   Fast, local, ~0.1s/segment  — good for batch processing
      - 'llm':    High-quality via Mistral-7B, ~3s/segment — best quality
      - 'hybrid': BART draft → LLM refine, ~3s/segment — best balance
    """

    MODES = ('bart', 'llm', 'hybrid')

    def __init__(self, mode: str = "hybrid", config: Dict = None,
                 bart_model_path: Optional[str] = None,
                 llm_model_name: Optional[str] = None):
        """
        Args:
            mode:           'bart', 'llm', or 'hybrid'
            config:         Config dict (loaded from config_loader if None)
            bart_model_path: Override BART model path
            llm_model_name:  Override LLM model name
        """
        assert mode in self.MODES, f"mode must be one of {self.MODES}"
        self.mode = mode
        self.config = config or config_loader.load_all()
        self.logger = setup_logger("lecture_summarizer")

        # Paths
        self.model_dir = Path(self.config['paths']['models'].get(
            'bart', 'models/text/bart_summarizer'))
        self.bart_path = bart_model_path or str(self.model_dir / "best_model")
        self.llm_name = llm_model_name or "mistralai/Mistral-7B-Instruct-v0.3"

        # Model state
        self._bart_model = None
        self._bart_tokenizer = None
        self._llm_model = None
        self._llm_tokenizer = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"LectureSummarizer initialized (mode={mode}, device={self.device})")

    # ------------------------------------------------------------------
    # Model loading / unloading
    # ------------------------------------------------------------------
    def _ensure_bart(self):
        if self._bart_model is not None:
            return
        from transformers import AutoTokenizer

        self.logger.info(f"Loading BART from {self.bart_path} …")
        best_path = Path(self.bart_path)

        adapter_config = best_path / "adapter_config.json"
        if adapter_config.exists():
            from transformers import BartForConditionalGeneration
            from peft import PeftModel

            base_name = self.config.get('bart', {}).get(
                'model_name', 'facebook/bart-large-cnn')
            base = BartForConditionalGeneration.from_pretrained(base_name)
            self._bart_tokenizer = AutoTokenizer.from_pretrained(str(best_path))
            base.resize_token_embeddings(len(self._bart_tokenizer))
            self._bart_model = PeftModel.from_pretrained(
                base, str(best_path), is_trainable=False)
        else:
            from transformers import BartForConditionalGeneration
            self._bart_model = BartForConditionalGeneration.from_pretrained(str(best_path))
            self._bart_tokenizer = AutoTokenizer.from_pretrained(str(best_path))

        self._bart_model.eval().to(self.device)
        self._bart_model.generation_config.max_length = None
        self._bart_model.generation_config.min_length = None
        self._bart_model.generation_config.max_new_tokens = None
        self.logger.info("BART loaded ✓")

    def _ensure_llm(self):
        if self._llm_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.logger.info(f"Loading LLM: {self.llm_name} (4-bit) …")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        self._llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        self._llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_name, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.float16,
        )
        if self._llm_tokenizer.pad_token is None:
            self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token
            self._llm_model.config.pad_token_id = self._llm_tokenizer.eos_token_id
        self.logger.info("LLM loaded ✓")

    def unload(self):
        """Free all GPU memory"""
        for attr in ('_bart_model', '_bart_tokenizer', '_llm_model', '_llm_tokenizer'):
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("All models unloaded")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _bart_generate(self, text: str) -> str:
        self._ensure_bart()
        max_target = self.config.get('bart', {}).get('max_target_length', 350)

        inputs = self._bart_tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=1024, padding=True,
        ).to(self.device)

        gen_ids = self._bart_model.generate(
            **inputs,
            max_new_tokens=max_target, min_new_tokens=10,
            num_beams=4, no_repeat_ngram_size=3,
            repetition_penalty=1.2, length_penalty=2.0,
            early_stopping=True,
        )
        return self._bart_tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def _llm_generate(self, text: str, draft: str = None) -> str:
        self._ensure_llm()

        words = text.split()
        if len(words) > 400:
            text = " ".join(words[:400])

        if draft and len(draft.split()) > 8:
            prompt = (
                f"<s>[INST] You are an expert teaching assistant. Refine this draft "
                f"summary of a lecture segment.\n\n"
                f"TRANSCRIPT:\n{text}\n\n"
                f"DRAFT SUMMARY:\n{draft}\n\n"
                f"Fix errors, remove repetition, ensure key terms are included. "
                f"Keep it concise (2-4 sentences). If the draft is gibberish, "
                f"write a fresh summary from the transcript.\n\n"
                f"REFINED SUMMARY:\n[/INST]"
            )
        else:
            prompt = (
                f"<s>[INST] Summarize this lecture segment in 2-4 concise sentences. "
                f"Focus on main concepts and key technical terms.\n\n"
                f"{text}\n[/INST]"
            )

        inputs = self._llm_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1200
        ).to(self.device)

        outputs = self._llm_model.generate(
            **inputs,
            max_new_tokens=250, min_new_tokens=20,
            do_sample=False, num_beams=4,
            no_repeat_ngram_size=3, repetition_penalty=1.2,
            length_penalty=1.2, early_stopping=True,
        )

        prompt_len = inputs['input_ids'].shape[1]
        decoded = self._llm_tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True).strip()
        return decoded.replace("</s>", "").strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def summarize(self, transcript: str, visual_tags: List[str] = None) -> Dict:
        """
        Summarize a single transcript segment.

        Returns:
            {
                'summary': str,
                'bart_draft': str or None,
                'mode': str,
                'time_seconds': float,
            }
        """
        t0 = time.time()
        bart_draft = None

        if self.mode == 'bart':
            summary = self._bart_generate(transcript)

        elif self.mode == 'llm':
            summary = self._llm_generate(transcript)

        elif self.mode == 'hybrid':
            bart_draft = self._bart_generate(transcript)
            # Free BART if we need LLM (GPU memory)
            if self._llm_model is None and self._bart_model is not None:
                del self._bart_model, self._bart_tokenizer
                self._bart_model = self._bart_tokenizer = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            summary = self._llm_generate(transcript, draft=bart_draft)

        return {
            'summary': summary,
            'bart_draft': bart_draft,
            'mode': self.mode,
            'time_seconds': round(time.time() - t0, 3),
        }

    def summarize_segments(self, segments: List[Dict],
                           text_key: str = 'raw_text') -> List[Dict]:
        """
        Summarize a list of segments (batch processing).
        For hybrid mode, runs all BART first, then all LLM (memory efficient).

        Returns list of dicts with 'summary' added to each segment.
        """
        from tqdm import tqdm

        self.logger.info(f"Summarizing {len(segments)} segments (mode={self.mode}) …")
        t0 = time.time()

        if self.mode == 'hybrid':
            # Phase 1: all BART drafts
            self._ensure_bart()
            for seg in tqdm(segments, desc="BART drafts"):
                seg['bart_draft'] = self._bart_generate(seg[text_key])

            # Free BART, load LLM
            del self._bart_model, self._bart_tokenizer
            self._bart_model = self._bart_tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Phase 2: all LLM refinements
            self._ensure_llm()
            for seg in tqdm(segments, desc="LLM refine"):
                seg['summary'] = self._llm_generate(
                    seg[text_key], draft=seg.get('bart_draft'))

        elif self.mode == 'bart':
            self._ensure_bart()
            for seg in tqdm(segments, desc="BART"):
                seg['summary'] = self._bart_generate(seg[text_key])

        elif self.mode == 'llm':
            self._ensure_llm()
            for seg in tqdm(segments, desc="LLM"):
                seg['summary'] = self._llm_generate(seg[text_key])

        elapsed = time.time() - t0
        self.logger.info(f"Done: {len(segments)} segments in {elapsed:.1f}s "
                         f"({elapsed/len(segments):.2f}s/seg)")
        return segments

    def summarize_lecture(self, annotation_path: str,
                          segmentation_results_path: str = None) -> Dict:
        """
        Summarize an entire lecture from its annotation JSON.

        Args:
            annotation_path: Path to lecture_XXX_annotated.json
            segmentation_results_path: Optional path to topic segmentation results

        Returns:
            Dict with per-segment summaries and a full-lecture summary
        """
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        video_id = annotation.get('video_id', Path(annotation_path).stem)
        transcript = annotation.get('processing', {}).get('transcript', {})
        full_text = transcript.get('text', '')

        if not full_text:
            self.logger.error(f"No transcript found in {annotation_path}")
            return {}

        # Try to load topic segments for better chunking
        segments = self._build_segments(full_text, transcript, video_id,
                                         segmentation_results_path)

        self.logger.info(f"Lecture {video_id}: {len(segments)} segments")

        # Summarize each segment
        segments = self.summarize_segments(segments)

        # Build full-lecture summary (concatenate segment summaries)
        all_summaries = [s.get('summary', '') for s in segments if s.get('summary')]
        full_summary = " ".join(all_summaries)

        return {
            'video_id': video_id,
            'segments': segments,
            'full_summary': full_summary,
            'n_segments': len(segments),
        }

    def _build_segments(self, full_text: str, transcript: Dict,
                        video_id: str, seg_results_path: str = None) -> List[Dict]:
        """Build segments from transcript, using topic boundaries if available"""
        segments = []

        # Try topic segmentation results
        if seg_results_path:
            try:
                with open(seg_results_path, 'r') as f:
                    seg_data = json.load(f)

                lecture_segs = [s for s in seg_data.get('segments', [])
                                if s.get('lecture_id') == video_id]
                if lecture_segs:
                    for s in lecture_segs:
                        segments.append({
                            'segment_id': s.get('segment_id', ''),
                            'lecture_id': video_id,
                            'raw_text': s.get('text', s.get('raw_text', '')),
                        })
                    return segments
            except Exception:
                pass

        # Fallback: chunk by word count
        words = full_text.split()
        chunk_size = self.config.get('data_prep', {}).get('fallback_chunk_size', 400)
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) >= 30:
                segments.append({
                    'segment_id': f"{video_id}_chunk_{i // chunk_size}",
                    'lecture_id': video_id,
                    'raw_text': chunk,
                })

        return segments


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Production Lecture Summarizer")
    parser.add_argument("--mode", choices=LectureSummarizer.MODES,
                        default="hybrid", help="Summarization mode")
    parser.add_argument("--input", type=str, help="Direct transcript text to summarize")
    parser.add_argument("--lecture", type=str,
                        help="Path to lecture annotation JSON")
    parser.add_argument("--batch", type=str,
                        help="Path to JSON with segments to summarize")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    summarizer = LectureSummarizer(mode=args.mode)

    if args.input:
        result = summarizer.summarize(args.input)
        print(f"\n{'='*50}")
        print(f"Mode: {result['mode']} | Time: {result['time_seconds']}s")
        print(f"{'='*50}")
        if result.get('bart_draft'):
            print(f"BART Draft: {result['bart_draft'][:200]}")
            print()
        print(f"Summary:    {result['summary']}")
        print(f"{'='*50}")

    elif args.lecture:
        result = summarizer.summarize_lecture(args.lecture)

        out_path = args.output or f"results/{result['video_id']}_summary.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*50}")
        print(f"Lecture: {result['video_id']}")
        print(f"Segments: {result['n_segments']}")
        print(f"Full summary preview: {result['full_summary'][:300]}…")
        print(f"Saved to: {out_path}")
        print(f"{'='*50}")

    elif args.batch:
        with open(args.batch, 'r', encoding='utf-8') as f:
            data = json.load(f)
        segments = data.get('segments', [])

        segments = summarizer.summarize_segments(segments)

        out_path = args.output or "results/batch_summaries.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'segments': segments}, f, indent=2, ensure_ascii=False)
        print(f"Batch results saved to {out_path}")

    else:
        parser.print_help()

    summarizer.unload()


if __name__ == "__main__":
    main()
