#!/usr/bin/env python3
"""
Step 5: LLM Enhancement Layer for Lecture Summaries
Takes BART draft summaries + original transcripts → refined summaries using Mistral-7B

Can be used standalone or as part of the hybrid pipeline (10_hybrid_inference.py)

Usage:
    python scripts/text_processing/09_llm_enhance_summaries.py                  # Full run
    python scripts/text_processing/09_llm_enhance_summaries.py --test           # 5 segments
    python scripts/text_processing/09_llm_enhance_summaries.py --input path.json

Requirements:
    pip install bitsandbytes accelerate
"""
import sys
import gc
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


class LLMEnhancer:
    """Enhance BART draft summaries using Mistral-7B-Instruct"""

    def __init__(self, config: Dict, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.logger = setup_logger("llm_enhancer")

        # Paths
        self.data_dir = Path(config['paths']['data'].get(
            'bart_dataset', 'data/processed/bart_dataset'))
        self.output_dir = Path(config['paths'].get('outputs', {}).get(
            'results', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model config
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.max_new_tokens = 250

        # Lazy-loaded
        self._model = None
        self._tokenizer = None
        self._sbert = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.logger.info("Loading Mistral-7B-Instruct (4-bit) …")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model.config.pad_token_id = self._tokenizer.eos_token_id

        self.logger.info("Mistral loaded ✓")

    def _unload_model(self):
        del self._model, self._tokenizer
        self._model = self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_sbert(self):
        if self._sbert is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._sbert = SentenceTransformer('all-mpnet-base-v2', device='cpu')
        self.logger.info("SBERT loaded ✓")

    # ------------------------------------------------------------------
    # Enhancement prompts
    # ------------------------------------------------------------------
    def _build_enhance_prompt(self, transcript: str, bart_draft: str,
                               visual_tags: List[str] = None) -> str:
        """Build refinement prompt: gives LLM the BART draft to improve"""
        # Truncate transcript
        words = transcript.split()
        if len(words) > 400:
            transcript = " ".join(words[:400])

        visual_note = ""
        if visual_tags:
            visual_note = f"\nVisual elements present: {', '.join(visual_tags)}\n"

        return (
            f"<s>[INST] You are an expert teaching assistant. Improve the draft summary below.\n\n"
            f"TRANSCRIPT (for reference):\n{transcript}\n{visual_note}\n\n"
            f"DRAFT SUMMARY:\n{bart_draft}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Keep ALL key technical terms (equations, function names, definitions).\n"
            f"- Fix grammar and remove repetitions.\n"
            f"- Do NOT add information not in the transcript.\n"
            f"- Output only the improved summary, 2-4 sentences.\n\n"
            f"IMPROVED SUMMARY:\n[/INST]"
        )

    def _build_direct_prompt(self, transcript: str,
                              visual_tags: List[str] = None) -> str:
        """Build direct summarization prompt (no BART draft available)"""
        words = transcript.split()
        if len(words) > 500:
            transcript = " ".join(words[:500])

        visual_note = ""
        if visual_tags:
            visual_note = f"\nVisual elements present: {', '.join(visual_tags)}\n"

        return (
            f"<s>[INST] Summarize the following lecture segment in 2 to 4 concise "
            f"sentences. Focus on the main concept, key technical terms, and important "
            f"definitions or relationships.\n\n"
            f"{transcript}\n"
            f"{visual_note}\n[/INST]"
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def enhance_single(self, transcript: str, bart_draft: str = None,
                       visual_tags: List[str] = None) -> str:
        """Enhance a single summary"""
        self._load_model()

        if bart_draft and len(bart_draft.split()) > 5:
            prompt = self._build_enhance_prompt(transcript, bart_draft, visual_tags)
        else:
            prompt = self._build_direct_prompt(transcript, visual_tags)

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1200
        ).to(self.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=20,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.2,
            early_stopping=True,
        )

        prompt_len = inputs['input_ids'].shape[1]
        generated = outputs[0][prompt_len:]
        summary = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        summary = summary.replace("</s>", "").strip()

        return summary

    def enhance_batch(self, segments: List[Dict]) -> List[Dict]:
        """Enhance a batch of segments"""
        self._load_model()

        self.logger.info(f"Enhancing {len(segments)} segments with Mistral-7B …")

        for seg in tqdm(segments, desc="Enhancing"):
            bart_draft = seg.get('bart_summary', seg.get('pseudo_summary', ''))
            transcript = seg.get('raw_text', seg.get('transcript', ''))
            visual_tags = seg.get('visual_tags', [])

            enhanced = self.enhance_single(transcript, bart_draft, visual_tags)
            seg['enhanced_summary'] = enhanced

        self.logger.info(f"Enhanced {len(segments)} segments ✓")
        return segments

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------
    def score_quality(self, segments: List[Dict]) -> List[Dict]:
        """Score enhanced summaries with SBERT similarity"""
        self._load_sbert()

        self.logger.info("Scoring enhanced summaries …")

        for seg in tqdm(segments, desc="Scoring"):
            transcript = seg.get('raw_text', seg.get('transcript', ''))
            enhanced = seg.get('enhanced_summary', '')

            if not enhanced:
                seg['enhance_quality'] = 0.0
                continue

            embs = self._sbert.encode([transcript, enhanced], convert_to_tensor=True)
            score = torch.nn.functional.cosine_similarity(
                embs[0].unsqueeze(0), embs[1].unsqueeze(0)
            ).item()

            seg['enhance_quality'] = round(score, 4)

        return segments

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(self, input_path: Optional[str] = None) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info("LLM Summary Enhancement")
        self.logger.info("=" * 60)

        # Load data
        if input_path:
            path = Path(input_path)
        else:
            path = self.data_dir / "dataset_with_summaries.json"

        assert path.exists(), f"Input file not found: {path}"

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data['segments']
        if self.test_mode:
            segments = segments[:5]
            self.logger.info(f"TEST MODE: processing {len(segments)} segments")

        # Enhance summaries
        segments = self.enhance_batch(segments)

        # Free GPU, score quality
        self._unload_model()
        segments = self.score_quality(segments)

        # Save results
        avg_quality = sum(s.get('enhance_quality', 0) for s in segments) / len(segments)

        output = {
            "segments": segments,
            "statistics": {
                "total_segments": len(segments),
                "avg_enhance_quality": round(avg_quality, 4),
                "enhancer": "mistral-7b-instruct-v0.3",
            }
        }

        out_path = self.output_dir / "enhanced_summaries.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # Save review samples
        n_review = min(20, len(segments))
        samples = random.sample(segments, n_review)
        review = []
        for s in samples:
            review.append({
                "segment_id": s.get('segment_id', 'unknown'),
                "raw_text": s.get('raw_text', '')[:400],
                "bart_draft": s.get('bart_summary', s.get('pseudo_summary', ''))[:300],
                "enhanced_summary": s.get('enhanced_summary', ''),
                "enhance_quality": s.get('enhance_quality', 'N/A'),
            })

        review_path = self.output_dir / "enhanced_review_samples.json"
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(review, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved enhanced summaries → {out_path}")
        self.logger.info(f"Saved review samples → {review_path}")
        self.logger.info(f"Avg enhance quality: {avg_quality:.4f}")
        self.logger.info("✅ LLM enhancement complete!")

        return output


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Enhance summaries with Mistral-7B")
    parser.add_argument("--test", action="store_true", help="Test mode (5 segments)")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input JSON (default: dataset_with_summaries.json)")
    args = parser.parse_args()

    config = config_loader.load_all()
    enhancer = LLMEnhancer(config, test_mode=args.test)
    result = enhancer.run(input_path=args.input)

    stats = result['statistics']
    print(f"\n{'=' * 50}")
    print("LLM Enhancement Complete!")
    print(f"  Segments:     {stats['total_segments']}")
    print(f"  Avg quality:  {stats['avg_enhance_quality']}")
    print(f"  Enhancer:     {stats['enhancer']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
