#!/usr/bin/env python3
"""
Step 2: Generate pseudo-summaries using FLAN-T5
Loads prepared segments → FLAN-T5 summary generation → quality filtering → saves dataset

Usage:
    python scripts/text_processing/06_generate_pseudo_summaries.py          # Full run
    python scripts/text_processing/06_generate_pseudo_summaries.py --test   # 5 segments only
"""
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


class PseudoSummaryGenerator:
    """Generate pseudo ground-truth summaries with FLAN-T5 + quality filtering"""

    def __init__(self, config: Dict, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.logger = setup_logger("pseudo_summary_gen")

        # Paths
        self.data_dir = Path(config['paths']['data'].get(
            'bart_dataset', 'data/processed/bart_dataset'))

        # FLAN-T5 config
        ft = config.get('flan_t5', {})
        self.model_name = ft.get('model_name', 'google/flan-t5-large')
        self.max_input = ft.get('max_input_length', 512)
        self.max_target = ft.get('max_target_length', 350)
        self.min_target = ft.get('min_target_length', 50)
        self.num_beams = ft.get('num_beams', 4)
        self.length_penalty = ft.get('length_penalty', 1.5)
        self.batch_size = ft.get('batch_size', 4)
        self.min_quality = ft.get('min_quality_score', 0.5)
        self.min_entity_coverage = ft.get('min_entity_coverage', 0.2)

        # Review samples count
        self.n_review = config.get('data_prep', {}).get('summary_review_samples', 20)

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Models (loaded lazily)
        self._flan_model = None
        self._flan_tokenizer = None
        self._sbert_model = None
        self._nlp = None

    # ------------------------------------------------------------------
    # Lazy model loading (keeps memory low until needed)
    # ------------------------------------------------------------------
    def _load_flan(self):
        if self._flan_model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.logger.info(f"Loading FLAN-T5: {self.model_name} …")
        self._flan_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._flan_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.logger.info("FLAN-T5 loaded ✓")

    def _load_sbert(self):
        if self._sbert_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        sbert_name = self.config.get('sentence_bert', {}).get(
            'base_model', 'all-mpnet-base-v2')
        # Guard: bert-base-uncased is NOT a sentence similarity model
        if 'mpnet' not in sbert_name and 'MiniLM' not in sbert_name:
            self.logger.warning(f"SBERT model '{sbert_name}' may not be optimized for "
                                "sentence similarity. Using all-mpnet-base-v2 instead.")
            sbert_name = 'all-mpnet-base-v2'
        self.logger.info(f"Loading SBERT: {sbert_name} …")
        self._sbert_model = SentenceTransformer(sbert_name, device='cpu')
        self.logger.info("SBERT loaded ✓")

    def _load_nlp(self):
        if self._nlp is not None:
            return
        import spacy

        # Priority: trained STEM NER > scispaCy > spaCy core
        ner_path = Path(self.config.get('paths', {}).get('models', {}).get(
            'ner', 'models/text/ner/final'))
        if ner_path.exists():
            try:
                self._nlp = spacy.load(str(ner_path))
                self.logger.info(f"Loaded trained STEM NER from {ner_path} ✓")
                return
            except Exception as e:
                self.logger.warning(f"Could not load trained NER ({e}), falling back")

        try:
            self._nlp = spacy.load("en_core_sci_sm")
            self.logger.info("Using en_core_sci_sm for entity detection")
        except OSError:
            self._nlp = spacy.load("en_core_web_sm")
            self.logger.info("Using en_core_web_sm for entity detection")

    def _unload_flan(self):
        """Free FLAN-T5 GPU memory"""
        if self._flan_model is not None:
            del self._flan_model, self._flan_tokenizer
            self._flan_model = self._flan_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("FLAN-T5 unloaded, GPU memory freed")

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------
    def _build_prompt(self, text: str) -> str:
        """Build a CNN/DailyMail-style summarization prompt for FLAN-T5"""
        # Truncate text to fit within model limits
        words = text.split()
        if len(words) > 400:
            text = " ".join(words[:400])

        return (
            "Summarize the following lecture segment in 1 to 3 sentences. "
            "Capture the core concept, key technical terms, and any "
            "important definitions or relationships mentioned:\n\n"
            f"{text}\n\n"
            "Summary:"
        )

    @torch.no_grad()
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate summaries for a batch of prompts using beam search"""
        inputs = self._flan_tokenizer(
            prompts,
            max_length=self.max_input,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self._flan_model.generate(
            **inputs,
            max_new_tokens=self.max_target,
            min_new_tokens=self.min_target,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        return self._flan_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def generate_summaries(self, segments: List[Dict]) -> List[Dict]:
        """Generate pseudo-summaries for all segments"""
        self._load_flan()

        self.logger.info(f"Generating summaries for {len(segments)} segments …")
        results = []

        for i in tqdm(range(0, len(segments), self.batch_size), desc="Generating"):
            batch = segments[i:i + self.batch_size]
            prompts = [self._build_prompt(s['raw_text']) for s in batch]
            summaries = self._generate_batch(prompts)

            for seg, summary in zip(batch, summaries):
                seg_copy = dict(seg)
                seg_copy['pseudo_summary'] = summary.strip()
                results.append(seg_copy)

        self.logger.info(f"Generated {len(results)} summaries")
        return results

    # ------------------------------------------------------------------
    # Quality filtering
    # ------------------------------------------------------------------
    def _sbert_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts using SBERT"""
        emb = self._sbert_model.encode([text_a, text_b], convert_to_numpy=True)
        from numpy.linalg import norm
        cos = float(np.dot(emb[0], emb[1]) / (norm(emb[0]) * norm(emb[1]) + 1e-9))
        return cos

    def _entity_coverage(self, source: str, summary: str) -> tuple:
        """Check how many key entities from source appear in summary"""
        src_doc = self._nlp(source[:5000])
        sum_doc = self._nlp(summary)

        # Collect source entities / noun chunks as key terms
        src_terms = set()
        for ent in src_doc.ents:
            src_terms.add(ent.text.lower())
        for chunk in src_doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                src_terms.add(chunk.text.lower())

        if not src_terms:
            return 1.0, list(src_terms)

        summary_lower = summary.lower()
        found = [t for t in src_terms if t in summary_lower]
        coverage = len(found) / len(src_terms)
        return coverage, found

    def filter_quality(self, segments: List[Dict]) -> List[Dict]:
        """Filter segments by summary quality and entity coverage"""
        self._load_sbert()
        self._load_nlp()

        self.logger.info("Running quality filtering …")
        self.logger.info(f"  SBERT threshold:   {self.min_quality}")
        self.logger.info(f"  Entity cov threshold: {self.min_entity_coverage}")

        passed, failed_quality, failed_entity, failed_length = [], 0, 0, 0

        for seg in tqdm(segments, desc="Quality check"):
            summary = seg.get('pseudo_summary', '')

            # Basic length checks
            if len(summary.split()) < 10:
                failed_length += 1
                continue

            # SBERT similarity score
            quality_score = self._sbert_similarity(seg['raw_text'], summary)

            # Entity coverage (using trained STEM NER)
            entity_cov, entities_found = self._entity_coverage(seg['raw_text'], summary)

            seg['quality_score'] = round(quality_score, 4)
            seg['entity_coverage'] = round(entity_cov, 4)
            seg['entities_found'] = entities_found

            # Both quality AND entity coverage must pass
            if quality_score < self.min_quality:
                failed_quality += 1
            elif entity_cov < self.min_entity_coverage:
                failed_entity += 1
            else:
                passed.append(seg)

        total_failed = failed_quality + failed_entity + failed_length
        self.logger.info(f"Quality filter results:")
        self.logger.info(f"  ✓ Passed:        {len(passed)}")
        self.logger.info(f"  ✗ Too short:     {failed_length}")
        self.logger.info(f"  ✗ Low SBERT:     {failed_quality}")
        self.logger.info(f"  ✗ Low entity cov: {failed_entity}")

        if len(passed) == 0:
            self.logger.warning("⚠️  All summaries failed quality filter! "
                                "Lowering threshold to keep top 80%.")
            segments.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            keep_n = max(int(len(segments) * 0.8), 1)
            passed = segments[:keep_n]
        elif len(passed) < 100:
            self.logger.warning(f"⚠️  Only {len(passed)} segments passed. "
                                "Consider lowering thresholds if this is too few.")

        return passed

    # ------------------------------------------------------------------
    # Review samples
    # ------------------------------------------------------------------
    def save_review_samples(self, segments: List[Dict]):
        """Save random samples for manual human review"""
        n = min(self.n_review, len(segments))
        samples = random.sample(segments, n)

        review = []
        for s in samples:
            review.append({
                "segment_id": s['segment_id'],
                "lecture_id": s['lecture_id'],
                "raw_text": s['raw_text'][:500] + ("…" if len(s['raw_text']) > 500 else ""),
                "pseudo_summary": s['pseudo_summary'],
                "quality_score": s.get('quality_score', 'N/A'),
                "entity_coverage": s.get('entity_coverage', 'N/A'),
                "entities_found": s.get('entities_found', []),
            })

        path = self.data_dir / "summary_review_samples.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(review, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {n} review samples → {path}")
        self.logger.info("👉 Please review these samples before proceeding to BART training!")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(self) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info("Starting Pseudo-Summary Generation")
        self.logger.info("=" * 60)

        # Load prepared segments from Step 1
        input_path = self.data_dir / "prepared_segments.json"
        assert input_path.exists(), f"Run 05_prepare_bart_data.py first! Missing {input_path}"

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data['segments']
        if self.test_mode:
            segments = segments[:5]
            self.logger.info(f"TEST MODE: processing {len(segments)} segments only")

        # Step A: Generate summaries with FLAN-T5
        segments = self.generate_summaries(segments)

        # Free FLAN-T5 GPU memory before loading SBERT + NER
        self._unload_flan()

        # Step B: Quality filtering
        segments = self.filter_quality(segments)

        # Step C: Save review samples
        self.save_review_samples(segments)

        # Step D: Save final dataset
        stats = {
            "total_segments": len(segments),
            "avg_quality_score": round(np.mean([s.get('quality_score', 0) for s in segments]), 4),
            "avg_entity_coverage": round(np.mean([s.get('entity_coverage', 0) for s in segments]), 4),
        }

        output = {
            "segments": segments,
            "full_transcripts": data.get('full_transcripts', []),
            "statistics": stats,
        }

        out_path = self.data_dir / "dataset_with_summaries.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved dataset → {out_path}")
        self.logger.info(f"Stats: {stats}")
        self.logger.info("✅ Pseudo-summary generation complete!")
        return output


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-summaries with FLAN-T5")
    parser.add_argument("--test", action="store_true", help="Test mode (5 segments)")
    args = parser.parse_args()

    config = config_loader.load_all()
    generator = PseudoSummaryGenerator(config, test_mode=args.test)
    result = generator.run()

    print(f"\n{'=' * 50}")
    print("Pseudo-Summary Generation Complete!")
    print(f"  Segments with summaries: {result['statistics']['total_segments']}")
    print(f"  Avg quality score:       {result['statistics']['avg_quality_score']}")
    print(f"  Avg entity coverage:     {result['statistics']['avg_entity_coverage']}")
    print(f"{'=' * 50}")
    print("\n⚠️  Review summary_review_samples.json before training BART!")


if __name__ == "__main__":
    main()
