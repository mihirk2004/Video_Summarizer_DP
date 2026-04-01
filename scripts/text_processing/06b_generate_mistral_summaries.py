#!/usr/bin/env python3
"""
Step 2b: Generate pseudo-summaries using Mistral-7B-Instruct (4-bit quantized)
Higher-quality alternative to FLAN-T5 — produces abstractive, coherent summaries

Usage:
    python scripts/text_processing/06b_generate_mistral_summaries.py         # Full run
    python scripts/text_processing/06b_generate_mistral_summaries.py --test  # 5 segments only

Requirements (install on Colab):
    pip install bitsandbytes accelerate
"""
import sys
import json
import gc
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


class MistralSummaryGenerator:
    """Generate high-quality pseudo-summaries with Mistral-7B-Instruct (4-bit)"""

    def __init__(self, config: Dict, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.logger = setup_logger("mistral_summary_gen")

        # Paths
        self.data_dir = Path(config['paths']['data'].get(
            'bart_dataset', 'data/processed/bart_dataset'))

        # Mistral config
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.max_new_tokens = 200
        self.min_quality = config.get('flan_t5', {}).get('min_quality_score', 0.5)
        self.min_entity_coverage = config.get('flan_t5', {}).get('min_entity_coverage', 0.2)
        self.n_review = config.get('data_prep', {}).get('summary_review_samples', 20)

        # Lazy-loaded models
        self._model = None
        self._tokenizer = None
        self._sbert_model = None
        self._nlp = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_mistral(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.logger.info(f"Loading Mistral-7B-Instruct (4-bit) …")

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

        # Set pad token if missing
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model.config.pad_token_id = self._tokenizer.eos_token_id

        vram = torch.cuda.get_device_properties(0).total_mem / 1e9 if torch.cuda.is_available() else 0
        self.logger.info(f"Mistral loaded ✓ (VRAM available: {vram:.1f} GB)")

    def _unload_mistral(self):
        """Free GPU memory"""
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Mistral unloaded, GPU memory freed")

    def _load_sbert(self):
        if self._sbert_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        sbert_name = self.config.get('sentence_bert', {}).get(
            'base_model', 'all-mpnet-base-v2')
        # Guard against non-sentence-similarity models
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

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------
    def _build_prompt(self, text: str) -> str:
        """Build Mistral instruct-format prompt"""
        # Truncate long segments
        words = text.split()
        if len(words) > 500:
            text = " ".join(words[:500])

        return (
            "<s>[INST] You are an expert teaching assistant. "
    "Summarize the lecture segment in 1–3 concise sentences. "
    "Capture the main concept, key definitions, and important technical terms. "
    "Do not use phrases like 'In this segment', 'The lecturer explains', or similar filler text. "
    "Return only the summary.\n\n"
    f"{text}\n"
    "[/INST]"
        )

    @torch.no_grad()
    def generate_summaries(self, segments: List[Dict]) -> List[Dict]:
        """Generate summaries using Mistral-7B-Instruct"""
        self._load_mistral()

        self.logger.info(f"Generating summaries for {len(segments)} segments …")

        for seg in tqdm(segments, desc="Generating"):
            prompt = self._build_prompt(seg['raw_text'])

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.device)

            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=20,
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                early_stopping=True,
            )

            # Decode only the new tokens (skip the prompt)
            prompt_len = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][prompt_len:]
            summary = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Clean up common artefacts
            summary = summary.replace("</s>", "").strip()

            seg['pseudo_summary'] = summary
            seg['formatted_input'] = seg.get('formatted_input', seg['raw_text'])

        self.logger.info(f"Generated {len(segments)} summaries")
        return segments

    # ------------------------------------------------------------------
    # Quality filtering (reuse same logic as 06)
    # ------------------------------------------------------------------
    def _sbert_similarity(self, text: str, summary: str) -> float:
        """Compute SBERT cosine similarity between source and summary"""
        embs = self._sbert_model.encode([text, summary], convert_to_tensor=True)
        cos = torch.nn.functional.cosine_similarity(embs[0].unsqueeze(0), embs[1].unsqueeze(0))
        return cos.item()

    def _entity_coverage(self, text: str, summary: str):
        """Check how many source entities appear in the summary"""
        doc_src = self._nlp(text)
        doc_sum = self._nlp(summary)

        src_ents = set()
        for ent in doc_src.ents:
            src_ents.add(ent.text.lower())
        for chunk in doc_src.noun_chunks:
            if len(chunk.text.split()) <= 3:
                src_ents.add(chunk.text.lower())

        if not src_ents:
            return 0.0, []

        sum_text_lower = summary.lower()
        found = [e for e in src_ents if e in sum_text_lower]
        coverage = len(found) / len(src_ents)
        return coverage, found

    def filter_quality(self, segments: List[Dict]) -> List[Dict]:
        """Filter segments by summary quality and entity coverage"""
        self._load_sbert()
        self._load_nlp()

        self.logger.info("Running quality filtering …")
        self.logger.info(f"  SBERT threshold:      {self.min_quality}")
        self.logger.info(f"  Entity cov threshold: {self.min_entity_coverage}")

        passed, failed_quality, failed_entity, failed_length = [], 0, 0, 0

        for seg in tqdm(segments, desc="Quality check"):
            summary = seg.get('pseudo_summary', '')

            if len(summary.split()) < 8:
                failed_length += 1
                continue

            quality_score = self._sbert_similarity(seg['raw_text'], summary)
            entity_cov, entities_found = self._entity_coverage(seg['raw_text'], summary)

            seg['quality_score'] = round(quality_score, 4)
            seg['entity_coverage'] = round(entity_cov, 4)
            seg['entities_found'] = entities_found

            if quality_score < self.min_quality:
                failed_quality += 1
            elif entity_cov < self.min_entity_coverage:
                failed_entity += 1
            else:
                passed.append(seg)

        self.logger.info(f"Quality filter results:")
        self.logger.info(f"  ✓ Passed:        {len(passed)}")
        self.logger.info(f"  ✗ Too short:     {failed_length}")
        self.logger.info(f"  ✗ Low SBERT:     {failed_quality}")
        self.logger.info(f"  ✗ Low entity cov: {failed_entity}")

        if len(passed) == 0:
            self.logger.warning("⚠️  All summaries failed! Keeping top 80%.")
            segments.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            passed = segments[:max(int(len(segments) * 0.8), 1)]
        elif len(passed) < 100:
            self.logger.warning(f"⚠️  Only {len(passed)} passed. "
                                "Consider lowering thresholds.")

        return passed

    # ------------------------------------------------------------------
    # Review samples
    # ------------------------------------------------------------------
    def save_review_samples(self, segments: List[Dict]):
        """Save random samples for manual review"""
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
        self.logger.info("👉 Please review these samples before proceeding!")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(self) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info("Mistral-7B Pseudo-Summary Generation")
        self.logger.info("=" * 60)

        # Load prepared segments from Step 05
        input_path = self.data_dir / "prepared_segments.json"
        assert input_path.exists(), f"Run 05_prepare_bart_data.py first! Missing {input_path}"

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data['segments']
        if self.test_mode:
            segments = segments[:5]
            self.logger.info(f"TEST MODE: processing {len(segments)} segments only")

        # Step A: Generate summaries with Mistral-7B
        segments = self.generate_summaries(segments)

        # Free GPU memory before quality filtering
        self._unload_mistral()

        # Step B: Quality filtering
        segments = self.filter_quality(segments)

        # Step C: Save review samples
        self.save_review_samples(segments)

        # Step D: Save final dataset (same format as script 06)
        stats = {
            "total_segments": len(segments),
            "avg_quality_score": round(np.mean([s.get('quality_score', 0) for s in segments]), 4),
            "avg_entity_coverage": round(np.mean([s.get('entity_coverage', 0) for s in segments]), 4),
            "generator": "mistral-7b-instruct-v0.3",
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
        self.logger.info("✅ Mistral pseudo-summary generation complete!")
        return output


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-summaries with Mistral-7B-Instruct (4-bit)")
    parser.add_argument("--test", action="store_true", help="Test mode (5 segments)")
    args = parser.parse_args()

    config = config_loader.load_all()
    generator = MistralSummaryGenerator(config, test_mode=args.test)
    result = generator.run()

    print(f"\n{'=' * 50}")
    print("Mistral Pseudo-Summary Generation Complete!")
    print(f"  Generator:           {result['statistics']['generator']}")
    print(f"  Segments:            {result['statistics']['total_segments']}")
    print(f"  Avg quality score:   {result['statistics']['avg_quality_score']}")
    print(f"  Avg entity coverage: {result['statistics']['avg_entity_coverage']}")
    print(f"{'=' * 50}")
    print("\n⚠️  Review summary_review_samples.json before training BART!")


if __name__ == "__main__":
    main()
