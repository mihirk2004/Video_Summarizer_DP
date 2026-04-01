#!/usr/bin/env python3
"""
Step 6: Hybrid Inference Pipeline — BART draft → LLM refinement → Quality gate
Compares three modes: BART-only, LLM-only, and Hybrid (BART + LLM)

Usage:
    python scripts/text_processing/10_hybrid_inference.py                    # Hybrid mode
    python scripts/text_processing/10_hybrid_inference.py --bart-only        # BART only
    python scripts/text_processing/10_hybrid_inference.py --llm-only         # LLM only
    python scripts/text_processing/10_hybrid_inference.py --compare          # Run all 3, compare
    python scripts/text_processing/10_hybrid_inference.py --test             # 5 segments only

Requirements:
    pip install peft bitsandbytes accelerate evaluate
"""
import sys
import gc
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


class HybridInferencePipeline:
    """End-to-end inference: BART → LLM → Quality Gate"""

    def __init__(self, config: Dict, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.logger = setup_logger("hybrid_inference")

        # Paths
        self.data_dir = Path(config['paths']['data'].get(
            'bart_dataset', 'data/processed/bart_dataset'))
        self.model_dir = Path(config['paths']['models'].get(
            'bart', 'models/text/bart_summarizer'))
        self.output_dir = Path(config['paths'].get('outputs', {}).get(
            'results', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / "hybrid_checkpoint.json"
        self.bart_quality_threshold = self.config.get("bart_quality_threshold", 0.3)
        self.batch_size = self.config.get("llm_batch_size", 4)
        self.bart_skip_threshold = self.config.get("bart_skip_threshold", 0.85)
        self.max_batch_tokens = self.config.get("max_batch_tokens", 4000)
        # Lazy-loaded models
        self._bart_model = None
        self._bart_tokenizer = None
        self._llm_model = None
        self._llm_tokenizer = None
        self._sbert = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # BART model
    # ------------------------------------------------------------------
    def _load_bart(self):
        if self._bart_model is not None:
            return

        from transformers import AutoTokenizer
        best_model_dir = self.model_dir / "best_model"

        if not best_model_dir.exists():
            self.logger.error(f"No BART model found at {best_model_dir}")
            self.logger.error("Run 08_train_bart_summarizer.py first!")
            raise FileNotFoundError(f"Missing {best_model_dir}")

        self.logger.info(f"Loading BART from {best_model_dir} …")

        # Check if this is a LoRA model
        adapter_config = best_model_dir / "adapter_config.json"
        if adapter_config.exists():
            from transformers import BartForConditionalGeneration
            from peft import PeftModel

            base_name = self.config.get('bart', {}).get(
                'model_name', 'facebook/bart-large-cnn')
            base_model = BartForConditionalGeneration.from_pretrained(base_name)

            self._bart_tokenizer = AutoTokenizer.from_pretrained(str(best_model_dir))
            base_model.resize_token_embeddings(len(self._bart_tokenizer))
            self._bart_model = PeftModel.from_pretrained(
                base_model, str(best_model_dir), is_trainable=False)
            self.logger.info("BART LoRA model loaded ✓")
        else:
            from transformers import BartForConditionalGeneration
            self._bart_model = BartForConditionalGeneration.from_pretrained(str(best_model_dir))
            self._bart_tokenizer = AutoTokenizer.from_pretrained(str(best_model_dir))
            self.logger.info("BART full model loaded ✓")

        self._bart_model.eval()
        self._bart_model.to(self.device)

        # Clear generation config overrides
        self._bart_model.generation_config.max_length = None
        self._bart_model.generation_config.min_length = None
        self._bart_model.generation_config.max_new_tokens = None

    def _unload_bart(self):
        del self._bart_model, self._bart_tokenizer
        self._bart_model = self._bart_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("BART unloaded")

    # ------------------------------------------------------------------
    # LLM model (Mistral-7B)
    # ------------------------------------------------------------------
    def _load_llm(self):
        if self._llm_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.logger.info("Loading Mistral-7B-Instruct (4-bit) …")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self._llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._llm_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map="auto", torch_dtype=torch.float16,
        )

        if self._llm_tokenizer.pad_token is None:
            self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token
            self._llm_model.config.pad_token_id = self._llm_tokenizer.eos_token_id

        self.logger.info("Mistral loaded ✓")

    def _unload_llm(self):
        del self._llm_model, self._llm_tokenizer
        self._llm_model = self._llm_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Mistral unloaded")

    # ------------------------------------------------------------------
    # SBERT
    # ------------------------------------------------------------------
    def _load_sbert(self):
        if self._sbert is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._sbert = SentenceTransformer('all-mpnet-base-v2', device='cpu')
        self.logger.info("SBERT loaded ✓")

    # ------------------------------------------------------------------
    # Single-segment inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _bart_generate(self, transcript: str) -> str:
        """Generate summary with BART"""
        max_target = self.config.get('bart', {}).get('max_target_length', 350)

        inputs = self._bart_tokenizer(
            transcript, return_tensors="pt", truncation=True,
            max_length=1024, padding=True,
        ).to(self.device)

        gen_ids = self._bart_model.generate(
            **inputs,
            max_new_tokens=max_target,
            min_new_tokens=10,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=2.0,
            early_stopping=True,
        )

        return self._bart_tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def _llm_generate(self, transcript: str, bart_draft: str = None) -> str:
        """Generate/refine summary with Mistral-7B"""
        words = transcript.split()
        if len(words) > 400:
            transcript = " ".join(words[:400])

        if bart_draft and len(bart_draft.split()) > 8:
            # Refinement mode
            prompt = (
                f"<s>[INST] You are an expert teaching assistant. Refine this draft "
                f"summary of a lecture segment.\n\n"
                f"TRANSCRIPT:\n{transcript}\n\n"
                f"DRAFT SUMMARY:\n{bart_draft}\n\n"
                f"Fix errors, remove repetition, ensure key terms are included. "
                f"Keep it concise (2-4 sentences). If the draft is mostly gibberish, "
                f"write a fresh summary from the transcript.\n\n"
                f"REFINED SUMMARY:\n[/INST]"
            )
        else:
            # Direct generation
            prompt = (
                f"<s>[INST] Summarize this lecture segment in 2-4 concise sentences. "
                f"Focus on main concepts and key technical terms.\n\n"
                f"{transcript}\n[/INST]"
            )

        inputs = self._llm_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1200
        ).to(self.device)

        outputs = self._llm_model.generate(
            **inputs,
            max_new_tokens=250,
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
        summary = self._llm_tokenizer.decode(generated, skip_special_tokens=True).strip()
        return summary.replace("</s>", "").strip()
    
    @torch.no_grad()
    def _llm_generate_batch(self, prompts: List[str], max_new_tokens=250) -> List[str]:
        inputs = self._llm_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1200,
        ).to(self.device)

        outputs = self._llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=20,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.2,
            early_stopping=True,
        )

        prompt_len = inputs['input_ids'].shape[1]

        decoded = self._llm_tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )

        return [s.replace("</s>", "").strip() for s in decoded]

    def _build_refinement_prompt(self, transcript, bart_draft=None, visual_tags=None):
        words = transcript.split()
        if len(words) > 400:
            transcript = " ".join(words[:400])

        visual_note = f"\nVisual elements: {', '.join(visual_tags)}" if visual_tags else ""

        if bart_draft and len(bart_draft.split()) > 8:
            return (
                f"<s>[INST] Refine this lecture summary.\n\n"
                f"TRANSCRIPT:\n{transcript}{visual_note}\n\n"
                f"DRAFT SUMMARY:\n{bart_draft}\n\n"
                f"- Keep key technical terms\n"
                f"- Fix grammar and repetition\n"
                f"- Do NOT add new info\n"
                f"- 2-4 sentences only\n\n"
                f"REFINED SUMMARY:\n[/INST]"
            )
        else:
            return (
                f"<s>[INST] Summarize this lecture segment in 2-4 sentences.\n\n"
                f"{transcript}{visual_note}\n[/INST]"
            )

    def _score_quality(self, transcript: str, summary: str) -> float:
        """SBERT cosine similarity between source and summary"""
        embs = self._sbert.encode([transcript, summary], convert_to_tensor=True)
        return torch.nn.functional.cosine_similarity(
            embs[0].unsqueeze(0), embs[1].unsqueeze(0)
        ).item()

    # ------------------------------------------------------------------
    # Pipeline modes
    # ------------------------------------------------------------------
    def run_bart_only(self, segments: List[Dict]) -> List[Dict]:
        """BART-only inference"""
        self._load_bart()
        self.logger.info("Mode: BART-only")

        for seg in tqdm(segments, desc="BART"):
            t0 = time.time()
            seg['bart_summary'] = self._bart_generate(seg['raw_text'])
            seg['bart_time'] = round(time.time() - t0, 3)
            seg['final_summary'] = seg['bart_summary']

        self._unload_bart()
        return segments

    def run_llm_only(self, segments: List[Dict]) -> List[Dict]:
        """LLM-only inference (no BART draft)"""
        self._load_llm()
        self.logger.info("Mode: LLM-only")

        for seg in tqdm(segments, desc="LLM"):
            t0 = time.time()
            seg['llm_summary'] = self._llm_generate(seg['raw_text'])
            seg['llm_time'] = round(time.time() - t0, 3)
            seg['final_summary'] = seg['llm_summary']

        self._unload_llm()
        return segments

    def run_hybrid(self, segments: List[Dict]) -> List[Dict]:
        """Hybrid: BART → Quality Gate → Batched LLM + Checkpointing"""

        self.logger.info("Mode: Hybrid (BART → LLM with batching + checkpoint)")

        # -------------------------------
        # Resume from checkpoint
        # -------------------------------
        processed_ids = set()
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed_ids = set(checkpoint.get("processed_ids", []))

            self.logger.info(f"Resuming from checkpoint: {len(processed_ids)} done")

        # -------------------------------
        # Step 1: BART + Quality
        # -------------------------------
        self._load_bart()
        self._load_sbert()

        for seg in tqdm(segments, desc="BART draft"):
            if seg.get("segment_id") in processed_ids:
                continue

            t0 = time.time()

            bart_summary = self._bart_generate(seg['raw_text'])
            seg['bart_summary'] = bart_summary

            # Quality score
            bart_quality = self._score_quality(seg['raw_text'], bart_summary)
            seg['bart_quality'] = round(bart_quality, 4)

            seg['bart_low_quality'] = bart_quality < self.bart_quality_threshold
            seg['bart_time'] = round(time.time() - t0, 3)
            if bart_quality > self.bart_skip_threshold:
                seg['final_summary'] = bart_summary
                seg['llm_summary'] = None
                seg['llm_time'] = 0.0

                processed_ids.add(seg.get("segment_id"))

        self._unload_bart()

        # -------------------------------
        # Step 2: Batched LLM
        # -------------------------------
        self._load_llm()

        prompt_list = []
        index_map = []

        for idx, seg in enumerate(segments):
            if seg.get("segment_id") in processed_ids:
                continue

            if seg.get("bart_low_quality", False):
                prompt = self._build_refinement_prompt(seg['raw_text'], bart_draft=None)
            else:
                prompt = self._build_refinement_prompt(
                    seg['raw_text'],
                    bart_draft=seg['bart_summary']
                )

            prompt_list.append(prompt)
            index_map.append(idx)

        # -------------------------------
        # Batch processing
        # -------------------------------
        for i in tqdm(range(0, len(prompt_list), self.batch_size), desc="LLM batches"):
            batch_prompts = prompt_list[i:i + self.batch_size]
            batch_indices = index_map[i:i + self.batch_size]

            t0 = time.time()

            summaries = self._llm_generate_batch(batch_prompts)

            batch_time = time.time() - t0
            per_item_time = batch_time / len(summaries)
            for j, summary in enumerate(summaries):
                idx = batch_indices[j]
                seg = segments[idx]

                seg['llm_summary'] = summary
                seg['final_summary'] = summary
                seg['llm_time'] = round(per_item_time, 3)

                processed_ids.add(seg.get("segment_id"))

            # -------------------------------
            # Save checkpoint
            # -------------------------------
            with open(self.checkpoint_file, 'w') as f:
                json.dump({"processed_ids": list(processed_ids)}, f)

        self._unload_llm()

        self.logger.info("Hybrid inference complete ✓")
        return segments

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, segments: List[Dict]) -> Dict:
        """Score all summaries and compute ROUGE against pseudo-labels"""
        self._load_sbert()

        self.logger.info("Evaluating summaries …")

        # SBERT quality scores
        for seg in tqdm(segments, desc="Quality scoring"):
            transcript = seg['raw_text']
            final = seg.get('final_summary', '')
            seg['final_quality'] = round(self._score_quality(transcript, final), 4)

            # Score individual components if available
            if 'bart_summary' in seg:
                seg['bart_quality'] = round(
                    self._score_quality(transcript, seg['bart_summary']), 4)

        # ROUGE against pseudo-labels (if available)
        rouge_results = {}
        try:
            import evaluate
            rouge = evaluate.load("rouge")

            pseudo_refs = [s.get('pseudo_summary', '') for s in segments]
            final_preds = [s.get('final_summary', '') for s in segments]

            if any(pseudo_refs):
                rouge_results = rouge.compute(
                    predictions=final_preds, references=pseudo_refs)
                rouge_results = {k: round(v, 4) for k, v in rouge_results.items()}
        except Exception as e:
            self.logger.warning(f"Could not compute ROUGE: {e}")

        # Summary stats
        avg_quality = np.mean([s.get('final_quality', 0) for s in segments])
        avg_bart_quality = np.mean([s.get('bart_quality', 0) for s in segments
                                     if 'bart_quality' in s]) if any('bart_quality' in s for s in segments) else 0

        stats = {
            "total_segments": len(segments),
            "avg_final_quality": round(avg_quality, 4),
            "avg_bart_quality": round(avg_bart_quality, 4),
            "rouge_vs_pseudo": rouge_results,
        }

        # Log sample outputs
        self.logger.info("─── Sample Results ───")
        n_show = min(3, len(segments))
        for i in range(n_show):
            seg = segments[i]
            self.logger.info(f"  [{seg.get('segment_id', i)}]")
            if 'bart_summary' in seg:
                self.logger.info(f"    BART:  {seg['bart_summary'][:150]}…")
            self.logger.info(f"    FINAL: {seg['final_summary'][:150]}…")
            self.logger.info(f"    Quality: {seg.get('final_quality', 'N/A')}")
            self.logger.info("")

        return stats

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self, mode: str = "hybrid") -> Dict:
        self.logger.info("=" * 60)
        self.logger.info(f"Hybrid Inference Pipeline — mode: {mode}")
        self.logger.info("=" * 60)

        # Load data
        input_path = self.data_dir / "dataset_with_summaries.json"
        assert input_path.exists(), f"Missing {input_path}"

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data['segments']
        if self.test_mode:
            segments = segments[:5]

        # Run selected mode
        t_start = time.time()

        if mode == "bart-only":
            segments = self.run_bart_only(segments)
        elif mode == "llm-only":
            segments = self.run_llm_only(segments)
        elif mode == "hybrid":
            segments = self.run_hybrid(segments)
        elif mode == "compare":
            return self._run_comparison(segments)

        total_time = time.time() - t_start

        # Evaluate
        stats = self.evaluate(segments)
        stats['mode'] = mode
        stats['total_time_seconds'] = round(total_time, 1)
        stats['avg_time_per_segment'] = round(total_time / len(segments), 2)

        # Save
        out_path = self.output_dir / f"inference_{mode}.json"
        output = {"segments": segments, "statistics": stats}
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved → {out_path}")
        self.logger.info(f"Stats: {json.dumps(stats, indent=2)}")
        return stats

    def _run_comparison(self, segments: List[Dict]) -> Dict:
        """Run all 3 modes and compare"""
        import copy

        self.logger.info("Running comparison across all 3 modes …")
        results = {}

        for mode in ["bart-only", "llm-only", "hybrid"]:
            self.logger.info(f"\n{'=' * 40}")
            self.logger.info(f"Running mode: {mode}")
            segs = copy.deepcopy(segments)

            t0 = time.time()
            if mode == "bart-only":
                segs = self.run_bart_only(segs)
            elif mode == "llm-only":
                segs = self.run_llm_only(segs)
            elif mode == "hybrid":
                segs = self.run_hybrid(segs)

            elapsed = time.time() - t0
            stats = self.evaluate(segs)
            stats['mode'] = mode
            stats['total_time_seconds'] = round(elapsed, 1)
            results[mode] = stats

            # Save individual results
            out_path = self.output_dir / f"inference_{mode}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({"segments": segs, "statistics": stats}, f, indent=2, ensure_ascii=False)

        # Print comparison table
        self.logger.info("\n" + "=" * 60)
        self.logger.info("COMPARISON RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"{'Mode':<15} {'Quality':>10} {'ROUGE-2':>10} {'Time':>10}")
        self.logger.info("-" * 50)
        for mode, stats in results.items():
            r2 = stats.get('rouge_vs_pseudo', {}).get('rouge2', 'N/A')
            self.logger.info(
                f"{mode:<15} {stats['avg_final_quality']:>10.4f} "
                f"{r2 if isinstance(r2, str) else f'{r2:.4f}':>10} "
                f"{stats['total_time_seconds']:>8.1f}s"
            )

        # Save comparison
        comp_path = self.output_dir / "inference_comparison.json"
        with open(comp_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"\nComparison saved → {comp_path}")
        return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hybrid BART+LLM inference pipeline")
    parser.add_argument("--test", action="store_true", help="Test mode (5 segments)")
    parser.add_argument("--bart-only", action="store_true", help="BART only mode")
    parser.add_argument("--llm-only", action="store_true", help="LLM only mode")
    parser.add_argument("--compare", action="store_true", help="Compare all 3 modes")
    args = parser.parse_args()

    if args.bart_only:
        mode = "bart-only"
    elif args.llm_only:
        mode = "llm-only"
    elif args.compare:
        mode = "compare"
    else:
        mode = "hybrid"

    config = config_loader.load_all()
    pipeline = HybridInferencePipeline(config, test_mode=args.test)
    results = pipeline.run(mode=mode)

    print(f"\n{'=' * 50}")
    if mode == "compare":
        print("Comparison Complete! See results/ for details.")
        for m, s in results.items():
            r2 = s.get('rouge_vs_pseudo', {}).get('rouge2', 'N/A')
            print(f"  {m:<15} Quality={s['avg_final_quality']:.4f}  "
                  f"ROUGE-2={r2 if isinstance(r2, str) else f'{r2:.4f}'}  "
                  f"Time={s['total_time_seconds']}s")
    else:
        print(f"Inference Complete! Mode: {mode}")
        print(f"  Quality: {results.get('avg_final_quality', 'N/A')}")
        print(f"  ROUGE-2: {results.get('rouge_vs_pseudo', {}).get('rouge2', 'N/A')}")
        print(f"  Time: {results.get('total_time_seconds', 'N/A')}s")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
