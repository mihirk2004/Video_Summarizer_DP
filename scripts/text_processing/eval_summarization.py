#!/usr/bin/env python3
"""
Summarization Evaluation — Comprehensive metrics for BART, LLM, and Hybrid summaries
Computes ROUGE, SBERT similarity, entity coverage, length analysis, and per-lecture breakdowns

Usage:
    python scripts/text_processing/eval_summarization.py                           # Evaluate hybrid results
    python scripts/text_processing/eval_summarization.py --input results/inference_hybrid.json
    python scripts/text_processing/eval_summarization.py --compare                 # Compare all modes
    python scripts/text_processing/eval_summarization.py --export-csv              # Export CSV report

Reads output from 10_hybrid_inference.py (or any JSON with segments containing summaries)
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


class SummarizationEvaluator:
    """Comprehensive evaluation of lecture summarization quality"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger("eval_summarization")

        self.results_dir = Path(config['paths'].get('outputs', {}).get(
            'results', 'results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._sbert = None
        self._nlp = None
        self._rouge = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_sbert(self):
        if self._sbert is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._sbert = SentenceTransformer('all-mpnet-base-v2', device='cpu')

    def _load_nlp(self):
        if self._nlp is not None:
            return
        import spacy
        ner_path = Path(self.config.get('paths', {}).get('models', {}).get(
            'ner', 'models/text/ner/final'))
        if ner_path.exists():
            try:
                self._nlp = spacy.load(str(ner_path))
                return
            except Exception:
                pass
        try:
            self._nlp = spacy.load("en_core_sci_sm")
        except OSError:
            self._nlp = spacy.load("en_core_web_sm")

    def _load_rouge(self):
        if self._rouge is not None:
            return
        import evaluate
        self._rouge = evaluate.load("rouge")

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------
    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute ROUGE-1/2/L scores"""
        self._load_rouge()

        # Filter empty pairs
        valid = [(p, r) for p, r in zip(predictions, references) if p and r]
        if not valid:
            return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

        preds, refs = zip(*valid)
        scores = self._rouge.compute(predictions=list(preds), references=list(refs))
        return {k: round(v, 4) for k, v in scores.items()}

    def compute_sbert_similarity(self, sources: List[str], summaries: List[str]) -> List[float]:
        """SBERT cosine similarity between source and summary"""
        self._load_sbert()
        import torch

        scores = []
        for src, summ in tqdm(zip(sources, summaries), desc="SBERT scoring", total=len(sources)):
            if not summ or not src:
                scores.append(0.0)
                continue
            embs = self._sbert.encode([src, summ], convert_to_tensor=True)
            score = torch.nn.functional.cosine_similarity(
                embs[0].unsqueeze(0), embs[1].unsqueeze(0)).item()
            scores.append(round(score, 4))
        return scores

    def compute_entity_coverage(self, sources: List[str], summaries: List[str]) -> List[float]:
        """Fraction of source entities appearing in summary"""
        self._load_nlp()

        coverages = []
        for src, summ in tqdm(zip(sources, summaries), desc="Entity coverage", total=len(sources)):
            if not summ or not src:
                coverages.append(0.0)
                continue

            doc_src = self._nlp(src)
            src_ents = set()
            for ent in doc_src.ents:
                src_ents.add(ent.text.lower())
            for chunk in doc_src.noun_chunks:
                if len(chunk.text.split()) <= 3:
                    src_ents.add(chunk.text.lower())

            if not src_ents:
                coverages.append(0.0)
                continue

            summ_lower = summ.lower()
            found = sum(1 for e in src_ents if e in summ_lower)
            coverages.append(round(found / len(src_ents), 4))

        return coverages

    def compute_length_metrics(self, summaries: List[str]) -> Dict:
        """Length statistics for summaries"""
        lengths = [len(s.split()) for s in summaries if s]
        if not lengths:
            return {'avg_words': 0, 'min_words': 0, 'max_words': 0, 'std_words': 0}
        return {
            'avg_words': round(np.mean(lengths), 1),
            'min_words': int(np.min(lengths)),
            'max_words': int(np.max(lengths)),
            'std_words': round(np.std(lengths), 1),
        }

    def detect_degenerate(self, summaries: List[str]) -> Dict:
        """Detect degenerate outputs (repetition, too short, gibberish)"""
        issues = {'repetitive': 0, 'too_short': 0, 'too_long': 0, 'empty': 0}
        flagged_ids = []

        for i, s in enumerate(summaries):
            if not s or len(s.strip()) == 0:
                issues['empty'] += 1
                flagged_ids.append(i)
                continue

            words = s.split()
            if len(words) < 8:
                issues['too_short'] += 1
                flagged_ids.append(i)
            elif len(words) > 200:
                issues['too_long'] += 1
                flagged_ids.append(i)

            # Check for extreme repetition
            if len(words) > 5:
                unique = set(words)
                if len(unique) / len(words) < 0.3:
                    issues['repetitive'] += 1
                    flagged_ids.append(i)

        issues['total_flagged'] = len(set(flagged_ids))
        issues['flagged_ratio'] = round(len(set(flagged_ids)) / max(len(summaries), 1), 4)
        return issues

    # ------------------------------------------------------------------
    # Evaluate a set of segments
    # ------------------------------------------------------------------
    def evaluate_segments(self, segments: List[Dict],
                          summary_key: str = 'final_summary',
                          reference_key: str = 'pseudo_summary') -> Dict:
        """Run all metrics on a segment list"""
        sources = [s.get('raw_text', '') for s in segments]
        summaries = [s.get(summary_key, '') for s in segments]
        references = [s.get(reference_key, '') for s in segments]

        self.logger.info(f"Evaluating {len(segments)} segments (key={summary_key}) …")

        # ROUGE vs references
        rouge = self.compute_rouge(summaries, references)
        self.logger.info(f"  ROUGE-1: {rouge['rouge1']}  ROUGE-2: {rouge['rouge2']}  ROUGE-L: {rouge['rougeL']}")

        # SBERT similarity
        sbert_scores = self.compute_sbert_similarity(sources, summaries)
        avg_sbert = round(np.mean(sbert_scores), 4) if sbert_scores else 0

        # Entity coverage
        ent_scores = self.compute_entity_coverage(sources, summaries)
        avg_ent = round(np.mean(ent_scores), 4) if ent_scores else 0

        # Length analysis
        length = self.compute_length_metrics(summaries)

        # Degenerate detection
        degen = self.detect_degenerate(summaries)

        # Per-lecture aggregation
        per_lecture = self._per_lecture_aggregate(segments, sbert_scores, ent_scores, summary_key)

        return {
            'rouge': rouge,
            'sbert_similarity': {'avg': avg_sbert, 'scores': sbert_scores},
            'entity_coverage': {'avg': avg_ent, 'scores': ent_scores},
            'length': length,
            'degenerate': degen,
            'per_lecture': per_lecture,
            'n_segments': len(segments),
        }

    def _per_lecture_aggregate(self, segments, sbert_scores, ent_scores, summary_key):
        """Aggregate metrics per lecture"""
        lectures = defaultdict(lambda: {'sbert': [], 'entity': [], 'count': 0})
        for i, seg in enumerate(segments):
            lid = seg.get('lecture_id', 'unknown')
            lectures[lid]['sbert'].append(sbert_scores[i])
            lectures[lid]['entity'].append(ent_scores[i])
            lectures[lid]['count'] += 1

        result = {}
        for lid, data in lectures.items():
            result[lid] = {
                'segments': data['count'],
                'avg_sbert': round(np.mean(data['sbert']), 4),
                'avg_entity_cov': round(np.mean(data['entity']), 4),
            }
        return result

    # ------------------------------------------------------------------
    # Compare multiple modes
    # ------------------------------------------------------------------
    def compare_modes(self, result_files: Dict[str, str]) -> Dict:
        """Compare BART-only, LLM-only, and Hybrid results"""
        comparison = {}

        for mode, path in result_files.items():
            path = Path(path)
            if not path.exists():
                self.logger.warning(f"Skipping {mode}: {path} not found")
                continue

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            segments = data.get('segments', [])
            self.logger.info(f"\n--- Evaluating mode: {mode} ({len(segments)} segments) ---")
            metrics = self.evaluate_segments(segments)

            comparison[mode] = {
                'rouge1': metrics['rouge']['rouge1'],
                'rouge2': metrics['rouge']['rouge2'],
                'rougeL': metrics['rouge']['rougeL'],
                'sbert': metrics['sbert_similarity']['avg'],
                'entity_cov': metrics['entity_coverage']['avg'],
                'avg_words': metrics['length']['avg_words'],
                'degenerate_ratio': metrics['degenerate']['flagged_ratio'],
                'n_segments': metrics['n_segments'],
            }

        # Print comparison table
        self.logger.info("\n" + "=" * 75)
        self.logger.info("COMPARISON TABLE")
        self.logger.info("=" * 75)
        header = f"{'Mode':<15} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'SBERT':>7} {'EntCov':>7} {'Words':>6} {'Degen%':>7}"
        self.logger.info(header)
        self.logger.info("-" * 75)

        for mode, m in comparison.items():
            line = (f"{mode:<15} {m['rouge1']:>8.4f} {m['rouge2']:>8.4f} "
                    f"{m['rougeL']:>8.4f} {m['sbert']:>7.4f} "
                    f"{m['entity_cov']:>7.4f} {m['avg_words']:>6.1f} "
                    f"{m['degenerate_ratio']*100:>6.1f}%")
            self.logger.info(line)

        self.logger.info("=" * 75)
        return comparison

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_csv(self, segments: List[Dict], metrics: Dict, output_path: str):
        """Export per-segment metrics to CSV"""
        import csv

        sbert = metrics.get('sbert_similarity', {}).get('scores', [])
        ent = metrics.get('entity_coverage', {}).get('scores', [])

        path = Path(output_path)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['segment_id', 'lecture_id', 'sbert_score', 'entity_coverage',
                             'summary_words', 'summary_preview'])
            for i, seg in enumerate(segments):
                summary = seg.get('final_summary', seg.get('enhanced_summary', ''))
                writer.writerow([
                    seg.get('segment_id', f'seg_{i}'),
                    seg.get('lecture_id', 'unknown'),
                    sbert[i] if i < len(sbert) else '',
                    ent[i] if i < len(ent) else '',
                    len(summary.split()),
                    summary[:100],
                ])

        self.logger.info(f"CSV exported → {path}")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self, input_path: Optional[str] = None, compare: bool = False,
            export_csv_flag: bool = False):
        self.logger.info("=" * 60)
        self.logger.info("Summarization Evaluation")
        self.logger.info("=" * 60)

        if compare:
            # Compare all available mode results
            files = {}
            for mode in ['bart-only', 'llm-only', 'hybrid']:
                p = self.results_dir / f"inference_{mode}.json"
                if p.exists():
                    files[mode] = str(p)
            if not files:
                self.logger.error("No inference results found. Run 10_hybrid_inference.py first!")
                return {}

            comparison = self.compare_modes(files)

            # Save comparison
            out_path = self.results_dir / "eval_comparison.json"
            with open(out_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            self.logger.info(f"Comparison saved → {out_path}")
            return comparison

        # Single file evaluation
        if input_path:
            path = Path(input_path)
        else:
            # Try hybrid first, then any available
            for name in ['inference_hybrid.json', 'inference_llm-only.json',
                         'inference_bart-only.json', 'enhanced_summaries.json']:
                p = self.results_dir / name
                if p.exists():
                    path = p
                    break
            else:
                self.logger.error("No results found. Run 10_hybrid_inference.py first!")
                return {}

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data.get('segments', [])
        metrics = self.evaluate_segments(segments)

        # Save full results
        out_path = self.results_dir / "eval_results.json"
        # Don't save per-segment scores in the summary (too large)
        summary = {k: v for k, v in metrics.items() if k not in ('sbert_similarity', 'entity_coverage')}
        summary['sbert_avg'] = metrics['sbert_similarity']['avg']
        summary['entity_coverage_avg'] = metrics['entity_coverage']['avg']
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Results saved → {out_path}")

        if export_csv_flag:
            csv_path = self.results_dir / "eval_per_segment.csv"
            self.export_csv(segments, metrics, str(csv_path))

        # Print summary
        self.logger.info("\n" + "=" * 50)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"  Segments:       {metrics['n_segments']}")
        self.logger.info(f"  ROUGE-1:        {metrics['rouge']['rouge1']}")
        self.logger.info(f"  ROUGE-2:        {metrics['rouge']['rouge2']}")
        self.logger.info(f"  ROUGE-L:        {metrics['rouge']['rougeL']}")
        self.logger.info(f"  SBERT:          {metrics['sbert_similarity']['avg']}")
        self.logger.info(f"  Entity Cov:     {metrics['entity_coverage']['avg']}")
        self.logger.info(f"  Avg Words:      {metrics['length']['avg_words']}")
        self.logger.info(f"  Degenerate:     {metrics['degenerate']['total_flagged']} "
                         f"({metrics['degenerate']['flagged_ratio']*100:.1f}%)")
        self.logger.info("=" * 50)

        return metrics


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate lecture summarization quality")
    parser.add_argument("--input", type=str, default=None, help="Input results JSON")
    parser.add_argument("--compare", action="store_true", help="Compare all modes")
    parser.add_argument("--export-csv", action="store_true", help="Export per-segment CSV")
    args = parser.parse_args()

    config = config_loader.load_all()
    evaluator = SummarizationEvaluator(config)
    evaluator.run(
        input_path=args.input,
        compare=args.compare,
        export_csv_flag=args.export_csv,
    )


if __name__ == "__main__":
    main()
