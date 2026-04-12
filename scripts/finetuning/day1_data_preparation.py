#!/usr/bin/env python3
"""
=============================================================================
Day 1 — SFT Data Preparation & Teacher Augmentation
=============================================================================
Fine-Tuning Pipeline for DeepSeek-Math-7B & CodeLlama-7B

This script:
  1. Loads multimodal_inference.json (1312 segments with quality scores)
  2. Cross-references annotation JSONs for subject metadata (Maths / CS)
  3. Filters by quality_score >= 0.4 and word_count >= 50
  4. Samples balanced Math/CS subsets (prioritizing visual-tag matches)
  5. Augments summaries using Gemini 2.5 Flash (smart: skips quality >= 0.85)
  6. Formats as ChatML JSONL for SFT training
  7. Outputs train/val splits for both Math and CS

Run in Google Colab:
  1. Upload this script + multimodal_inference.json + annotations/ to Drive
  2. Mount Drive, set paths below
  3. Set GEMINI_API_KEY
  4. Run: !python day1_data_preparation.py

Author: Auto-generated for DP Project fine-tuning pipeline
=============================================================================
"""

import json
import os
import re
import sys
import time
import random
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION — Update these paths for your environment
# ─────────────────────────────────────────────────────────────────────

class Config:
    """All configurable parameters in one place."""

    # ---------- Paths ----------
    # For local:
    # INFERENCE_JSON = r"d:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\results\multimodal_inference.json"
    # ANNOTATIONS_DIR = r"d:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\data\annotations"
    # OUTPUT_DIR = r"d:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\data\sft_data"

    # For Colab (uncomment after mounting Drive):
    INFERENCE_JSON = "/content/drive/MyDrive/Colab_text_modelling_dp_01/results/multimodal_inference.json"
    ANNOTATIONS_DIR = "/content/drive/MyDrive/Colab_text_modelling_dp_01/data/annotations"
    OUTPUT_DIR = "/content/drive/MyDrive/Colab_text_modelling_dp_01/data/processed/sft_data"

    # ---------- Filtering ----------
    MIN_QUALITY_SCORE = 0.4
    MIN_WORD_COUNT = 50          # Skip quiz fragments, intros, outros
    HIGH_QUALITY_THRESHOLD = 0.85  # Skip augmentation for these
    LOW_QUALITY_THRESHOLD = 0.5   # Generate 2 augmented variants for these

    # ---------- Sampling ----------
    MATH_SAMPLE_SIZE = 200       # Total math examples to include
    CS_SAMPLE_SIZE = 200         # Total CS examples to include
    TRAIN_RATIO = 0.9            # 90% train, 10% val
    RANDOM_SEED = 42

    # ---------- Gemini API ----------
    GEMINI_API_KEY = ""          # Set via env var GEMINI_API_KEY or here
    GEMINI_MODEL = "gemini-2.5-flash"
    GEMINI_RPM_LIMIT = 15       # Free tier: ~15 requests/min
    GEMINI_RETRY_DELAY = 5      # Seconds between retries
    GEMINI_MAX_RETRIES = 3

    # ---------- System Prompts ----------
    MATH_SYSTEM_PROMPT = (
        "You are a mathematics lecture summarizer. "
        "Summarize the lecture segment preserving all equations, "
        "mathematical notation, and step-by-step reasoning. "
        "Use LaTeX notation for equations where appropriate."
    )
    CS_SYSTEM_PROMPT = (
        "You are a computer science lecture summarizer. "
        "Summarize the lecture segment preserving all code references, "
        "function names, variable names, and technical accuracy. "
        "Use proper code formatting for technical terms."
    )


# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("day1_prep")


# ─────────────────────────────────────────────────────────────────────
# STEP 1: Load Data
# ─────────────────────────────────────────────────────────────────────

def load_inference_data(path: str) -> List[Dict]:
    """Load multimodal_inference.json and return segments list."""
    log.info(f"Loading inference data from {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data["segments"]
    log.info(f"  Loaded {len(segments)} segments")
    return segments


def load_subject_mapping(annotations_dir: str) -> Dict[str, str]:
    """
    Load subject metadata from annotation JSONs.
    Returns: {lecture_id: subject} mapping
    """
    log.info(f"Loading subject mapping from {annotations_dir}")
    subject_map = {}
    annot_dir = Path(annotations_dir)

    if not annot_dir.exists():
        log.warning(f"  Annotations dir not found: {annotations_dir}")
        log.warning("  Falling back to heuristic subject detection")
        return {}

    for f in sorted(annot_dir.glob("lecture_*_annotated.json")):
        lid = f.stem.replace("_annotated", "")
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            subject = d.get("metadata", {}).get("subject", "Unknown")
            subject_map[lid] = subject
        except Exception as e:
            log.warning(f"  Error reading {f.name}: {e}")

    # Count distribution
    counts = Counter(subject_map.values())
    log.info(f"  Subject mapping loaded: {dict(counts)}")
    return subject_map


def detect_subject_heuristic(segment: Dict) -> str:
    """
    Fallback: detect subject from visual_tags and raw_text keywords.
    Used when annotation metadata is unavailable.
    """
    visual_tags = segment.get("visual_tags", [])
    raw_text = segment.get("raw_text", "").lower()

    # Math indicators
    math_patterns = re.compile(
        r"equation|formula|integral|derivative|matrix|theorem|proof|"
        r"mathematical|calculus|algebra|polynomial|logarithm|exponential|"
        r"probability|permutation|combination|gaussian|distribution|"
        r"convergence|divergence|limit|series|summation|trigonometr",
        re.I,
    )
    # CS indicators
    cs_patterns = re.compile(
        r"function|variable|class |object|array|pointer|struct|"
        r"algorithm|program|code|compiler|syntax|operator|"
        r"inheritance|polymorphism|loop|linked.?list|stack|queue|"
        r"binary.?tree|sorting|searching|recursion|iteration|"
        r"std.?c.?out|header.?file|data.?type|access.?specifier",
        re.I,
    )

    math_score = len(math_patterns.findall(raw_text))
    cs_score = len(cs_patterns.findall(raw_text))

    if "Equation" in visual_tags:
        math_score += 5
    if "Computer_Code" in visual_tags:
        cs_score += 5

    if math_score > cs_score and math_score >= 2:
        return "Maths"
    elif cs_score > math_score and cs_score >= 2:
        return "Computer Science"
    else:
        return "General"


# ─────────────────────────────────────────────────────────────────────
# STEP 2: Filter & Split
# ─────────────────────────────────────────────────────────────────────

def filter_and_split(
    segments: List[Dict],
    subject_map: Dict[str, str],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Filter by quality and word count, split into Math/CS/General.
    Attaches 'subject' field to each segment.
    """
    math_segs, cs_segs, general_segs = [], [], []
    filtered_count = 0

    for seg in segments:
        qs = seg.get("quality_score", 0)
        wc = seg.get("word_count", 0)

        # Apply quality and length filters
        if qs < Config.MIN_QUALITY_SCORE or wc < Config.MIN_WORD_COUNT:
            filtered_count += 1
            continue

        # Determine subject
        lid = seg.get("lecture_id", "")
        if lid in subject_map:
            subject = subject_map[lid]
        else:
            subject = detect_subject_heuristic(seg)

        seg["subject"] = subject

        if subject == "Maths":
            math_segs.append(seg)
        elif subject == "Computer Science":
            cs_segs.append(seg)
        else:
            general_segs.append(seg)

    log.info(f"  Filtered out {filtered_count} segments (low quality/short)")
    log.info(f"  Math: {len(math_segs)} | CS: {len(cs_segs)} | General: {len(general_segs)}")
    return math_segs, cs_segs, general_segs


# ─────────────────────────────────────────────────────────────────────
# STEP 3: Smart Sampling
# ─────────────────────────────────────────────────────────────────────

def stratified_sample(
    segments: List[Dict],
    target_size: int,
    priority_tags: List[str],
    seed: int = 42,
) -> List[Dict]:
    """
    Sample segments with priority for those matching visual tags.
    Ensures diversity in the training data.
    """
    rng = random.Random(seed)

    # Separate priority vs non-priority
    priority = [s for s in segments if any(
        t in s.get("visual_tags", []) for t in priority_tags
    )]
    non_priority = [s for s in segments if s not in priority]

    rng.shuffle(priority)
    rng.shuffle(non_priority)

    # Take all priority first, then fill with non-priority
    sampled = priority[:target_size]
    remaining = target_size - len(sampled)
    if remaining > 0:
        sampled.extend(non_priority[:remaining])

    log.info(
        f"  Sampled {len(sampled)} segments "
        f"({len(sampled) - max(0, remaining)} priority, "
        f"{min(remaining, len(non_priority)) if remaining > 0 else 0} non-priority)"
    )
    return sampled


# ─────────────────────────────────────────────────────────────────────
# STEP 4: Gemini Teacher Augmentation
# ─────────────────────────────────────────────────────────────────────

from google import genai
from google.genai import types

def setup_gemini():
    """Initialize the new Gemini API client (google.genai)."""
    api_key = Config.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        log.warning("⚠️  No GEMINI_API_KEY found. Augmentation will be SKIPPED.")
        return None
    try:
        client = genai.Client(api_key=api_key)
        # Quick test
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'ready' in one word."
        )
        log.info(f"✅ Gemini API initialized (gemini-2.5-flash)")
        return client
    except Exception as e:
        log.error(f"❌ Gemini API error: {e}")
        return None
        return None


def build_augmentation_prompt(segment: Dict, subject: str) -> str:
    """Build the Gemini prompt for teacher augmentation."""
    raw_text = segment.get("raw_text", "")
    target_summary = segment.get("target_summary", "")
    visual_tags = ", ".join(segment.get("visual_tags", []))

    # Truncate transcript if very long (Gemini token limits)
    words = raw_text.split()
    if len(words) > 500:
        raw_text = " ".join(words[:500]) + "..."

    if subject == "Maths":
        return (
            "You are an expert mathematics lecturer creating study materials.\n\n"
            "Given this lecture transcript and its existing summary, "
            "rewrite the summary as a clear, step-by-step explanation that:\n"
            "1. Preserves ALL equations and mathematical notation EXACTLY as stated\n"
            "2. Explains the reasoning chain: definition → derivation → result\n"
            "3. Uses LaTeX notation for equations (e.g., $f(x) = x^2$) where possible\n"
            "4. Keeps ALL technical mathematical terms precise\n"
            "5. Maintains 3-6 sentences, concise but complete\n\n"
            f"TRANSCRIPT:\n{raw_text}\n\n"
            f"EXISTING SUMMARY:\n{target_summary}\n\n"
            f"VISUAL CONTEXT: {visual_tags}\n\n"
            "REWRITTEN STEP-BY-STEP SUMMARY:"
        )
    else:  # CS
        return (
            "You are an expert computer science instructor creating study notes.\n\n"
            "Given this lecture transcript and its existing summary, "
            "rewrite the summary as a clear, step-by-step explanation that:\n"
            "1. Preserves ALL function names, variable names, and code snippets EXACTLY\n"
            "2. Explains the logic flow: concept → implementation → output\n"
            "3. Uses `backtick` formatting for code-related terms\n"
            "4. Preserves operator names, syntax references, and language names accurately\n"
            "5. Maintains 3-6 sentences, concise but complete\n\n"
            f"TRANSCRIPT:\n{raw_text}\n\n"
            f"EXISTING SUMMARY:\n{target_summary}\n\n"
            f"VISUAL CONTEXT: {visual_tags}\n\n"
            "REWRITTEN STEP-BY-STEP SUMMARY:"
        )


def augment_single_segment(client, segment: Dict, subject: str) -> Optional[str]:
    """Call Gemini (new SDK) to augment a single segment."""
    prompt = build_augmentation_prompt(segment, subject)

    for attempt in range(Config.GEMINI_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=500,
                    top_p=0.9,
                    top_k=40,
                )
            )
            augmented = response.text.strip()

            if len(augmented.split()) < 30:
                log.warning(f"  Short augmentation ({len(augmented.split())} words), retrying...")
                continue
            return augmented

        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "rate" in error_str or "429" in error_str:
                wait = Config.GEMINI_RETRY_DELAY * (attempt + 1) * 2
                log.warning(f"  Rate limited, waiting {wait}s... (attempt {attempt+1})")
                time.sleep(wait)
            else:
                log.error(f"  Gemini error: {e} (attempt {attempt+1})")
                time.sleep(Config.GEMINI_RETRY_DELAY)
    return None


def augment_segments(
    model,
    segments: List[Dict],
    subject: str,
) -> List[Dict]:
    """
    Augment a list of segments with Gemini.
    Smart logic: skip segments with quality >= HIGH_QUALITY_THRESHOLD.
    Double-augment segments with quality < LOW_QUALITY_THRESHOLD.
    """
    results = []
    augmented_count = 0
    skipped_count = 0
    failed_count = 0

    total = len(segments)
    log.info(f"  Starting augmentation for {total} {subject} segments...")

    for i, seg in enumerate(segments):
        qs = seg.get("quality_score", 0)
        seg_id = seg.get("segment_id", f"seg_{i}")

        # Progress logging
        if (i + 1) % 10 == 0 or i == 0:
            log.info(f"  [{i+1}/{total}] Processing {seg_id} (quality={qs:.3f})")

        # HIGH QUALITY — skip augmentation, use target_summary directly
        if qs >= Config.HIGH_QUALITY_THRESHOLD:
            results.append({
                **seg,
                "augmented_summary": seg.get("target_summary", ""),
                "augmentation_method": "direct_high_quality",
            })
            skipped_count += 1
            continue

        # NEEDS AUGMENTATION
        if model is None:
            # No Gemini available — fallback to target_summary
            results.append({
                **seg,
                "augmented_summary": seg.get("target_summary", ""),
                "augmentation_method": "fallback_no_api",
            })
            skipped_count += 1
            continue

        augmented = augment_single_segment(model, seg, subject)

        if augmented:
            results.append({
                **seg,
                "augmented_summary": augmented,
                "augmentation_method": "gemini_augmented",
            })
            augmented_count += 1

            # LOW QUALITY — generate a second variant for diversity
            if qs < Config.LOW_QUALITY_THRESHOLD:
                augmented2 = augment_single_segment(model, seg, subject)
                if augmented2 and augmented2 != augmented:
                    results.append({
                        **seg,
                        "segment_id": seg_id + "_aug2",
                        "augmented_summary": augmented2,
                        "augmentation_method": "gemini_double_augmented",
                    })
                    augmented_count += 1
        else:
            # Gemini failed — use target_summary as fallback
            results.append({
                **seg,
                "augmented_summary": seg.get("target_summary", ""),
                "augmentation_method": "fallback_api_failed",
            })
            failed_count += 1

        # Rate limiting
        time.sleep(60.0 / Config.GEMINI_RPM_LIMIT)

    log.info(
        f"  Augmentation complete: "
        f"{augmented_count} augmented, {skipped_count} skipped (high-q/no-api), "
        f"{failed_count} fallback"
    )
    return results


# ─────────────────────────────────────────────────────────────────────
# STEP 5: Format as ChatML JSONL
# ─────────────────────────────────────────────────────────────────────

def format_chatml(segment: Dict, subject: str) -> Dict:
    """
    Format a single segment as ChatML message dict.
    Uses formatted_input (includes visual tags) as user message.
    """
    # Build user message with transcript + visual context
    raw_text = segment.get("raw_text", "")
    visual_tags = segment.get("visual_tags", [])

    # Build visual tag string
    visual_str = ""
    if visual_tags:
        tag_markers = []
        for tag in visual_tags:
            if tag == "Equation":
                tag_markers.append("[EQUATION]")
            elif tag == "Computer_Code":
                tag_markers.append("[CODE]")
            elif tag == "Graph_Chart":
                tag_markers.append("[GRAPH]")
            elif tag == "Diagrams":
                tag_markers.append("[DIAGRAM]")
            elif tag == "Question":
                tag_markers.append("[QUESTION]")
        visual_str = f"\n[VISUAL] {', '.join(visual_tags)} {' '.join(tag_markers)}"

    user_content = f"[TRANSCRIPT] {raw_text}{visual_str}"

    # Get the best available summary
    summary = segment.get("augmented_summary", segment.get("target_summary", ""))

    # Select system prompt
    if subject == "Maths":
        system_prompt = Config.MATH_SYSTEM_PROMPT
    else:
        system_prompt = Config.CS_SYSTEM_PROMPT

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": summary},
        ],
        # Keep metadata for debugging (won't be fed to model)
        "_metadata": {
            "segment_id": segment.get("segment_id", ""),
            "lecture_id": segment.get("lecture_id", ""),
            "quality_score": segment.get("quality_score", 0),
            "augmentation_method": segment.get("augmentation_method", "original"),
            "subject": subject,
            "visual_tags": visual_tags,
            "word_count": segment.get("word_count", 0),
        },
    }


def build_sft_dataset(
    segments: List[Dict],
    subject: str,
) -> List[Dict]:
    """
    Build the full SFT dataset from augmented segments.
    Also includes original target_summary variants for diversity.
    """
    dataset = []

    for seg in segments:
        # Primary: augmented version
        primary = format_chatml(seg, subject)
        dataset.append(primary)

        # Secondary: if augmented via Gemini, also include original target_summary
        # This gives the model exposure to both styles
        method = seg.get("augmentation_method", "")
        if method.startswith("gemini_"):
            original = seg.copy()
            original["augmented_summary"] = seg.get("target_summary", "")
            original["segment_id"] = seg.get("segment_id", "") + "_orig"
            original["augmentation_method"] = "original_target"
            secondary = format_chatml(original, subject)
            dataset.append(secondary)

    log.info(f"  Built {len(dataset)} ChatML examples for {subject}")
    return dataset


# ─────────────────────────────────────────────────────────────────────
# STEP 6: Train/Validation Split & Save
# ─────────────────────────────────────────────────────────────────────

def split_and_save(
    dataset: List[Dict],
    subject_tag: str,
    output_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42,
):
    """
    Stratified split and save as JSONL.
    Ensures segments from the same lecture stay in the same split.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group by lecture to prevent data leakage
    lecture_groups = {}
    for item in dataset:
        lid = item["_metadata"]["lecture_id"]
        lecture_groups.setdefault(lid, []).append(item)

    # Shuffle lectures, then split
    rng = random.Random(seed)
    lecture_ids = list(lecture_groups.keys())
    rng.shuffle(lecture_ids)

    # Calculate split point by lecture count
    split_idx = int(len(lecture_ids) * train_ratio)
    train_lectures = set(lecture_ids[:split_idx])
    val_lectures = set(lecture_ids[split_idx:])

    train_data = []
    val_data = []
    for lid, items in lecture_groups.items():
        if lid in train_lectures:
            train_data.extend(items)
        else:
            val_data.extend(items)

    # Shuffle within splits
    rng.shuffle(train_data)
    rng.shuffle(val_data)

    # Save
    train_path = os.path.join(output_dir, f"{subject_tag}_train.jsonl")
    val_path = os.path.join(output_dir, f"{subject_tag}_val.jsonl")

    def write_jsonl(data: List[Dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(train_data, train_path)
    write_jsonl(val_data, val_path)

    log.info(f"  {subject_tag} Train: {len(train_data)} examples → {train_path}")
    log.info(f"  {subject_tag} Val:   {len(val_data)} examples → {val_path}")

    return {
        "subject": subject_tag,
        "train_count": len(train_data),
        "val_count": len(val_data),
        "train_lectures": len(train_lectures),
        "val_lectures": len(val_lectures),
        "train_path": train_path,
        "val_path": val_path,
    }


# ─────────────────────────────────────────────────────────────────────
# STEP 7: Generate Data Report
# ─────────────────────────────────────────────────────────────────────

def generate_report(
    math_stats: Dict,
    cs_stats: Dict,
    math_raw_count: int,
    cs_raw_count: int,
    augmentation_stats: Dict,
    output_dir: str,
):
    """Generate a JSON report with all preparation statistics."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "min_quality_score": Config.MIN_QUALITY_SCORE,
            "min_word_count": Config.MIN_WORD_COUNT,
            "high_quality_threshold": Config.HIGH_QUALITY_THRESHOLD,
            "low_quality_threshold": Config.LOW_QUALITY_THRESHOLD,
            "math_sample_size": Config.MATH_SAMPLE_SIZE,
            "cs_sample_size": Config.CS_SAMPLE_SIZE,
            "train_ratio": Config.TRAIN_RATIO,
        },
        "source_data": {
            "math_segments_available": math_raw_count,
            "cs_segments_available": cs_raw_count,
        },
        "augmentation": augmentation_stats,
        "output": {
            "math": math_stats,
            "cs": cs_stats,
            "total_train": math_stats["train_count"] + cs_stats["train_count"],
            "total_val": math_stats["val_count"] + cs_stats["val_count"],
        },
    }

    report_path = os.path.join(output_dir, "data_prep_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'='*60}")
    log.info(f"📊 DATA PREPARATION REPORT")
    log.info(f"{'='*60}")
    log.info(f"Math: {math_stats['train_count']} train + {math_stats['val_count']} val")
    log.info(f"CS:   {cs_stats['train_count']} train + {cs_stats['val_count']} val")
    log.info(f"Total: {report['output']['total_train']} train + {report['output']['total_val']} val")
    log.info(f"Report saved: {report_path}")
    log.info(f"{'='*60}")

    return report


# ─────────────────────────────────────────────────────────────────────
# STEP 8: Validation — Quick sanity checks on output
# ─────────────────────────────────────────────────────────────────────

def validate_outputs(output_dir: str):
    """Validate JSONL outputs are well-formed."""
    log.info("\n🔍 Validating outputs...")
    issues = []

    for fname in ["math_train.jsonl", "math_val.jsonl", "cs_train.jsonl", "cs_val.jsonl"]:
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath):
            issues.append(f"Missing file: {fname}")
            continue

        line_count = 0
        empty_summaries = 0
        short_inputs = 0

        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    msgs = item.get("messages", [])
                    if len(msgs) != 3:
                        issues.append(f"{fname}:{line_num}: Expected 3 messages, got {len(msgs)}")
                        continue

                    user_msg = msgs[1]["content"]
                    asst_msg = msgs[2]["content"]

                    if len(asst_msg.split()) < 10:
                        empty_summaries += 1
                    if len(user_msg.split()) < 20:
                        short_inputs += 1

                    line_count += 1
                except json.JSONDecodeError:
                    issues.append(f"{fname}:{line_num}: Invalid JSON")

        log.info(f"  ✅ {fname}: {line_count} valid examples")
        if empty_summaries:
            log.warning(f"     ⚠️  {empty_summaries} examples with very short summaries (<10 words)")
        if short_inputs:
            log.warning(f"     ⚠️  {short_inputs} examples with very short inputs (<20 words)")

    if issues:
        log.error(f"\n❌ Validation issues found:")
        for issue in issues:
            log.error(f"   {issue}")
    else:
        log.info("  ✅ All outputs validated successfully!")


# ─────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("DAY 1 — SFT Data Preparation & Teacher Augmentation")
    log.info("=" * 60)

    random.seed(Config.RANDOM_SEED)

    # ── Step 1: Load data ──
    segments = load_inference_data(Config.INFERENCE_JSON)
    subject_map = load_subject_mapping(Config.ANNOTATIONS_DIR)

    # ── Step 2: Filter & Split ──
    log.info("\n📋 Step 2: Filtering and splitting by subject...")
    math_segs, cs_segs, general_segs = filter_and_split(segments, subject_map)

    # ── Step 3: Sample ──
    log.info("\n📋 Step 3: Stratified sampling...")
    log.info("  Math sampling:")
    math_sampled = stratified_sample(
        math_segs,
        target_size=Config.MATH_SAMPLE_SIZE,
        priority_tags=["Equation"],
        seed=Config.RANDOM_SEED,
    )
    log.info("  CS sampling:")
    cs_sampled = stratified_sample(
        cs_segs,
        target_size=Config.CS_SAMPLE_SIZE,
        priority_tags=["Computer_Code"],
        seed=Config.RANDOM_SEED + 1,
    )

    # ── Step 4: Augment ──
    log.info("\n📋 Step 4: Teacher augmentation with Gemini...")
    gemini_model = setup_gemini()

    augmentation_stats = {"gemini_available": gemini_model is not None}

    log.info("\n  --- Math Augmentation ---")
    math_augmented = augment_segments(gemini_model, math_sampled, "Maths")
    augmentation_stats["math_augmented"] = len([
        s for s in math_augmented if s.get("augmentation_method", "").startswith("gemini")
    ])
    augmentation_stats["math_skipped_high_quality"] = len([
        s for s in math_augmented if s.get("augmentation_method") == "direct_high_quality"
    ])

    log.info("\n  --- CS Augmentation ---")
    cs_augmented = augment_segments(gemini_model, cs_sampled, "Computer Science")
    augmentation_stats["cs_augmented"] = len([
        s for s in cs_augmented if s.get("augmentation_method", "").startswith("gemini")
    ])
    augmentation_stats["cs_skipped_high_quality"] = len([
        s for s in cs_augmented if s.get("augmentation_method") == "direct_high_quality"
    ])

    # ── Step 5: Build ChatML datasets ──
    log.info("\n📋 Step 5: Building ChatML datasets...")
    math_dataset = build_sft_dataset(math_augmented, "Maths")
    cs_dataset = build_sft_dataset(cs_augmented, "Computer Science")

    # ── Step 6: Split & Save ──
    log.info("\n📋 Step 6: Splitting and saving...")
    math_stats = split_and_save(
        math_dataset, "math", Config.OUTPUT_DIR,
        train_ratio=Config.TRAIN_RATIO, seed=Config.RANDOM_SEED,
    )
    cs_stats = split_and_save(
        cs_dataset, "cs", Config.OUTPUT_DIR,
        train_ratio=Config.TRAIN_RATIO, seed=Config.RANDOM_SEED,
    )

    # ── Step 7: Report ──
    log.info("\n📋 Step 7: Generating report...")
    generate_report(
        math_stats, cs_stats,
        len(math_segs), len(cs_segs),
        augmentation_stats,
        Config.OUTPUT_DIR,
    )

    # ── Step 8: Validate ──
    validate_outputs(Config.OUTPUT_DIR)

    log.info("\n✅ Day 1 data preparation complete!")
    log.info("=" * 60)
    log.info("NEXT STEPS:")
    log.info("  1. Download the JSONL files from Drive")
    log.info(f"     📂 {Config.OUTPUT_DIR}/")
    log.info("  2. Upload math_*.jsonl to Account #1 Drive for Day 2")
    log.info("  3. Upload cs_*.jsonl to Account #2 Drive for Day 3")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
