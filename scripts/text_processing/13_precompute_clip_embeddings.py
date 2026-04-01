#!/usr/bin/env python3
"""
Phase 4 — Step 3: Precompute CLIP Image Embeddings
Encode all selected frames with the fine-tuned CLIP model and cache as .npz files.

Input:
    - data/processed/multimodal_dataset/multimodal_segments.json
    - models/multimodal/clip_finetuned/best_model/  (LoRA adapter)

Output:
    - data/processed/multimodal_dataset/clip_embeddings/{lecture_id}.npz

Usage:
    python scripts/text_processing/13_precompute_clip_embeddings.py           # Full run
    python scripts/text_processing/13_precompute_clip_embeddings.py --test    # 5 lectures
    python scripts/text_processing/13_precompute_clip_embeddings.py --use-base  # Use pretrained CLIP (skip LoRA)

Requirements:
    pip install transformers peft pillow
"""
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


class CLIPEmbeddingPrecomputer:
    """Precompute and cache CLIP image embeddings for all selected frames"""

    def __init__(self, config: Dict, test_mode: bool = False, use_base: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.use_base = use_base
        self.logger = setup_logger("clip_embeddings")

        # Paths
        self.dataset_path = Path(config['paths'].get('multimodal', {}).get(
            'dataset', 'data/processed/multimodal_dataset')) / "multimodal_segments.json"
        self.clip_model_dir = Path(config['paths'].get('multimodal', {}).get(
            'clip_model', 'models/multimodal/clip_finetuned')) / "best_model"
        self.output_dir = Path(config['paths'].get('multimodal', {}).get(
            'clip_embeddings', 'data/processed/multimodal_dataset/clip_embeddings'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Config
        clip_cfg = config.get('clip', {})
        self.model_name = clip_cfg.get('model_name', 'openai/clip-vit-base-patch32')
        self.embed_dim = clip_cfg.get('embed_dim', 512)
        self.batch_size = 32  # Large batch OK since inference only

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        """Load CLIP model (fine-tuned or base)"""
        from transformers import CLIPModel, CLIPProcessor

        if self.use_base or not self.clip_model_dir.exists():
            if not self.use_base:
                self.logger.warning(
                    f"Fine-tuned CLIP not found at {self.clip_model_dir}, using base model")
            self.logger.info(f"Loading base CLIP: {self.model_name} ...")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        else:
            self.logger.info(f"Loading fine-tuned CLIP from {self.clip_model_dir} ...")
            from peft import PeftModel
            base_model = CLIPModel.from_pretrained(self.model_name)
            self.model = PeftModel.from_pretrained(base_model, str(self.clip_model_dir))
            self.model = self.model.merge_and_unload()  # Merge LoRA for speed
            self.processor = CLIPProcessor.from_pretrained(str(self.clip_model_dir))
            self.logger.info("  LoRA adapters merged ✓")

        self.model.eval()
        self.model.to(self.device)
        self.logger.info("  CLIP model loaded ✓")

    def _collect_frames_per_lecture(self) -> Dict[str, List[Dict]]:
        """Group frames by lecture from multimodal segments"""
        self.logger.info(f"Loading segments from {self.dataset_path} ...")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        lecture_frames = defaultdict(dict)  # lecture_id -> {frame_idx: frame_info}

        for seg in data['segments']:
            lecture_id = seg['lecture_id']
            for i, path in enumerate(seg.get('image_paths', [])):
                frame_idx = seg['image_frame_indices'][i]
                if frame_idx not in lecture_frames[lecture_id]:
                    lecture_frames[lecture_id][frame_idx] = {
                        'path': path,
                        'frame_index': frame_idx,
                        'timestamp': seg['image_timestamps'][i],
                        'category': seg['image_categories'][i],
                    }

        # Convert to sorted lists
        result = {}
        for lec_id, frames_dict in lecture_frames.items():
            result[lec_id] = sorted(frames_dict.values(), key=lambda x: x['frame_index'])

        total_frames = sum(len(v) for v in result.values())
        self.logger.info(f"  Unique frames to embed: {total_frames} across {len(result)} lectures")

        return result

    @torch.no_grad()
    def precompute(self):
        self.logger.info("=" * 60)
        self.logger.info("CLIP Embedding Precomputation")
        self.logger.info("=" * 60)

        t_start = time.time()

        # Load model
        self._load_model()

        # Collect frames
        lecture_frames = self._collect_frames_per_lecture()

        if self.test_mode:
            keys = sorted(lecture_frames.keys())[:5]
            lecture_frames = {k: lecture_frames[k] for k in keys}
            self.logger.info(f"Test mode: processing {len(lecture_frames)} lectures")

        # Process each lecture
        from PIL import Image

        total_embedded = 0
        skipped = 0

        for lecture_id, frames in tqdm(lecture_frames.items(), desc="Embedding lectures"):
            out_path = self.output_dir / f"{lecture_id}.npz"

            # Skip if already computed
            if out_path.exists() and not self.test_mode:
                total_embedded += len(frames)
                continue

            embeddings = []
            frame_indices = []
            timestamps = []
            categories = []
            valid_paths = []

            # Batch encode images
            for batch_start in range(0, len(frames), self.batch_size):
                batch_frames = frames[batch_start:batch_start + self.batch_size]
                images = []
                batch_valid = []

                for frame_info in batch_frames:
                    try:
                        img = Image.open(frame_info['path']).convert('RGB')
                        images.append(img)
                        batch_valid.append(frame_info)
                    except Exception:
                        skipped += 1
                        continue

                if not images:
                    continue

                pixel_values = self.processor(
                    images=images, return_tensors="pt"
                ).pixel_values.to(self.device)

                image_features = self.model.get_image_features(pixel_values=pixel_values)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                embeddings.append(image_features.cpu().numpy().astype(np.float16))

                for info in batch_valid:
                    frame_indices.append(info['frame_index'])
                    timestamps.append(info['timestamp'])
                    categories.append(info['category'])
                    valid_paths.append(info['path'])

            if embeddings:
                all_embeddings = np.concatenate(embeddings, axis=0)

                np.savez_compressed(
                    str(out_path),
                    embeddings=all_embeddings,
                    frame_indices=np.array(frame_indices, dtype=np.int32),
                    timestamps=np.array(timestamps, dtype=np.float32),
                    categories=np.array(categories),
                    paths=np.array(valid_paths),
                )

                total_embedded += len(all_embeddings)

        elapsed = time.time() - t_start

        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"Embedding Precomputation Complete!")
        self.logger.info(f"  Frames embedded: {total_embedded}")
        self.logger.info(f"  Frames skipped:  {skipped}")
        self.logger.info(f"  Output dir:      {self.output_dir}")
        self.logger.info(f"  Time: {elapsed:.0f}s")
        self.logger.info(f"{'=' * 50}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4: Precompute CLIP image embeddings")
    parser.add_argument("--test", action="store_true", help="Test mode (5 lectures)")
    parser.add_argument("--use-base", action="store_true",
                        help="Use base CLIP instead of fine-tuned")
    args = parser.parse_args()

    config = config_loader.load_all()
    precomputer = CLIPEmbeddingPrecomputer(
        config, test_mode=args.test, use_base=args.use_base)
    precomputer.precompute()


if __name__ == "__main__":
    main()
