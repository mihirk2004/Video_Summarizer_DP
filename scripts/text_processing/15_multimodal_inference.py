#!/usr/bin/env python3
"""
Phase 4 — Step 5: Multimodal Inference Pipeline
Runs the trained CLIP+BART fusion model on lecture segments.
Optionally integrates LLaVA for frame captioning (requires --llava flag).

Input:
    - data/processed/multimodal_dataset/multimodal_segments.json
    - models/multimodal/clip_bart_fusion/best_model/
    - data/processed/multimodal_dataset/clip_embeddings/

Output:
    - results/multimodal_inference.json

Usage:
    python scripts/text_processing/15_multimodal_inference.py                  # Fusion only
    python scripts/text_processing/15_multimodal_inference.py --test           # 10 segments
    python scripts/text_processing/15_multimodal_inference.py --llava          # + LLaVA captions
    python scripts/text_processing/15_multimodal_inference.py --lecture lecture_107  # Single lecture
    python scripts/text_processing/15_multimodal_inference.py --compare        # Fusion vs text-only

Requirements:
    pip install transformers peft accelerate
"""
import sys
import gc
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger

# Import fusion architecture — module starts with digit, need importlib
import importlib
_fusion_module = importlib.import_module("scripts.text_processing.14_train_clip_bart_fusion")
CLIPBARTFusion = _fusion_module.CLIPBARTFusion
VisualProjection = _fusion_module.VisualProjection
FusionDataset = _fusion_module.FusionDataset



class MultimodalInferencePipeline:
    """End-to-end multimodal inference: CLIP+BART fusion + optional LLaVA"""

    def __init__(self, config: Dict, test_mode: bool = False,
                 use_llava: bool = False, lecture_filter: str = None):
        self.config = config
        self.test_mode = test_mode
        self.use_llava = use_llava
        self.lecture_filter = lecture_filter
        self.logger = setup_logger("multimodal_inference")

        # Paths
        mm_paths = config['paths'].get('multimodal', {})
        self.dataset_path = Path(mm_paths.get(
            'dataset', 'data/processed/multimodal_dataset')) / "multimodal_segments.json"
        self.embeddings_dir = Path(mm_paths.get(
            'clip_embeddings', 'data/processed/multimodal_dataset/clip_embeddings'))
        self.fusion_model_dir = Path(mm_paths.get(
            'fusion_model', 'models/multimodal/clip_bart_fusion')) / "best_model"
        self.output_dir = Path(config['paths']['outputs'].get('results', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model config
        fusion_cfg = config.get('fusion', {})
        self.clip_dim = fusion_cfg.get('clip_embed_dim', 512)
        self.bart_dim = fusion_cfg.get('bart_embed_dim', 1024)
        self.hidden_dim = fusion_cfg.get('projection_hidden_dim', 768)
        self.max_images = fusion_cfg.get('max_images_per_segment', 3)
        self.max_input = fusion_cfg.get('max_input_length', 1024)
        self.max_target = fusion_cfg.get('max_target_length', 350)
        self.num_beams = fusion_cfg.get('num_beams', 4)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy loaded
        self._fusion_model = None
        self._tokenizer = None
        self._llava_model = None
        self._llava_processor = None
        self._sbert = None

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    def _load_fusion_model(self):
        """Load trained CLIP+BART fusion model"""
        if self._fusion_model is not None:
            return

        from transformers import BartForConditionalGeneration, AutoTokenizer
        from peft import PeftModel

        self.logger.info(f"Loading fusion model from {self.fusion_model_dir} ...")

        # Load metadata
        meta_path = self.fusion_model_dir / "fusion_meta.json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        bart_base = meta.get('bart_base', 'facebook/bart-large-cnn')
        bart_lora_dir = self.fusion_model_dir / "bart_lora"

        # Load BART with LoRA
        self._tokenizer = AutoTokenizer.from_pretrained(str(bart_lora_dir))
        base_bart = BartForConditionalGeneration.from_pretrained(bart_base)
        base_bart.resize_token_embeddings(len(self._tokenizer))
        bart_model = PeftModel.from_pretrained(
            base_bart, str(bart_lora_dir), is_trainable=False)

        # Build fusion model
        self._fusion_model = CLIPBARTFusion(
            bart_model=bart_model,
            clip_dim=meta['clip_dim'],
            bart_dim=meta['bart_dim'],
            hidden_dim=meta['hidden_dim'],
            max_images=meta['max_images'],
        )

        # Load projection weights
        proj_path = self.fusion_model_dir / "projection.pt"
        proj_state = torch.load(str(proj_path), map_location='cpu')
        self._fusion_model.visual_projection.load_state_dict(proj_state['visual_projection'])
        self._fusion_model.visual_gate.data = proj_state['visual_gate']

        self._fusion_model.eval()
        self._fusion_model.to(self.device)
        self.logger.info("  Fusion model loaded ✓")

    def _load_llava(self):
        """Load LLaVA for frame captioning"""
        if self._llava_model is not None:
            return

        from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        llava_cfg = self.config.get('llava', {})
        model_name = llava_cfg.get('model_name', 'llava-hf/llava-1.5-7b-hf')

        self.logger.info(f"Loading LLaVA: {model_name} (4-bit) ...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Check for LoRA adapter
        llava_adapter_dir = Path(self.config['paths'].get('multimodal', {}).get(
            'llava_adapter', 'models/multimodal/llava_lora'))

        self._llava_processor = AutoProcessor.from_pretrained(model_name)
        self._llava_model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        if llava_adapter_dir.exists():
            from peft import PeftModel
            self._llava_model = PeftModel.from_pretrained(
                self._llava_model, str(llava_adapter_dir))
            self.logger.info("  LLaVA LoRA adapter loaded ✓")

        self.logger.info("  LLaVA loaded ✓")

    def _load_sbert(self):
        if self._sbert is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._sbert = SentenceTransformer('all-mpnet-base-v2', device='cpu')

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_fusion_inference(self, segments: List[Dict]) -> List[Dict]:
        """Run CLIP+BART fusion model on all segments"""
        self._load_fusion_model()

        self.logger.info(f"Running fusion inference on {len(segments)} segments ...")

        dataset = FusionDataset(
            segments, self.embeddings_dir, self._tokenizer,
            max_images=self.max_images, clip_dim=self.clip_dim,
            max_input_length=self.max_input, max_target_length=self.max_target,
        )

        for i in tqdm(range(len(segments)), desc="Fusion inference"):
            batch = dataset[i]
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
            clip_embeddings = batch['clip_embeddings'].unsqueeze(0).to(self.device)
            image_mask = batch['image_mask'].unsqueeze(0).to(self.device)

            t0 = time.time()
            gen_ids = self._fusion_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                clip_embeddings=clip_embeddings,
                image_mask=image_mask,
                max_new_tokens=self.max_target,
                min_new_tokens=10,
                num_beams=self.num_beams,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=2.0,
                early_stopping=True,
            )

            summary = self._tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            segments[i]['fusion_summary'] = summary
            segments[i]['fusion_time'] = round(time.time() - t0, 3)
            segments[i]['final_summary'] = summary

        return segments

    @torch.no_grad()
    def run_llava_captions(self, segments: List[Dict]) -> List[Dict]:
        """Generate captions for visual frames using LLaVA"""
        self._load_llava()

        from PIL import Image

        PROMPTS = {
            "Equation": "Describe the mathematical equation or formula shown. If possible, write it in LaTeX notation.",
            "Diagrams": "Describe this technical diagram. What concept is being illustrated?",
            "Computer_Code": "Describe the code shown. What programming language and what does it do?",
            "Graph_Chart": "Describe this graph or chart. What trends are shown?",
            "Table": "Describe the data in this table. What are the key data points?",
            "Flow_diagram": "Describe this flow diagram. What process does it represent?",
            "Slide_Presentation": "Describe the key points on this presentation slide.",
            "Question": "What question is being shown or discussed?",
        }

        llava_cfg = self.config.get('llava', {})
        max_new_tokens = llava_cfg.get('max_new_tokens', 256)

        visual_segments = [s for s in segments if s.get('has_visuals')]
        self.logger.info(f"Captioning frames for {len(visual_segments)} visual segments ...")

        for seg in tqdm(visual_segments, desc="LLaVA captions"):
            captions = []
            for img_path, category in zip(seg.get('image_paths', []), seg.get('image_categories', [])):
                try:
                    image = Image.open(img_path).convert('RGB')
                except Exception:
                    captions.append({"path": img_path, "category": category, "caption": ""})
                    continue

                prompt_text = PROMPTS.get(category, "Describe this image from a lecture.")
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ]
                prompt = self._llava_processor.apply_chat_template(
                    conversation, add_generation_prompt=True)

                inputs = self._llava_processor(
                    text=prompt, images=image, return_tensors="pt"
                ).to(self._llava_model.device)

                output = self._llava_model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False)

                caption = self._llava_processor.decode(
                    output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                captions.append({
                    "path": img_path,
                    "category": category,
                    "caption": caption.strip(),
                })

            seg['frame_captions'] = captions

        return segments

    def score_quality(self, segments: List[Dict]) -> List[Dict]:
        """Score summaries with SBERT cosine similarity"""
        self._load_sbert()
        for seg in tqdm(segments, desc="Quality scoring"):
            transcript = seg['raw_text']
            summary = seg.get('final_summary', '')
            if summary:
                embs = self._sbert.encode([transcript, summary], convert_to_tensor=True)
                score = F.cosine_similarity(
                    embs[0].unsqueeze(0), embs[1].unsqueeze(0)).item()
                seg['quality_score'] = round(score, 4)
            else:
                seg['quality_score'] = 0.0
        return segments

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self, compare: bool = False) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info("Multimodal Inference Pipeline")
        self.logger.info("=" * 60)

        t_start = time.time()

        # Load segments
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data['segments']

        if self.lecture_filter:
            segments = [s for s in segments if s['lecture_id'] == self.lecture_filter]
            self.logger.info(f"Filtered to lecture {self.lecture_filter}: {len(segments)} segments")

        if self.test_mode:
            segments = segments[:10]

        # Run fusion inference
        segments = self.run_fusion_inference(segments)

        # Optional: LLaVA captions
        if self.use_llava:
            # Free fusion model first
            del self._fusion_model
            self._fusion_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            segments = self.run_llava_captions(segments)

        # Quality scoring
        segments = self.score_quality(segments)

        elapsed = time.time() - t_start

        # Statistics
        avg_quality = np.mean([s.get('quality_score', 0) for s in segments])
        avg_time = np.mean([s.get('fusion_time', 0) for s in segments])
        visual_segs = sum(1 for s in segments if s.get('has_visuals'))

        stats = {
            'total_segments': len(segments),
            'visual_segments': visual_segs,
            'avg_quality': round(avg_quality, 4),
            'avg_time_per_segment': round(avg_time, 3),
            'total_time': round(elapsed, 1),
            'with_llava': self.use_llava,
        }

        # ROUGE if target summaries exist
        try:
            import evaluate
            rouge = evaluate.load("rouge")
            refs = [s.get('target_summary', '') for s in segments]
            preds = [s.get('final_summary', '') for s in segments]
            if any(refs):
                rouge_results = rouge.compute(predictions=preds, references=refs)
                stats['rouge'] = {k: round(v, 4) for k, v in rouge_results.items()}
        except Exception as e:
            self.logger.warning(f"ROUGE computation failed: {e}")

        # Save results
        output = {'segments': segments, 'statistics': stats}
        out_path = self.output_dir / "multimodal_inference.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\nResults saved → {out_path}")
        self.logger.info(f"  Avg Quality: {stats['avg_quality']}")
        self.logger.info(f"  ROUGE: {stats.get('rouge', 'N/A')}")
        self.logger.info(f"  Time: {stats['total_time']}s")

        return stats


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4: Multimodal Inference")
    parser.add_argument("--test", action="store_true", help="Test mode (10 segments)")
    parser.add_argument("--llava", action="store_true", help="Include LLaVA frame captions")
    parser.add_argument("--lecture", type=str, default=None, help="Filter to one lecture")
    parser.add_argument("--compare", action="store_true", help="Compare fusion vs text-only")
    args = parser.parse_args()

    config = config_loader.load_all()
    pipeline = MultimodalInferencePipeline(
        config, test_mode=args.test, use_llava=args.llava,
        lecture_filter=args.lecture)
    results = pipeline.run(compare=args.compare)

    print(f"\n{'=' * 50}")
    print("Multimodal Inference Complete!")
    print(f"  Quality: {results.get('avg_quality', 'N/A')}")
    print(f"  ROUGE: {results.get('rouge', 'N/A')}")
    print(f"  Time: {results.get('total_time', 'N/A')}s")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
