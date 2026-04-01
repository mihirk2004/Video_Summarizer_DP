#!/usr/bin/env python3
"""
Phase 4 — Step 4: Train CLIP+BART Prefix Fusion Model
Prepends projected CLIP visual embeddings as prefix tokens to BART encoder input.
Uses LoRA on BART for parameter-efficient training.

Architecture:
    CLIP visual embeddings → Linear Projection → [IMG] prefix tokens
    → [IMG_1][IMG_2][IMG_3][TRANSCRIPT] text tokens → BART Encoder → BART Decoder → Summary

Input:
    - data/processed/multimodal_dataset/multimodal_segments.json
    - data/processed/multimodal_dataset/clip_embeddings/{lecture_id}.npz

Output:
    - models/multimodal/clip_bart_fusion/best_model/

Usage:
    python scripts/text_processing/14_train_clip_bart_fusion.py               # Full training
    python scripts/text_processing/14_train_clip_bart_fusion.py --test        # 2 epochs, small
    python scripts/text_processing/14_train_clip_bart_fusion.py --epochs 5    # Custom epochs

Requirements:
    pip install transformers peft accelerate datasets evaluate
"""
import sys
import gc
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math  # ensure at top of file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


# ======================================================================
# Model Architecture
# ======================================================================
class VisualProjection(nn.Module):
    """Project CLIP embeddings (512) → BART embedding space (1024)"""

    def __init__(self, clip_dim: int = 512, bart_dim: int = 1024,
                 hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bart_dim),
            nn.LayerNorm(bart_dim),
        )
        # Learnable visual type embedding
        self.visual_type_embed = nn.Parameter(torch.randn(1, 1, bart_dim) * 0.02)

    def forward(self, clip_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_embeddings: (batch, n_images, clip_dim)
        Returns:
            (batch, n_images, bart_dim) projected embeddings
        """
        projected = self.projection(clip_embeddings)
        projected = projected + self.visual_type_embed
        return projected


class CLIPBARTFusion(nn.Module):
    """
    Prefix Fusion: prepend projected CLIP visual tokens before BART text tokens.
    The encoder attends to both visual and text tokens.
    The decoder cross-attends to the full encoder output.
    """


    def __init__(
        self,
        bart_model,
        clip_dim: int = 512,
        bart_dim: int = 1024,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        max_images: int = 3,
    ):
        super().__init__()
        self.bart = bart_model

        # Unwrap LoRA to get the underlying BartForConditionalGeneration
        if hasattr(bart_model, 'base_model'):          # PeftModel
            base = bart_model.base_model
            if hasattr(base, 'model'):                 # LoraModel -> original model
                self.bart_model = base.model
            else:
                self.bart_model = base
        else:
            self.bart_model = bart_model

        # Get the encoder
        self.encoder = self.bart_model.get_encoder()

        # Extract encoder components safely
        self.embed_tokens = self.encoder.embed_tokens
        # embed_scale might be missing; compute from config as BART does
        config = self.bart_model.config
        if hasattr(config, 'embed_scale'):
            self.embed_scale = config.embed_scale
        else:
            self.embed_scale = math.sqrt(config.d_model)
        self.embed_positions = self.encoder.embed_positions
        self.layernorm_embedding = self.encoder.layernorm_embedding
        self.encoder_layers = self.encoder.layers
        self.max_positions = self.embed_positions.weight.size(0)

        # Visual projection and gate
        self.visual_projection = VisualProjection(clip_dim, bart_dim, hidden_dim, dropout)
        self.max_images = max_images
        self.visual_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, attention_mask, clip_embeddings, image_mask, labels=None):
        device = input_ids.device

        # 1. Text embeddings (now using stored components)
        text_embeds = self.embed_tokens(input_ids)
        text_embeds = text_embeds * self.embed_scale

        # 2. Visual prefix
        visual_embeds = self.visual_projection(clip_embeddings)
        gate = torch.sigmoid(self.visual_gate)
        visual_embeds = visual_embeds * gate

        # 3. Concatenate
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # 4. Combined attention mask
        combined_mask = torch.cat([image_mask, attention_mask], dim=1)

        # 5. Positional embeddings (with offset)
        seq_len = combined_embeds.size(1)
        positions = torch.arange(2, seq_len + 2, device=device).unsqueeze(0).expand(
            combined_embeds.size(0), -1
        )
        positions = positions.clamp(max=self.max_positions - 1)
        pos_embeds = self.embed_positions.weight[positions]
        combined_embeds = combined_embeds + pos_embeds

        # 6. LayerNorm before encoder
        combined_embeds = self.layernorm_embedding(combined_embeds)

        # 7. Manual encoder forward (because we have visual prefix)
        hidden_states = combined_embeds
        for layer in self.encoder_layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=self._expand_mask(combined_mask, hidden_states.dtype),
            )
            hidden_states = layer_outputs[0]
        encoder_output = hidden_states

        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_output)

        # 8. Decoder forward
        decoder_outputs = self.bart(
            encoder_outputs=encoder_outputs,
            attention_mask=combined_mask,
            labels=labels,
        )
        return decoder_outputs

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Expand attention mask from (B, L) to (B, 1, 1, L) for attention layers"""
        expanded = mask[:, None, None, :].to(dtype)
        expanded = (1.0 - expanded) * torch.finfo(dtype).min
        return expanded

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        clip_embeddings: torch.Tensor,
        image_mask: torch.Tensor,
        **generate_kwargs,
    ):
        """Generate summaries with visual prefix"""
        device = input_ids.device

        # Encode with visual prefix
        text_embeds = self.embed_tokens(input_ids) * self.embed_scale

        visual_embeds = self.visual_projection(clip_embeddings)
        gate = torch.sigmoid(self.visual_gate)
        visual_embeds = visual_embeds * gate

        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        combined_mask = torch.cat([image_mask, attention_mask], dim=1)

        seq_len = combined_embeds.size(1)
        positions = torch.arange(2, seq_len + 2, device=device).unsqueeze(0).expand(
            combined_embeds.size(0), -1)
        max_positions = self.embed_positions.weight.size(0)
        positions = positions.clamp(max=max_positions - 1)
        pos_embeds = self.embed_positions.weight[positions]
        combined_embeds = combined_embeds + pos_embeds

        combined_embeds = self.layernorm_embedding(combined_embeds)

        hidden_states = self.layernorm_embedding(combined_embeds)
        for layer in self.encoder_layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=self._expand_mask(combined_mask, hidden_states.dtype),
            )
            hidden_states = layer_outputs[0]

        encoder_output = hidden_states

        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_output)

        # Use BART's generate with pre-computed encoder output
        generated = self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=combined_mask,
            **generate_kwargs,
        )
        return generated


# ======================================================================
# Dataset
# ======================================================================
class FusionDataset(Dataset):
    """Dataset for CLIP+BART fusion training"""

    def __init__(
        self,
        segments: List[Dict],
        embeddings_dir: Path,
        tokenizer,
        max_images: int = 3,
        clip_dim: int = 512,
        max_input_length: int = 1024,
        max_target_length: int = 350,
    ):
        self.segments = segments
        self.embeddings_dir = embeddings_dir
        self.tokenizer = tokenizer
        self.max_images = max_images
        self.clip_dim = clip_dim
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # Pre-load all embeddings into memory
        self._embedding_cache = {}
        self._load_embeddings()

    def _load_embeddings(self):
        """Load all .npz files into memory"""
        for npz_file in self.embeddings_dir.glob("*.npz"):
            lecture_id = npz_file.stem
            data = np.load(str(npz_file), allow_pickle=True)
            self._embedding_cache[lecture_id] = {
                'embeddings': data['embeddings'],
                'frame_indices': data['frame_indices'],
            }

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]

        # --- Text ---
        formatted_input = seg.get('formatted_input', seg['raw_text'])
        target = seg.get('target_summary', '')

        # Tokenize input
        text_enc = self.tokenizer(
            formatted_input,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize target
        if hasattr(self.tokenizer, 'as_target_tokenizer'):
            with self.tokenizer.as_target_tokenizer():
                target_enc = self.tokenizer(
                    target,
                    max_length=self.max_target_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
        else:
            target_enc = self.tokenizer(
                target,
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        labels = target_enc['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        # --- Images ---
        clip_embeddings = torch.zeros(self.max_images, self.clip_dim)
        image_mask = torch.zeros(self.max_images)

        lecture_id = seg['lecture_id']
        frame_indices = seg.get('image_frame_indices', [])

        if lecture_id in self._embedding_cache and frame_indices:
            cache = self._embedding_cache[lecture_id]
            cached_indices = cache['frame_indices']
            cached_embeds = cache['embeddings']

            n_filled = 0
            for fi in frame_indices[:self.max_images]:
                # Find this frame index in the cached embeddings
                mask = (cached_indices == fi)
                if mask.any():
                    pos = np.where(mask)[0][0]
                    clip_embeddings[n_filled] = torch.from_numpy(
                        cached_embeds[pos].astype(np.float32))
                    image_mask[n_filled] = 1.0
                    n_filled += 1

        return {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'clip_embeddings': clip_embeddings,
            'image_mask': image_mask,
            'labels': labels,
        }


# ======================================================================
# Trainer
# ======================================================================
class CLIPBARTTrainer:
    """Train the CLIP+BART prefix fusion model"""

    def __init__(self, config: Dict, test_mode: bool = False, epochs_override: int = None):
        self.config = config
        self.test_mode = test_mode
        self.logger = setup_logger("clip_bart_fusion")

        # Config
        fusion_cfg = config.get('fusion', {})
        self.bart_base = fusion_cfg.get('bart_base', 'facebook/bart-large-cnn')
        self.use_finetuned = fusion_cfg.get('use_finetuned_bart', True)
        self.clip_dim = fusion_cfg.get('clip_embed_dim', 512)
        self.bart_dim = fusion_cfg.get('bart_embed_dim', 1024)
        self.hidden_dim = fusion_cfg.get('projection_hidden_dim', 768)
        self.proj_dropout = fusion_cfg.get('projection_dropout', 0.1)
        self.max_images = fusion_cfg.get('max_images_per_segment', 3)
        self.lr = fusion_cfg.get('learning_rate', 2e-5)
        self.lora_lr = fusion_cfg.get('lora_learning_rate', 1e-4)
        self.lora_r = fusion_cfg.get('lora_r', 8)
        self.lora_alpha = fusion_cfg.get('lora_alpha', 16)
        self.batch_size = fusion_cfg.get('batch_size', 4)
        self.grad_accum = fusion_cfg.get('gradient_accumulation', 4)
        self.epochs = epochs_override or fusion_cfg.get('epochs', 10)
        self.warmup_ratio = fusion_cfg.get('warmup_ratio', 0.1)
        self.patience = fusion_cfg.get('early_stopping_patience', 4)
        self.fp16 = fusion_cfg.get('fp16', True)
        self.grad_checkpoint = fusion_cfg.get('gradient_checkpointing', True)
        self.max_input = fusion_cfg.get('max_input_length', 1024)
        self.max_target = fusion_cfg.get('max_target_length', 350)
        self.label_smoothing = fusion_cfg.get('label_smoothing', 0.1)

        if test_mode:
            self.epochs = min(self.epochs, 2)
            self.batch_size = min(self.batch_size, 2)

        # Paths
        mm_paths = config['paths'].get('multimodal', {})
        self.dataset_path = Path(mm_paths.get(
            'dataset', 'data/processed/multimodal_dataset')) / "multimodal_segments.json"
        self.embeddings_dir = Path(mm_paths.get(
            'clip_embeddings', 'data/processed/multimodal_dataset/clip_embeddings'))
        self.bart_model_dir = Path(config['paths']['models'].get(
            'bart', 'models/text/bart_summarizer'))
        self.output_dir = Path(mm_paths.get(
            'fusion_model', 'models/multimodal/clip_bart_fusion'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load and split multimodal segments"""
        self.logger.info(f"Loading data from {self.dataset_path} ...")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data['segments']

        # Filter segments that have target summaries
        valid = [s for s in segments if s.get('target_summary', '').strip()]
        self.logger.info(f"  Total segments: {len(segments)}, with summaries: {len(valid)}")

        if self.test_mode:
            valid = valid[:50]

        # Split by lecture
        np.random.seed(42)
        lecture_ids = sorted(set(s['lecture_id'] for s in valid))
        np.random.shuffle(lecture_ids)
        split = int(0.85 * len(lecture_ids))
        train_lecs = set(lecture_ids[:split])
        val_lecs = set(lecture_ids[split:])

        train_segs = [s for s in valid if s['lecture_id'] in train_lecs]
        val_segs = [s for s in valid if s['lecture_id'] in val_lecs]

        self.logger.info(f"  Train: {len(train_segs)} segs ({len(train_lecs)} lectures)")
        self.logger.info(f"  Val: {len(val_segs)} segs ({len(val_lecs)} lectures)")

        return train_segs, val_segs

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _build_model(self, tokenizer):
        """Build CLIPBARTFusion model with LoRA on BART"""
        from transformers import BartForConditionalGeneration
        from peft import LoraConfig, get_peft_model

        # Load BART
        finetuned_path = self.bart_model_dir / "best_model"
        if self.use_finetuned and finetuned_path.exists():
            self.logger.info(f"Loading fine-tuned BART from {finetuned_path} ...")
            # Check if it's already a LoRA model
            adapter_config = finetuned_path / "adapter_config.json"
            if adapter_config.exists():
                base_model = BartForConditionalGeneration.from_pretrained(self.bart_base)
                from peft import PeftModel
                base_model.resize_token_embeddings(len(tokenizer))
                bart = PeftModel.from_pretrained(
                    base_model, str(finetuned_path), is_trainable=False)
                bart = bart.merge_and_unload()
                self.logger.info("  Merged existing LoRA adapters ✓")
            else:
                bart = BartForConditionalGeneration.from_pretrained(str(finetuned_path))
                bart.resize_token_embeddings(len(tokenizer))
        else:
            self.logger.info(f"Loading base BART: {self.bart_base} ...")
            bart = BartForConditionalGeneration.from_pretrained(self.bart_base)
            bart.resize_token_embeddings(len(tokenizer))

        # Enable gradient checkpointing
        if self.grad_checkpoint:
            bart.gradient_checkpointing_enable()

        # Apply LoRA to BART
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        bart = get_peft_model(bart, lora_config)

        trainable = sum(p.numel() for p in bart.parameters() if p.requires_grad)
        total = sum(p.numel() for p in bart.parameters())
        self.logger.info(f"  BART LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # Build fusion model
        model = CLIPBARTFusion(
            bart_model=bart,
            clip_dim=self.clip_dim,
            bart_dim=self.bart_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.proj_dropout,
            max_images=self.max_images,
        )

        # Log projection params
        proj_params = sum(p.numel() for p in model.visual_projection.parameters())
        self.logger.info(f"  Visual projection params: {proj_params:,}")

        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("CLIP+BART Prefix Fusion Training")
        self.logger.info("=" * 60)

        t_start = time.time()

        # Load tokenizer
        from transformers import AutoTokenizer
        finetuned_path = self.bart_model_dir / "best_model"
        if self.use_finetuned and finetuned_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(finetuned_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.bart_base)

        # Ensure special tokens
        special = ["[TRANSCRIPT]", "[VISUAL]", "[EQUATION]", "[DIAGRAM]",
                    "[CODE]", "[SLIDE]", "[GRAPH]", "[TABLE]", "[QUESTION]"]
        tokenizer.add_special_tokens({'additional_special_tokens': special})

        # Load data
        train_segs, val_segs = self._load_data()

        # Build model
        model = self._build_model(tokenizer)
        model.to(self.device)

        # Datasets
        train_dataset = FusionDataset(
            train_segs, self.embeddings_dir, tokenizer,
            max_images=self.max_images, clip_dim=self.clip_dim,
            max_input_length=self.max_input, max_target_length=self.max_target,
        )
        val_dataset = FusionDataset(
            val_segs, self.embeddings_dir, tokenizer,
            max_images=self.max_images, clip_dim=self.clip_dim,
            max_input_length=self.max_input, max_target_length=self.max_target,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=2, pin_memory=True,
        )

        # Optimizer: different LR for projection vs LoRA
        projection_params = list(model.visual_projection.parameters()) + [model.visual_gate]
        lora_params = [p for n, p in model.bart.named_parameters() if p.requires_grad and 'lora' in n]
        other_params = [p for n, p in model.bart.named_parameters()
                        if p.requires_grad and 'lora' not in n]

        optimizer = torch.optim.AdamW([
            {'params': projection_params, 'lr': self.lr * 5},  # Projection learns faster
            {'params': lora_params, 'lr': self.lora_lr},
            {'params': other_params, 'lr': self.lr},
        ], weight_decay=0.01)

        # Scheduler
        total_steps = len(train_loader) * self.epochs // self.grad_accum
        warmup_steps = int(total_steps * self.warmup_ratio)

        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        # Mixed precision
        scaler = torch.amp.GradScaler('cuda') if self.fp16 and self.device.type == 'cuda' else None

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            # --- Train ---
            model.train()
            train_losses = []
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for step, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                clip_embeddings = batch['clip_embeddings'].to(self.device)
                image_mask = batch['image_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            clip_embeddings=clip_embeddings,
                            image_mask=image_mask,
                            labels=labels,
                        )
                        loss = outputs.loss / self.grad_accum

                    scaler.scale(loss).backward()

                    if (step + 1) % self.grad_accum == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        clip_embeddings=clip_embeddings,
                        image_mask=image_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.grad_accum
                    loss.backward()

                    if (step + 1) % self.grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                train_losses.append(loss.item() * self.grad_accum)
                pbar.set_postfix({'loss': f"{np.mean(train_losses[-50:]):.4f}"})

            avg_train_loss = np.mean(train_losses)

            # --- Validate ---
            val_loss = self._validate(model, val_loader)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)

            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} — "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early stopping + checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model(model, tokenizer, epoch, val_loss)
                self.logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

        elapsed = time.time() - t_start

        # Save history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Generate sample outputs
        self._generate_samples(model, tokenizer, val_segs[:3])

        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"Fusion Training Complete!")
        self.logger.info(f"  Best val loss: {best_val_loss:.4f}")
        self.logger.info(f"  Time: {elapsed:.0f}s")
        self.logger.info(f"  Model: {self.output_dir}")
        self.logger.info(f"{'=' * 50}")

    @torch.no_grad()
    def _validate(self, model, val_loader):
        model.eval()
        losses = []
        for batch in val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            clip_embeddings = batch['clip_embeddings'].to(self.device)
            image_mask = batch['image_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                clip_embeddings=clip_embeddings,
                image_mask=image_mask,
                labels=labels,
            )
            losses.append(outputs.loss.item())

        return np.mean(losses)

    @torch.no_grad()
    def _generate_samples(self, model, tokenizer, sample_segs):
        """Generate sample summaries for logging"""
        model.eval()
        self.logger.info("\n─── Sample Outputs ───")

        dataset = FusionDataset(
            sample_segs, self.embeddings_dir, tokenizer,
            max_images=self.max_images, clip_dim=self.clip_dim,
            max_input_length=self.max_input, max_target_length=self.max_target,
        )

        for i, seg in enumerate(sample_segs):
            batch = dataset[i]
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
            clip_embeddings = batch['clip_embeddings'].unsqueeze(0).to(self.device)
            image_mask = batch['image_mask'].unsqueeze(0).to(self.device)

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                clip_embeddings=clip_embeddings,
                image_mask=image_mask,
                max_new_tokens=self.max_target,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True,
            )

            generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            self.logger.info(f"\n  [{seg['segment_id']}] visuals={seg.get('visual_tags', [])}")
            self.logger.info(f"  TARGET: {seg.get('target_summary', '')[:200]}")
            self.logger.info(f"  GENERATED: {generated[:200]}")

    def _save_model(self, model, tokenizer, epoch, val_loss):
        """Save the fusion model"""
        save_dir = self.output_dir / "best_model"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save BART (with LoRA)
        model.bart.save_pretrained(str(save_dir / "bart_lora"))
        tokenizer.save_pretrained(str(save_dir / "bart_lora"))

        # Save projection layer
        torch.save({
            'visual_projection': model.visual_projection.state_dict(),
            'visual_gate': model.visual_gate.data,
        }, str(save_dir / "projection.pt"))

        # Save metadata
        meta = {
            'epoch': epoch,
            'val_loss': val_loss,
            'clip_dim': self.clip_dim,
            'bart_dim': self.bart_dim,
            'hidden_dim': self.hidden_dim,
            'max_images': self.max_images,
            'bart_base': self.bart_base,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
        }
        with open(save_dir / "fusion_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4: Train CLIP+BART Prefix Fusion")
    parser.add_argument("--test", action="store_true", help="Test mode (2 epochs, small data)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    config = config_loader.load_all()
    trainer = CLIPBARTTrainer(config, test_mode=args.test, epochs_override=args.epochs)
    trainer.train()


if __name__ == "__main__":
    main()
