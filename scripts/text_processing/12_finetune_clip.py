#!/usr/bin/env python3
"""
Phase 4 — Step 2: Fine-tune CLIP with LoRA on Lecture Text-Image Pairs
Contrastive learning: aligns lecture segment text ↔ lecture frame images.

Input:
    - data/processed/multimodal_dataset/multimodal_segments.json

Output:
    - models/multimodal/clip_finetuned/  (LoRA adapter weights)

Usage:
    python scripts/text_processing/12_finetune_clip.py               # Full training
    python scripts/text_processing/12_finetune_clip.py --test         # 2 epochs, small data
    python scripts/text_processing/12_finetune_clip.py --epochs 5     # Custom epochs

Requirements:
    pip install transformers peft accelerate pillow
"""
import sys
import gc
import json
import math
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from PIL import Image
# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class CLIPLectureDataset(Dataset):
    """Dataset of (text, image) pairs from multimodal segments"""

    def __init__(self, pairs: List[Dict], processor, max_text_length: int = 77):
        self.pairs = pairs
        self.processor = processor
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load and process image
        try:
            image = Image.open(pair['image_path']).convert('RGB')
        except Exception:
            # Fallback: create a blank image
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Truncate text for CLIP tokenizer
        text = pair['text'][:300]  # rough truncation before tokenization

        text_inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )

        return {
            'pixel_values': pixel_values,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
        }


# ------------------------------------------------------------------
# Contrastive Loss
# ------------------------------------------------------------------
class CLIPContrastiveLoss(nn.Module):
    """Symmetric InfoNCE loss for CLIP training"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(math.log(1 / temperature)))

    def forward(self, image_embeds, text_embeds):
        # Normalize
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Cosine similarity matrix
        temperature = self.temperature.exp().clamp(max=100)
        logits = image_embeds @ text_embeds.t() * temperature

        # Labels: diagonal is positive pairs
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        # Symmetric loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        return (loss_i2t + loss_t2i) / 2


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------
class CLIPFineTuner:
    """Fine-tune CLIP with LoRA on lecture text-image pairs"""

    def __init__(self, config: Dict, test_mode: bool = False, epochs_override: int = None):
        self.config = config
        self.test_mode = test_mode
        self.logger = setup_logger("clip_finetune")

        # Config
        clip_cfg = config.get('clip', {})
        self.model_name = clip_cfg.get('model_name', 'openai/clip-vit-base-patch32')
        self.embed_dim = clip_cfg.get('embed_dim', 512)
        self.lora_r = clip_cfg.get('lora_r', 8)
        self.lora_alpha = clip_cfg.get('lora_alpha', 16)
        self.lr = clip_cfg.get('learning_rate', 1e-4)
        self.weight_decay = clip_cfg.get('weight_decay', 0.01)
        self.batch_size = clip_cfg.get('batch_size', 16)
        self.grad_accum = clip_cfg.get('gradient_accumulation', 2)
        self.epochs = epochs_override or clip_cfg.get('epochs', 10)
        self.warmup_ratio = clip_cfg.get('warmup_ratio', 0.1)
        self.patience = clip_cfg.get('early_stopping_patience', 3)
        self.temperature = clip_cfg.get('temperature', 0.07)
        self.max_text_length = clip_cfg.get('max_text_length', 77)

        if test_mode:
            self.epochs = min(self.epochs, 2)
            self.batch_size = min(self.batch_size, 4)

        # Paths
        self.dataset_path = Path(config['paths'].get('multimodal', {}).get(
            'dataset', 'data/processed/multimodal_dataset')) / "multimodal_segments.json"
        self.output_dir = Path(config['paths'].get('multimodal', {}).get(
            'clip_model', 'models/multimodal/clip_finetuned'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_pairs(self) -> Tuple[List[Dict], List[Dict]]:
        """Create text-image pairs from multimodal segments"""
        self.logger.info(f"Loading multimodal segments from {self.dataset_path} ...")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data['segments']
        pairs = []

        for seg in segments:
            if not seg.get('image_paths'):
                continue

            text = seg['raw_text'][:300]  # CLIP has limited text context

            for img_path in seg['image_paths']:
                if Path(img_path).exists():
                    pairs.append({
                        'text': text,
                        'image_path': img_path,
                        'lecture_id': seg['lecture_id'],
                        'segment_id': seg['segment_id'],
                    })

        self.logger.info(f"  Total text-image pairs: {len(pairs)}")

        if self.test_mode:
            pairs = pairs[:100]
            self.logger.info(f"  Test mode: using {len(pairs)} pairs")

        # Split: 85% train, 15% val
        np.random.seed(42)
        indices = np.random.permutation(len(pairs))
        split = int(0.85 * len(pairs))
        train_pairs = [pairs[i] for i in indices[:split]]
        val_pairs = [pairs[i] for i in indices[split:]]

        self.logger.info(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")
        return train_pairs, val_pairs

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    def _setup_model(self):
        """Load CLIP model and apply LoRA"""
        from transformers import CLIPModel, CLIPProcessor
        from peft import LoraConfig, get_peft_model

        self.logger.info(f"Loading CLIP model: {self.model_name} ...")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)

        # Apply LoRA to both text and vision encoders
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # attention projections
            lora_dropout=0.1,
            bias="none",
            modules_to_save=None,
        )

        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        self.model.to(self.device)

        # Loss
        self.criterion = CLIPContrastiveLoss(temperature=self.temperature).to(self.device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("CLIP Fine-tuning with LoRA (Contrastive Learning)")
        self.logger.info("=" * 60)

        t_start = time.time()

        # Load data
        train_pairs, val_pairs = self._load_pairs()
        if not train_pairs:
            self.logger.error("No training pairs found! Run 11_prepare_multimodal_data.py first.")
            return

        # Setup model
        self._setup_model()

        # Datasets & loaders
        train_dataset = CLIPLectureDataset(train_pairs, self.processor, self.max_text_length)
        val_dataset = CLIPLectureDataset(val_pairs, self.processor, self.max_text_length)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=2, pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=2, pin_memory=True,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad] +
            list(self.criterion.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * self.epochs // self.grad_accum
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(total_steps, 1), T_mult=1,
        )

        # Training
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_recall': []}

        scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        for epoch in range(self.epochs):
            # --- Train ---
            self.model.train()
            train_losses = []
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for step, batch in enumerate(pbar):
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        loss = self.criterion(
                            outputs.image_embeds, outputs.text_embeds
                        )
                        loss = loss / self.grad_accum

                    scaler.scale(loss).backward()

                    if (step + 1) % self.grad_accum == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    scheduler.step()
                else:
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    loss = self.criterion(
                        outputs.image_embeds, outputs.text_embeds
                    )
                    loss = loss / self.grad_accum
                    loss.backward()

                    if (step + 1) % self.grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    scheduler.step()

                train_losses.append(loss.item() * self.grad_accum)
                pbar.set_postfix({'loss': f"{np.mean(train_losses[-50:]):.4f}"})

            avg_train_loss = np.mean(train_losses)

            # --- Validate ---
            val_loss, val_recall = self._validate(val_loader)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_recall'].append(val_recall)

            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} — "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val R@1: {val_recall:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model(epoch, val_loss, val_recall)
                self.logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

        elapsed = time.time() - t_start

        # Save training history
        history_path = self.output_dir / "best_model" / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        self.logger.info(f"\nCLIP fine-tuning complete! Time: {elapsed:.0f}s")
        self.logger.info(f"Best val loss: {best_val_loss:.4f}")
        self.logger.info(f"Model saved → {self.output_dir}")

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Compute validation loss and Recall@1"""
        self.model.eval()
        losses = []
        all_image_embeds = []
        all_text_embeds = []

        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = self.criterion(outputs.image_embeds, outputs.text_embeds)
            losses.append(loss.item())

            all_image_embeds.append(F.normalize(outputs.image_embeds, dim=-1).cpu())
            all_text_embeds.append(F.normalize(outputs.text_embeds, dim=-1).cpu())

        # Recall@1
        img_emb = torch.cat(all_image_embeds, dim=0)
        txt_emb = torch.cat(all_text_embeds, dim=0)
        sim = img_emb @ txt_emb.t()
        recall_at_1 = (sim.argmax(dim=1) == torch.arange(len(sim))).float().mean().item()

        return np.mean(losses), recall_at_1

    def _save_model(self, epoch: int, val_loss: float, val_recall: float):
        """Save LoRA adapter and processor"""
        self.model.save_pretrained(str(self.output_dir / "best_model"))
        self.processor.save_pretrained(str(self.output_dir / "best_model"))

        # Save metadata
        meta = {
            'epoch': epoch,
            'val_loss': val_loss,
            'val_recall_at_1': val_recall,
            'base_model': self.model_name,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
        }
        with open(self.output_dir / "best_model" / "training_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4: Fine-tune CLIP with LoRA")
    parser.add_argument("--test", action="store_true", help="Test mode (2 epochs, small data)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    config = config_loader.load_all()
    trainer = CLIPFineTuner(config, test_mode=args.test, epochs_override=args.epochs)
    trainer.train()


if __name__ == "__main__":
    main()
