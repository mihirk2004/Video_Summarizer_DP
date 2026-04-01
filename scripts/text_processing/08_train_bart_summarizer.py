#!/usr/bin/env python3
"""
Step 4: Fine-tune BART for lecture summarization (with optional LoRA)
Loads tokenized dataset → trains with entity-aware loss → evaluates with ROUGE → saves model

Usage:
    python scripts/text_processing/08_train_bart_summarizer.py                    # Full fine-tuning
    python scripts/text_processing/08_train_bart_summarizer.py --lora             # LoRA fine-tuning
    python scripts/text_processing/08_train_bart_summarizer.py --lora --adapter-path models/text/bart_summarizer/scitldr_lora_adapter
    python scripts/text_processing/08_train_bart_summarizer.py --test             # 2 steps only
    python scripts/text_processing/08_train_bart_summarizer.py --small            # Use bart-base (4GB GPU)
    python scripts/text_processing/08_train_bart_summarizer.py --resume           # Resume from checkpoint

Requirements for LoRA:
    pip install peft
"""
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


# ======================================================================
# Custom data collator (handles entity_mask)
# ======================================================================
@dataclass
class SummarizationCollator:
    """Data collator that handles entity_mask alongside standard Seq2Seq fields"""
    pad_token_id: int
    max_target_length: int = 350

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Separate entity masks
        has_entity_mask = 'entity_mask' in features[0]
        entity_masks = [f.pop('entity_mask', None) for f in features] if has_entity_mask else None

        # Stack standard fields
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features]),
        }

        # Add entity masks
        if entity_masks is not None and entity_masks[0] is not None:
            batch['entity_mask'] = torch.stack(entity_masks)

        return batch


# ======================================================================
# Custom Trainer with entity-aware loss + LoRA support
# ======================================================================
class EntityAwareTrainer:
    """Custom training loop with entity-weighted cross-entropy and optional LoRA"""

    def __init__(self, config: Dict, model_name_override: Optional[str] = None,
                 test_mode: bool = False, resume: bool = False,
                 use_lora: bool = False, adapter_path: Optional[str] = None,lr_override=None):
        self.config = config
        self.test_mode = test_mode
        self.resume = resume
        self.use_lora = use_lora
        self.adapter_path = adapter_path
        self.logger = setup_logger("bart_trainer")

        # BART config
        bart = config.get('bart', {})
        self.model_name = model_name_override or bart.get('model_name', 'facebook/bart-large-cnn')
        self.max_input = bart.get('max_input_length', 1024)
        self.max_target = bart.get('max_target_length', 350)
        self.batch_size = bart.get('batch_size', 2)
        self.grad_accum = bart.get('gradient_accumulation_steps', 4)
        self.epochs = bart.get('epochs', 10)
        self.warmup_ratio = bart.get('warmup_ratio', 0.1)
        self.weight_decay = bart.get('weight_decay', 0.01)
        self.use_fp16 = bart.get('fp16', True) and torch.cuda.is_available()
        self.grad_ckpt = bart.get('gradient_checkpointing', True)
        self.patience = bart.get('early_stopping_patience', 5)
        self.num_beams = bart.get('num_beams', 4)
        self.min_target_len = bart.get('min_target_length', 30)
        self.label_smoothing = bart.get('label_smoothing', 0.1)

        # Entity-aware loss: DISABLE for LoRA mode (amplifies gradient instability)
        if self.use_lora:
            self.entity_weight = 1.0  # Standard CE loss for LoRA
            self.logger.info("Entity-aware loss DISABLED for LoRA mode (weight=1.0)")
        else:
            self.entity_weight = bart.get('entity_loss_weight', 1.5)

        # Learning rate selection:
        #   - Full fine-tuning: 1e-5 (all 406M params)
        #   - Fresh LoRA: 3e-4 (1.5M params, needs larger updates)
        #   - Pre-trained adapter: 5e-5 (continuing from learned weights, gentle)
        if lr_override is not None:
            self.lr = lr_override
        elif self.use_lora and self.adapter_path:
            self.lr = bart.get('lora_adapter_learning_rate', 5e-5)
        elif self.use_lora:
            self.lr = bart.get('lora_learning_rate', 3e-4)
        else:
            self.lr = bart.get('learning_rate', 1e-5)

        # LoRA config
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1

        # Paths
        self.data_dir = Path(config['paths']['data'].get(
            'bart_dataset', 'data/processed/bart_dataset'))
        self.model_dir = Path(config['paths']['models'].get(
            'bart', 'models/text/bart_summarizer'))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path("logs/bart_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Load data & model
    # ------------------------------------------------------------------
    def _load_dataset(self):
        """Load tokenized dataset from Step 3"""
        from scripts.text_processing.create_bart_dataset import LectureSummarizationDataset
        from transformers import AutoTokenizer

        ds_path = self.data_dir / "bart_tokenized_dataset.pkl"
        assert ds_path.exists(), f"Run 07_create_bart_dataset.py first! Missing {ds_path}"

        with open(ds_path, 'rb') as f:
            data = pickle.load(f)

        tok_path = data['tokenizer_path']
        # for colab
        tok_path = Path(tok_path).as_posix()
        tokenizer = AutoTokenizer.from_pretrained(tok_path)

        def make_ds(split_name):
            return LectureSummarizationDataset(
                data['splits'][split_name], tokenizer,
                data['max_input_length'], data['max_target_length']
            )

        return {
            'train': make_ds('train'),
            'val': make_ds('val'),
            'test': make_ds('test'),
            'tokenizer': tokenizer,
            'meta': data,
        }

    def _load_model(self, tokenizer):
        """Load BART model — optionally with LoRA adapters"""
        from transformers import BartForConditionalGeneration

        self.logger.info(f"Loading model: {self.model_name}")

        if self.use_lora and self.adapter_path:
            # Load LoRA adapter from intermediate fine-tuning (08a)
            from peft import PeftModel

            self.logger.info(f"Loading LoRA adapter from: {self.adapter_path}")
            base_model = BartForConditionalGeneration.from_pretrained(self.model_name)
            base_model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base_model, self.adapter_path, is_trainable=True)

            trainable, total = model.get_nb_trainable_parameters()
            self.logger.info(f"LoRA adapter loaded ✓ (from SciTLDR pre-training)")
            self.logger.info(f"  Trainable: {trainable:,} / {total:,} "
                             f"({100 * trainable / total:.2f}%)")

        elif self.use_lora:
            # Fresh LoRA (no pre-trained adapter)
            from peft import LoraConfig, get_peft_model, TaskType

            base_model = BartForConditionalGeneration.from_pretrained(self.model_name)
            base_model.resize_token_embeddings(len(tokenizer))

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                modules_to_save=["lm_head"],
            )
            model = get_peft_model(base_model, lora_config)

            trainable, total = model.get_nb_trainable_parameters()
            self.logger.info(f"Fresh LoRA applied ✓")
            self.logger.info(f"  Trainable: {trainable:,} / {total:,} "
                             f"({100 * trainable / total:.2f}%)")
        else:
            # Full fine-tuning (original behavior)
            model = BartForConditionalGeneration.from_pretrained(self.model_name)
            model.resize_token_embeddings(len(tokenizer))
            self.logger.info(f"Token embeddings resized to {len(tokenizer)}")

        # Gradient checkpointing
        if self.grad_ckpt and not self.use_lora:
            model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled ✓")

        return model.to(self.device)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_loss(self, model, batch):
        """Compute entity-aware cross-entropy loss with label smoothing"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        entity_mask = batch.get('entity_mask')

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Use standard loss when entity-aware is disabled
        if entity_mask is None or self.entity_weight <= 1.0:
            if self.label_smoothing > 0:
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(
                    reduction='mean', ignore_index=-100,
                    label_smoothing=self.label_smoothing
                )
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                return loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            return outputs.loss

        # Entity-aware weighted loss
        entity_mask = entity_mask.to(self.device).float()
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=-100,
            label_smoothing=self.label_smoothing
        )
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = entity_mask[:, 1:].contiguous()

        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())

        weights = 1.0 + shift_mask * (self.entity_weight - 1.0)
        valid = (shift_labels != -100).float()
        weighted_loss = (per_token_loss * weights * valid).sum() / valid.sum().clamp(min=1)

        return weighted_loss

    # ------------------------------------------------------------------
    # ROUGE evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, model, dataset, tokenizer, max_samples=None):
        """Evaluate with ROUGE metrics on full validation set"""
        import evaluate

        model.eval()
        rouge = evaluate.load("rouge")

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=SummarizationCollator(tokenizer.pad_token_id, self.max_target)
        )

        all_preds, all_labels = [], []

        # Override BART's built-in generation config
        model.generation_config.max_length = None
        model.generation_config.min_length = None
        model.generation_config.max_new_tokens = None

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_target,
                min_new_tokens=10,
                num_beams=self.num_beams,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=2.0,
                early_stopping=True,
            )

            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            labs = batch['labels'].clone()
            labs[labs == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labs, skip_special_tokens=True)

            all_preds.extend(preds)
            all_labels.extend(refs)

            if max_samples and len(all_preds) >= max_samples:
                break

        results = rouge.compute(predictions=all_preds, references=all_labels)

        # Log sample predictions
        n_show = min(3, len(all_preds))
        self.logger.info("─── Sample Predictions ───")
        for i in range(n_show):
            self.logger.info(f"  [REF {i+1}]: {all_labels[i][:200]}…")
            self.logger.info(f"  [PRED {i+1}]: {all_preds[i][:200]}…")
            self.logger.info("")

        model.train()
        return results

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("Starting BART Fine-Tuning")
        if self.use_lora:
            self.logger.info("Mode: LoRA" + (" + SciTLDR adapter" if self.adapter_path else ""))
        else:
            self.logger.info("Mode: Full fine-tuning")
        self.logger.info("=" * 60)

        # Load data & model
        data = self._load_dataset()
        tokenizer = data['tokenizer']
        model = self._load_model(tokenizer)

        train_ds = data['train']
        val_ds = data['val']

        collator = SummarizationCollator(tokenizer.pad_token_id, self.max_target)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=collator
        )

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = (len(train_loader) // self.grad_accum) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # Mixed precision
        scaler = torch.amp.GradScaler('cuda') if self.use_fp16 else None

        self.logger.info(f"Training samples: {len(train_ds)}")
        self.logger.info(f"Validation samples: {len(val_ds)}")
        self.logger.info(f"Batch size: {self.batch_size} x {self.grad_accum} accum = "
                         f"{self.batch_size * self.grad_accum}")
        self.logger.info(f"Total steps: {total_steps} ({self.epochs} epochs)")
        self.logger.info(f"Warmup steps: {warmup_steps}")
        self.logger.info(f"Learning rate: {self.lr}")
        self.logger.info(f"FP16: {self.use_fp16}")
        self.logger.info(f"Label smoothing: {self.label_smoothing}")
        self.logger.info(f"Entity loss weight: {self.entity_weight}")

        # Test mode: only 2 steps
        max_steps = 2 if self.test_mode else None

        # Training state
        best_rouge = 0.0
        patience_counter = 0
        history = []
        global_step = 0
        start_epoch = 1

        # Resume from checkpoint if requested
        if self.resume and not self.use_lora:
            start_epoch, best_rouge, history, global_step = self._load_checkpoint(
                model, optimizer, scheduler, scaler
            )
            patience_counter = 0

        for epoch in range(start_epoch, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")
            for step, batch in enumerate(pbar):
                # Forward pass
                if self.use_fp16:
                    with torch.amp.autocast('cuda'):
                        loss = self._compute_loss(model, batch)
                        loss = loss / self.grad_accum
                    scaler.scale(loss).backward()
                else:
                    loss = self._compute_loss(model, batch)
                    loss = loss / self.grad_accum
                    loss.backward()

                epoch_loss += loss.item() * self.grad_accum
                n_batches += 1

                # Gradient accumulation step
                if (step + 1) % self.grad_accum == 0:
                    if self.use_fp16:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                pbar.set_postfix({'loss': f'{epoch_loss / n_batches:.4f}',
                                  'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

                if max_steps and global_step >= max_steps:
                    break

            avg_loss = epoch_loss / max(n_batches, 1)
            self.logger.info(f"Epoch {epoch} — avg loss: {avg_loss:.4f}")

            # Validate (full validation set)
            self.logger.info("Running validation …")
            eval_max = 5 if self.test_mode else None
            rouge_scores = self._evaluate(model, val_ds, tokenizer, max_samples=eval_max)
            rouge2 = rouge_scores.get('rouge2', 0)
            rougeL = rouge_scores.get('rougeL', 0)

            self.logger.info(f"Epoch {epoch} — ROUGE-1: {rouge_scores.get('rouge1', 0):.4f}  "
                             f"ROUGE-2: {rouge2:.4f}  ROUGE-L: {rougeL:.4f}")

            history.append({
                'epoch': epoch,
                'train_loss': round(avg_loss, 4),
                'rouge1': round(rouge_scores.get('rouge1', 0), 4),
                'rouge2': round(rouge2, 4),
                'rougeL': round(rougeL, 4),
            })

            # Save best model
            if rouge2 > best_rouge:
                best_rouge = rouge2
                patience_counter = 0
                self._save_checkpoint(model, tokenizer, epoch, rouge_scores,
                                     optimizer, scheduler, scaler,
                                     best_rouge, history, global_step,
                                     is_best=True)
                self.logger.info(f"New best model saved! ROUGE-2: {rouge2:.4f}")
            else:
                patience_counter += 1
                self.logger.info(f"No improvement ({patience_counter}/{self.patience})")

            # Early stopping
            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

            if max_steps and global_step >= max_steps:
                self.logger.info("Test mode: stopping after max steps")
                break

        # Save training history
        self._save_history(history)

        self.logger.info("=" * 60)
        self.logger.info(f"Training complete! Best ROUGE-2: {best_rouge:.4f}")
        self.logger.info(f"Model saved to {self.model_dir}")
        self.logger.info("=" * 60)
        return history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, model, tokenizer, epoch, rouge_scores,
                         optimizer=None, scheduler=None, scaler=None,
                         best_rouge=0.0, history=None, global_step=0,
                         is_best=False):
        save_dir = self.model_dir / ("best_model" if is_best else f"checkpoint_epoch_{epoch}")
        save_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))

        meta = {
            'epoch': epoch,
            'global_step': global_step,
            'model_name': self.model_name,
            'best_rouge2': round(best_rouge, 4),
            'rouge_scores': {k: round(v, 4) for k, v in rouge_scores.items()},
            'history': history or [],
            'config': {
                'learning_rate': self.lr,
                'batch_size': self.batch_size,
                'gradient_accumulation': self.grad_accum,
                'entity_loss_weight': self.entity_weight,
                'label_smoothing': self.label_smoothing,
                'use_lora': self.use_lora,
                'adapter_path': self.adapter_path,
                'fp16': self.use_fp16,
            },
        }
        with open(save_dir / "training_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        # Save training state for resume (full fine-tuning only)
        if not self.use_lora:
            training_state = {
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'epoch': epoch,
                'global_step': global_step,
                'best_rouge': best_rouge,
                'history': history or [],
            }
            torch.save(training_state, save_dir / "training_state.pt")

        self.logger.info(f"Checkpoint saved → {save_dir}")

    def _load_checkpoint(self, model, optimizer, scheduler, scaler):
        """Load training state from the best checkpoint"""
        ckpt_dir = self.model_dir / "best_model"
        state_path = ckpt_dir / "training_state.pt"

        if not state_path.exists():
            self.logger.warning(f"No training_state.pt found at {ckpt_dir}, starting fresh")
            return 1, 0.0, [], 0

        self.logger.info(f"Resuming from checkpoint: {ckpt_dir}")
        state = torch.load(state_path, map_location=self.device, weights_only=False)

        from transformers import BartForConditionalGeneration
        loaded = BartForConditionalGeneration.from_pretrained(str(ckpt_dir))
        model.load_state_dict(loaded.state_dict())
        del loaded

        if state.get('optimizer_state_dict'):
            optimizer.load_state_dict(state['optimizer_state_dict'])
        if state.get('scheduler_state_dict'):
            scheduler.load_state_dict(state['scheduler_state_dict'])
        if scaler and state.get('scaler_state_dict'):
            scaler.load_state_dict(state['scaler_state_dict'])

        resume_epoch = state.get('epoch', 0) + 1
        best_rouge = state.get('best_rouge', 0.0)
        history = state.get('history', [])
        global_step = state.get('global_step', 0)

        self.logger.info(f"Resumed: epoch {resume_epoch}, global_step {global_step}, "
                         f"best ROUGE-2: {best_rouge:.4f}")
        return resume_epoch, best_rouge, history, global_step

    def _save_history(self, history):
        path = self.model_dir / "training_history.json"
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        self.logger.info(f"Training history → {path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune BART for lecture summarization")
    parser.add_argument("--test", action="store_true", help="Test mode (2 training steps)")
    parser.add_argument("--small", action="store_true",
                        help="Use bart-base instead of bart-large-cnn (fits 4GB GPU)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the last saved checkpoint")
    parser.add_argument("--lora", action="store_true",
                        help="Use LoRA (Low-Rank Adaptation) instead of full fine-tuning")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to pre-trained LoRA adapter (from 08a_intermediate_finetune.py)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()

    config = config_loader.load_all()

    model_override = "facebook/bart-base" if args.small else None

    trainer = EntityAwareTrainer(
        config,
        model_name_override=model_override,
        test_mode=args.test,
        resume=args.resume,
        use_lora=args.lora,
        adapter_path=args.adapter_path,
        lr_override=args.lr
    )
    history = trainer.train()

    print(f"\n{'=' * 50}")
    print("BART Training Complete!")
    if history:
        best = max(history, key=lambda h: h['rouge2'])
        print(f"  Mode:     {'LoRA' if args.lora else 'Full fine-tuning'}")
        print(f"  Best epoch:  {best['epoch']}")
        print(f"  Best ROUGE-1: {best['rouge1']}")
        print(f"  Best ROUGE-2: {best['rouge2']}")
        print(f"  Best ROUGE-L: {best['rougeL']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()