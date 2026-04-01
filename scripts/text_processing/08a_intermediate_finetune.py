#!/usr/bin/env python3
"""
Step 3a: Intermediate fine-tuning of BART on SciTLDR with LoRA
Downloads SciTLDR (scientific paper TL;DRs) → applies LoRA to BART → trains 1-2 epochs → saves adapter

This teaches BART academic/STEM summarization style BEFORE lecture fine-tuning.

Pipeline: BART-large-CNN → LoRA on SciTLDR → LoRA on lecture data (script 08)

Usage:
    python scripts/text_processing/08a_intermediate_finetune.py              # Full run
    python scripts/text_processing/08a_intermediate_finetune.py --test       # Quick test
    python scripts/text_processing/08a_intermediate_finetune.py --small      # bart-base (4GB GPU)

Requirements (install on Colab):
    pip install peft datasets evaluate
"""
import sys
import json
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
# Data collator for SciTLDR
# ======================================================================
@dataclass
class SciTLDRCollator:
    """Simple data collator for SciTLDR Seq2Seq training"""
    pad_token_id: int
    max_input_length: int = 1024
    max_target_length: int = 128

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features]),
        }
        return batch


# ======================================================================
# SciTLDR Dataset
# ======================================================================
class SciTLDRDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for SciTLDR"""

    def __init__(self, samples: List[Dict], tokenizer, max_input: int, max_target: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Tokenize input (paper abstract/source)
        inputs = self.tokenizer(
            item['source'],
            max_length=self.max_input,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        # Tokenize target (TL;DR summary)
        targets = self.tokenizer(
            item['target'],
            max_length=self.max_target,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        labels = targets['input_ids'].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels,
        }


# ======================================================================
# Intermediate Fine-Tuner
# ======================================================================
class IntermediateFineTuner:
    """LoRA fine-tune BART on SciTLDR before lecture training"""

    def __init__(self, config: Dict, model_name_override: Optional[str] = None,
                 test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.logger = setup_logger("intermediate_ft")

        # BART config
        bart = config.get('bart', {})
        self.model_name = model_name_override or bart.get('model_name', 'facebook/bart-large-cnn')
        self.max_input = 1024
        self.max_target = 128   # SciTLDR TL;DRs are short

        # Training hyperparams (lighter for intermediate step)
        self.lr = 3e-4          # LoRA uses higher LR than full fine-tuning
        self.batch_size = bart.get('batch_size', 2)
        self.grad_accum = bart.get('gradient_accumulation_steps', 4)
        self.epochs = 2         # Only 1-2 epochs needed for intermediate step
        self.warmup_ratio = 0.06
        self.weight_decay = 0.01
        self.use_fp16 = bart.get('fp16', True) and torch.cuda.is_available()

        # LoRA config
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1

        # Paths
        self.model_dir = Path(config['paths']['models'].get(
            'bart', 'models/text/bart_summarizer'))
        self.adapter_dir = self.model_dir / "scitldr_lora_adapter"
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Load SciTLDR dataset
    # ------------------------------------------------------------------
    def _load_scitldr(self):
        """Download and prepare SciTLDR dataset from HuggingFace"""
        from datasets import load_dataset

        self.logger.info("Downloading SciTLDR dataset from HuggingFace …")
        dataset = load_dataset("allenai/scitldr", "Abstract", trust_remote_code=True)

        def format_sample(example):
            """Format SciTLDR to source/target pairs"""
            # Source: join all source sentences
            source = " ".join(example['source'])
            # Target: use the first (author-written) TL;DR
            target = example['target'][0] if isinstance(example['target'], list) else example['target']
            return {'source': source, 'target': target}

        train_data = [format_sample(ex) for ex in dataset['train']]
        val_data = [format_sample(ex) for ex in dataset['validation']]

        if self.test_mode:
            train_data = train_data[:20]
            val_data = val_data[:10]

        self.logger.info(f"SciTLDR loaded: {len(train_data)} train, {len(val_data)} val")
        return train_data, val_data

    # ------------------------------------------------------------------
    # Load model with LoRA
    # ------------------------------------------------------------------
    def _load_model_with_lora(self, tokenizer):
        """Load BART and apply LoRA adapters"""
        from transformers import BartForConditionalGeneration
        from peft import LoraConfig, get_peft_model, TaskType

        self.logger.info(f"Loading model: {self.model_name}")
        model = BartForConditionalGeneration.from_pretrained(self.model_name)

        # Resize for special tokens
        bart_config = self.config.get('bart', {})
        special_tokens = bart_config.get('special_tokens', [
            '[TRANSCRIPT]', '[VISUAL]', '[EQUATION]', '[DIAGRAM]',
            '[CODE]', '[SLIDE]', '[GRAPH]',
        ])
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        model.resize_token_embeddings(len(tokenizer))

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            modules_to_save=["lm_head"],  # Keep lm_head trainable
        )

        model = get_peft_model(model, lora_config)

        trainable, total = model.get_nb_trainable_parameters()
        self.logger.info(f"LoRA applied ✓")
        self.logger.info(f"  Trainable params: {trainable:,} / {total:,} "
                         f"({100 * trainable / total:.2f}%)")

        return model.to(self.device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("Intermediate Fine-Tuning: BART + LoRA on SciTLDR")
        self.logger.info("=" * 60)

        # Load data
        train_data, val_data = self._load_scitldr()

        # Load tokenizer and model
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = self._load_model_with_lora(tokenizer)

        # Create datasets
        train_ds = SciTLDRDataset(train_data, tokenizer, self.max_input, self.max_target)
        val_ds = SciTLDRDataset(val_data, tokenizer, self.max_input, self.max_target)

        collator = SciTLDRCollator(tokenizer.pad_token_id, self.max_input, self.max_target)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=collator
        )

        # Optimizer (LoRA uses higher LR)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = (len(train_loader) // self.grad_accum) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        scaler = torch.amp.GradScaler('cuda') if self.use_fp16 else None

        self.logger.info(f"Training samples: {len(train_ds)}")
        self.logger.info(f"Validation samples: {len(val_ds)}")
        self.logger.info(f"Batch size: {self.batch_size} x {self.grad_accum} accum = "
                         f"{self.batch_size * self.grad_accum}")
        self.logger.info(f"Total steps: {total_steps} ({self.epochs} epochs)")
        self.logger.info(f"LoRA rank: {self.lora_r}, alpha: {self.lora_alpha}")

        max_steps = 5 if self.test_mode else None
        best_rouge = 0.0
        history = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")
            for step, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                if self.use_fp16:
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)
                        loss = outputs.loss / self.grad_accum
                    scaler.scale(loss).backward()
                else:
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                    loss = outputs.loss / self.grad_accum
                    loss.backward()

                epoch_loss += loss.item() * self.grad_accum
                n_batches += 1

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

                pbar.set_postfix({'loss': f'{epoch_loss / n_batches:.4f}',
                                  'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

                if max_steps and (step + 1) >= max_steps:
                    break

            avg_loss = epoch_loss / max(n_batches, 1)
            self.logger.info(f"Epoch {epoch} — avg loss: {avg_loss:.4f}")

            # Quick validation
            self.logger.info("Running validation …")
            rouge_scores = self._evaluate(model, val_ds, tokenizer,
                                          max_samples=10 if self.test_mode else 50)
            rouge2 = rouge_scores.get('rouge2', 0)
            rougeL = rouge_scores.get('rougeL', 0)

            self.logger.info(f"Epoch {epoch} — ROUGE-2: {rouge2:.4f}  ROUGE-L: {rougeL:.4f}")

            history.append({
                'epoch': epoch,
                'train_loss': round(avg_loss, 4),
                'rouge2': round(rouge2, 4),
                'rougeL': round(rougeL, 4),
            })

            if rouge2 > best_rouge:
                best_rouge = rouge2

        # Save LoRA adapter (not the full model — just the adapter weights)
        self.logger.info(f"Saving LoRA adapter → {self.adapter_dir}")
        model.save_pretrained(str(self.adapter_dir))
        tokenizer.save_pretrained(str(self.adapter_dir))

        # Save metadata
        meta = {
            'base_model': self.model_name,
            'dataset': 'allenai/scitldr',
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'epochs': self.epochs,
            'best_rouge2': round(best_rouge, 4),
            'history': history,
        }
        with open(self.adapter_dir / "adapter_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        self.logger.info("=" * 60)
        self.logger.info(f"Intermediate fine-tuning complete! Best ROUGE-2: {best_rouge:.4f}")
        self.logger.info(f"LoRA adapter saved to: {self.adapter_dir}")
        self.logger.info("=" * 60)
        return history

    # ------------------------------------------------------------------
    # ROUGE evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, model, dataset, tokenizer, max_samples=50):
        """Quick ROUGE evaluation"""
        import evaluate

        model.eval()
        rouge = evaluate.load("rouge")

        collator = SciTLDRCollator(tokenizer.pad_token_id, self.max_input, self.max_target)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collator
        )

        all_preds, all_labels = [], []

        # Clear BART's built-in generation config
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
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True,
            )

            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            labs = batch['labels'].clone()
            labs[labs == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labs, skip_special_tokens=True)

            all_preds.extend(preds)
            all_labels.extend(refs)

            if len(all_preds) >= max_samples:
                break

        # Log sample predictions
        n_show = min(2, len(all_preds))
        self.logger.info("─── Sample Predictions (SciTLDR) ───")
        for i in range(n_show):
            self.logger.info(f"  [REF {i+1}]: {all_labels[i][:200]}…")
            self.logger.info(f"  [PRED {i+1}]: {all_preds[i][:200]}…")

        results = rouge.compute(predictions=all_preds, references=all_labels)
        model.train()
        return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Intermediate LoRA fine-tuning of BART on SciTLDR")
    parser.add_argument("--test", action="store_true", help="Test mode (limited steps)")
    parser.add_argument("--small", action="store_true",
                        help="Use bart-base instead of bart-large-cnn")
    args = parser.parse_args()

    config = config_loader.load_all()
    model_override = "facebook/bart-base" if args.small else None

    tuner = IntermediateFineTuner(config, model_name_override=model_override,
                                  test_mode=args.test)
    history = tuner.train()

    print(f"\n{'=' * 50}")
    print("SciTLDR Intermediate Fine-Tuning Complete!")
    if history:
        best = max(history, key=lambda h: h['rouge2'])
        print(f"  Best ROUGE-2: {best['rouge2']}")
        print(f"  Best ROUGE-L: {best['rougeL']}")
    print(f"  LoRA adapter saved for lecture fine-tuning (Step 08)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
