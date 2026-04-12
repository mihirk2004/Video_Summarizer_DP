#!/usr/bin/env python3
"""
=============================================================================
Day 2 — Math LoRA SFT (DeepSeek-Math-7B-Instruct)
=============================================================================
Fine-tune DeepSeek-Math-7B with LoRA on the Math SFT dataset from Day 1.

Features:
  - 4-bit NF4 quantization (fits T4 12GB VRAM)
  - LoRA r=16, alpha=32 on all attention + MLP projections
  - Gradient checkpointing + paged AdamW 8-bit
  - Cosine annealing scheduler
  - Milestone checkpoints at 50% and 75% of training
  - Resume from checkpoint on Colab reconnect
  - Early stopping (patience=3 evaluations)
  - Comprehensive training logs saved as JSON

Run on Google Colab (T4 GPU):
  1. Upload this script to Colab
  2. Ensure math_train.jsonl & math_val.jsonl are on Drive
  3. Run: !python day2_math_sft.py

Author: Auto-generated for DP Project fine-tuning pipeline
=============================================================================
"""

import json
import os
import sys
import time
import gc
import logging
import glob
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """All hyperparameters and paths in one place."""

    # ── Paths (Colab default) ──
    data_dir: str = "/content/drive/MyDrive/Colab_text_modelling_dp_01/data/processed/sft_data"
    output_dir: str = "/content/drive/MyDrive/Colab_text_modelling_dp_01/lora_math"
    logging_dir: str = "/content/drive/MyDrive/Colab_text_modelling_dp_01/lora_math/logs"

    # ── Model ──
    base_model: str = "deepseek-ai/deepseek-math-7b-instruct"
    trust_remote_code: bool = True

    # ── Quantization ──
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    use_double_quant: bool = True

    # ── LoRA ──
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_bias: str = "none"

    # ── Training Hyperparameters ──
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4      # Effective batch = 2*4 = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"

    # ── Sequence Length ──
    max_seq_length: int = 1024

    # ── Evaluation & Saving ──
    eval_steps: int = 25
    save_steps: int = 50
    logging_steps: int = 5
    save_total_limit: int = 3

    # ── Milestone Checkpoints (fraction of total steps) ──
    milestone_pcts: list = field(default_factory=lambda: [0.50, 0.75])

    # ── Early Stopping ──
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.005

    # ── Gradient Checkpointing ──
    gradient_checkpointing: bool = True

    # ── Misc ──
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 2
    group_by_length: bool = True
    report_to: str = "none"
    resume_from_checkpoint: bool = True


# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("day2_math_sft")


# ─────────────────────────────────────────────────────────────────────
# STEP 1: Install Dependencies (for Colab)
# ─────────────────────────────────────────────────────────────────────

def install_dependencies():
    """Install required packages in Colab environment."""
    log.info("Checking and installing dependencies...")
    try:
        import transformers, peft, trl, bitsandbytes, accelerate, datasets
        log.info("  All packages already installed")
    except ImportError:
        log.info("  Installing missing packages...")
        os.system(
            "pip install -q "
            "transformers>=4.40.0 "
            "peft>=0.10.0 "
            "trl>=0.8.0 "
            "bitsandbytes>=0.43.0 "
            "accelerate>=0.28.0 "
            "datasets>=2.18.0 "
            "scipy"
        )
        log.info("  Packages installed")


# ─────────────────────────────────────────────────────────────────────
# STEP 2: Load SFT Data
# ─────────────────────────────────────────────────────────────────────

def load_sft_data(config: TrainingConfig):
    """Load the JSONL SFT data produced by Day 1."""
    from datasets import Dataset

    train_path = os.path.join(config.data_dir, "math_train.jsonl")
    val_path = os.path.join(config.data_dir, "math_val.jsonl")

    def load_jsonl(path: str) -> List[Dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    log.info(f"Loading training data from {train_path}")
    train_raw = load_jsonl(train_path)
    log.info(f"  Train: {len(train_raw)} examples")

    log.info(f"Loading validation data from {val_path}")
    val_raw = load_jsonl(val_path)
    log.info(f"  Val: {len(val_raw)} examples")

    train_dataset = Dataset.from_list(train_raw)
    val_dataset = Dataset.from_list(val_raw)

    return train_dataset, val_dataset


# ─────────────────────────────────────────────────────────────────────
# STEP 3: Load Model & Tokenizer (4-bit quantized + LoRA)
# ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(config: TrainingConfig):
    """Load the base model with 4-bit quantization and attach LoRA adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    log.info(f"Loading model: {config.base_model}")

    # ── Quantization Config ──
    quant_config = None
    if config.use_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_double_quant,
        )
        log.info(f"  4-bit NF4, compute={config.bnb_4bit_compute_dtype}, double_quant={config.use_double_quant}")

    # ── Tokenizer ──
    log.info("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    log.info(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # ── Model ──
    log.info("  Loading base model (this may take 2-5 min)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    # ── Prep for kbit training ──
    if config.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
        log.info("  Model prepared for 4-bit training")

    # ── LoRA ──
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"  LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_mem / 1e9
        log.info(f"  VRAM: {vram_used:.2f} / {vram_total:.2f} GB ({100*vram_used/vram_total:.1f}%)")

    return model, tokenizer, peft_config


# ─────────────────────────────────────────────────────────────────────
# STEP 4: Format Dataset for SFTTrainer
# ─────────────────────────────────────────────────────────────────────

def format_chat_to_text(example, tokenizer):
    """
    Convert ChatML messages to a single formatted string.

    DeepSeek-Math uses its own chat template. We attempt to use the
    tokenizer's built-in template; if unavailable, fall back to a
    generic Llama-style format.
    """
    messages = example["messages"]

    # Try using the model's native chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    except Exception:
        pass

    # Fallback: generic format
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"### System:\n{content}\n")
        elif role == "user":
            parts.append(f"### User:\n{content}\n")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}")
    text = "\n".join(parts) + tokenizer.eos_token
    return {"text": text}


def prepare_datasets(train_dataset, val_dataset, tokenizer, config: TrainingConfig):
    """Map the ChatML datasets into formatted text columns."""
    log.info("Formatting datasets for SFTTrainer...")

    train_formatted = train_dataset.map(
        lambda ex: format_chat_to_text(ex, tokenizer),
        remove_columns=train_dataset.column_names,
        desc="Formatting train",
    )
    val_formatted = val_dataset.map(
        lambda ex: format_chat_to_text(ex, tokenizer),
        remove_columns=val_dataset.column_names,
        desc="Formatting val",
    )

    # Quick stats
    train_lengths = [len(tokenizer.encode(t)) for t in train_formatted["text"][:20]]
    log.info(f"  Sample token lengths (first 20 train): "
             f"min={min(train_lengths)}, max={max(train_lengths)}, "
             f"avg={np.mean(train_lengths):.0f}")

    return train_formatted, val_formatted


# ─────────────────────────────────────────────────────────────────────
# STEP 5: Checkpoint & Resume Logic
# ─────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint directory for resuming training."""
    if not os.path.isdir(output_dir):
        return None

    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None

    # Sort by step number
    def get_step(path):
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return -1

    checkpoints = sorted(checkpoints, key=get_step)
    latest = checkpoints[-1]
    log.info(f"  Found checkpoint for resume: {latest}")
    return latest


# ─────────────────────────────────────────────────────────────────────
# STEP 6: Custom Callbacks
# ─────────────────────────────────────────────────────────────────────

def create_callbacks(config: TrainingConfig, total_steps: int):
    """Create training callbacks for milestones, early stopping, and logging."""
    from transformers import TrainerCallback, EarlyStoppingCallback

    callbacks = []

    # ── Early Stopping ──
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        )
    )

    # ── Milestone Checkpoint Callback ──
    class MilestoneCheckpointCallback(TrainerCallback):
        """Save named checkpoints at 50% and 75% of total training steps."""

        def __init__(self, milestones_pct, output_dir, total_steps):
            self.milestone_steps = {
                int(pct * total_steps): f"milestone_{int(pct*100)}pct"
                for pct in milestones_pct
            }
            self.output_dir = output_dir
            self.saved = set()

        def on_step_end(self, args, state, control, **kwargs):
            current_step = state.global_step
            if current_step in self.milestone_steps and current_step not in self.saved:
                name = self.milestone_steps[current_step]
                milestone_dir = os.path.join(self.output_dir, name)
                log.info(f"  Saving milestone checkpoint: {name} (step {current_step})")

                # The trainer will save on next opportunity; we just mark it
                control.should_save = True
                self.saved.add(current_step)

                # Also copy adapter to a named milestone directory after save
                self._schedule_copy(current_step, name)

            return control

        def _schedule_copy(self, step, name):
            """Remember to copy the checkpoint after it's saved."""
            self._pending_copy = (step, name)

        def on_save(self, args, state, control, **kwargs):
            if hasattr(self, "_pending_copy"):
                step, name = self._pending_copy
                src = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                dst = os.path.join(self.output_dir, name)
                if os.path.isdir(src) and not os.path.isdir(dst):
                    try:
                        shutil.copytree(src, dst)
                        log.info(f"    Milestone saved: {dst}")
                    except Exception as e:
                        log.warning(f"    Could not copy milestone: {e}")
                del self._pending_copy
            return control

    callbacks.append(
        MilestoneCheckpointCallback(
            config.milestone_pcts,
            config.output_dir,
            total_steps,
        )
    )

    # ── VRAM + Loss Logging Callback ──
    class TrainingLoggerCallback(TrainerCallback):
        """Log VRAM usage and collect per-step losses for the final report."""

        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            self.start_time = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.time()
            log.info("  Training started!")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.train_losses.append({
                    "step": state.global_step,
                    "loss": logs["loss"],
                    "learning_rate": logs.get("learning_rate", 0),
                    "epoch": logs.get("epoch", 0),
                })

            if logs and "eval_loss" in logs:
                self.eval_losses.append({
                    "step": state.global_step,
                    "eval_loss": logs["eval_loss"],
                    "epoch": logs.get("epoch", 0),
                })

            # Periodic VRAM logging
            if state.global_step % 50 == 0 and torch.cuda.is_available():
                vram_gb = torch.cuda.memory_allocated() / 1e9
                log.info(f"  [Step {state.global_step}] VRAM: {vram_gb:.2f} GB")

        def on_train_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self.start_time if self.start_time else 0
            log.info(f"  Training complete! Total time: {elapsed/60:.1f} min")

    logger_cb = TrainingLoggerCallback()
    callbacks.append(logger_cb)

    return callbacks, logger_cb


# ─────────────────────────────────────────────────────────────────────
# STEP 7: Sanity Check — Quick Forward Pass
# ─────────────────────────────────────────────────────────────────────

def sanity_check(model, tokenizer, config: TrainingConfig):
    """Run a quick forward pass to verify the model works before training."""
    log.info("Running sanity check...")

    test_prompt = (
        "### System:\n"
        "You are a mathematics lecture summarizer.\n\n"
        "### User:\n"
        "[TRANSCRIPT] The derivative of x squared is 2x.\n"
        "[VISUAL] Equation [EQUATION]\n\n"
        "### Assistant:\n"
    )

    inputs = tokenizer(
        test_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0)

    log.info(f"  Sanity check passed! Logits shape: {outputs.logits.shape}")

    # Quick generation test
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_text = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    log.info(f"  Sample generation: {gen_text[:150]}...")

    # Free memory
    del inputs, outputs, gen_ids
    torch.cuda.empty_cache()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────
# STEP 8: Train
# ─────────────────────────────────────────────────────────────────────

def train(model, tokenizer, train_dataset, val_dataset, config: TrainingConfig):
    """Run SFT training with all bells and whistles."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)

    # Calculate total training steps
    num_train_examples = len(train_dataset)
    steps_per_epoch = max(
        1,
        num_train_examples // (config.per_device_train_batch_size * config.gradient_accumulation_steps),
    )
    total_steps = steps_per_epoch * config.num_epochs
    log.info(f"  Training plan: {num_train_examples} examples, "
             f"{steps_per_epoch} steps/epoch, {total_steps} total steps")

    # ── Callbacks ──
    callbacks, logger_cb = create_callbacks(config, total_steps)

    # ── Training Arguments ──
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=config.group_by_length,
        dataloader_num_workers=config.dataloader_num_workers,
        report_to=config.report_to,
        seed=config.seed,
        remove_unused_columns=False,
    )

    # ── SFTTrainer ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        max_seq_length=config.max_seq_length,
        packing=False,  # No packing — maintain example boundaries
    )

    # ── Resume logic ──
    resume_checkpoint = None
    if config.resume_from_checkpoint:
        resume_checkpoint = find_latest_checkpoint(config.output_dir)
        if resume_checkpoint:
            log.info(f"  Resuming from checkpoint: {resume_checkpoint}")
        else:
            log.info("  No checkpoint found, starting fresh")

    # ── Train ──
    log.info("=" * 60)
    log.info(" STARTING TRAINING")
    log.info("=" * 60)

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ── Save final adapter ──
    log.info("Saving final adapter...")
    final_dir = os.path.join(config.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info(f"  Final adapter saved to: {final_dir}")

    return trainer, train_result, logger_cb


# ─────────────────────────────────────────────────────────────────────
# STEP 9: Save Training Log & Report
# ─────────────────────────────────────────────────────────────────────

def save_training_report(
    trainer,
    train_result,
    logger_cb,
    config: TrainingConfig,
):
    """Save a comprehensive training report as JSON."""
    report = {
        "model": config.base_model,
        "subject": "math",
        "adapter": "lora_math",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_targets": config.lora_target_modules,
            "learning_rate": config.learning_rate,
            "epochs": config.num_epochs,
            "batch_size": config.per_device_train_batch_size,
            "grad_accum": config.gradient_accumulation_steps,
            "effective_batch": config.per_device_train_batch_size * config.gradient_accumulation_steps,
            "max_seq_length": config.max_seq_length,
            "scheduler": config.lr_scheduler_type,
            "optimizer": config.optim,
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "fp16": config.fp16,
        },
        "results": {
            "train_loss": train_result.metrics.get("train_loss", None),
            "train_runtime_sec": train_result.metrics.get("train_runtime", None),
            "train_samples_per_sec": train_result.metrics.get("train_samples_per_second", None),
            "total_steps": train_result.metrics.get("train_steps", None),
        },
        "loss_history": {
            "train": logger_cb.train_losses,
            "eval": logger_cb.eval_losses,
        },
    }

    # Final eval
    log.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    report["results"]["final_eval_loss"] = eval_metrics.get("eval_loss", None)
    log.info(f"  Final eval loss: {eval_metrics.get('eval_loss', 'N/A')}")

    report_path = os.path.join(config.output_dir, "training_log.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"  Training report saved: {report_path}")

    return report


# ─────────────────────────────────────────────────────────────────────
# STEP 10: Post-Training Validation — Generate Sample Outputs
# ─────────────────────────────────────────────────────────────────────

def validate_adapter(model, tokenizer, config: TrainingConfig):
    """Generate a few sample outputs to verify the adapter works."""
    log.info("Validating adapter with sample generations...")

    test_prompts = [
        {
            "transcript": (
                "The derivative of x squared is 2x. This can be derived using "
                "the power rule, which states that d/dx of x^n equals n times "
                "x to the power n minus 1."
            ),
            "visual": "Equation",
        },
        {
            "transcript": (
                "To find the probability of getting exactly 3 heads in 5 coin "
                "flips, we use the binomial distribution formula. C(5,3) times "
                "p cubed times (1-p) squared."
            ),
            "visual": "Equation",
        },
        {
            "transcript": (
                "The integral of 1/x dx is the natural logarithm of the "
                "absolute value of x plus the constant of integration C. "
                "This is a fundamental integral formula."
            ),
            "visual": "Equation",
        },
    ]

    results = []
    for i, prompt in enumerate(test_prompts):
        messages = [
            {"role": "system", "content": "You are a mathematics lecture summarizer. Summarize the lecture segment preserving all equations, mathematical notation, and step-by-step reasoning. Use LaTeX notation for equations where appropriate."},
            {"role": "user", "content": f"[TRANSCRIPT] {prompt['transcript']}\n[VISUAL] {prompt['visual']} [EQUATION]"},
        ]

        # Format as text
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = (
                f"### System:\n{messages[0]['content']}\n\n"
                f"### User:\n{messages[1]['content']}\n\n"
                f"### Assistant:\n"
            )

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        output = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append(output.strip())
        log.info(f"  Sample {i+1}: {output.strip()[:200]}...")

        del inputs, gen_ids
        torch.cuda.empty_cache()

    # Save samples
    samples_path = os.path.join(config.output_dir, "sample_outputs.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"input": p["transcript"], "output": r} for p, r in zip(test_prompts, results)],
            f, indent=2, ensure_ascii=False,
        )
    log.info(f"  Sample outputs saved: {samples_path}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info(" DAY 2 — Math LoRA SFT (DeepSeek-Math-7B)")
    log.info("=" * 60)

    config = TrainingConfig()

    # Step 0: Install deps
    install_dependencies()

    # Step 1: GPU check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        log.info(f"  GPU: {gpu_name} ({vram_gb:.1f} GB)")
    else:
        log.error("  No GPU detected! Training will be extremely slow.")
        log.error("  Switch runtime to GPU: Runtime → Change runtime type → T4 GPU")
        sys.exit(1)

    # Step 2: Load data
    log.info("\n--- STEP 2: Loading SFT Data ---")
    train_dataset, val_dataset = load_sft_data(config)

    # Step 3: Load model
    log.info("\n--- STEP 3: Loading Model & LoRA ---")
    model, tokenizer, peft_config = load_model_and_tokenizer(config)

    # Step 4: Format datasets
    log.info("\n--- STEP 4: Formatting Datasets ---")
    train_formatted, val_formatted = prepare_datasets(train_dataset, val_dataset, tokenizer, config)

    # Step 5: Sanity check
    log.info("\n--- STEP 5: Sanity Check ---")
    sanity_check(model, tokenizer, config)

    # Step 6: Train
    log.info("\n--- STEP 6: Training ---")
    trainer, train_result, logger_cb = train(
        model, tokenizer, train_formatted, val_formatted, config
    )

    # Step 7: Save report
    log.info("\n--- STEP 7: Saving Report ---")
    report = save_training_report(trainer, train_result, logger_cb, config)

    # Step 8: Validate
    log.info("\n--- STEP 8: Post-Training Validation ---")
    validate_adapter(model, tokenizer, config)

    # Summary
    log.info("\n" + "=" * 60)
    log.info(" DAY 2 COMPLETE!")
    log.info("=" * 60)
    log.info(f"  Adapter saved: {config.output_dir}/final/")
    log.info(f"  Training log:  {config.output_dir}/training_log.json")
    log.info(f"  Final loss:    {report['results'].get('train_loss', 'N/A')}")
    log.info(f"  Eval loss:     {report['results'].get('final_eval_loss', 'N/A')}")
    log.info("=" * 60)
    log.info("NEXT STEPS:")
    log.info("  1. Check training_log.json for loss curves")
    log.info("  2. Check sample_outputs.json for generation quality")
    log.info("  3. Download lora_math/final/ adapter for Day 4 evaluation")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
