"""
=============================================================================
Day 2 — Colab Setup & Runner (Math LoRA SFT)
=============================================================================
Run this in Google Colab to train the Math LoRA adapter
on DeepSeek-Math-7B-Instruct.

Copy each cell block into a separate Colab cell and run in order.

Account #1 — Same account used for Day 1 data preparation.
=============================================================================
"""

# ═══════════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════════
# Run this cell FIRST. You may need to restart runtime after install.
#
# !pip install -q transformers>=4.40.0 peft>=0.10.0 trl>=0.8.0 \
#     bitsandbytes>=0.43.0 accelerate>=0.28.0 datasets>=2.18.0 scipy
#
# # Verify GPU
# !nvidia-smi

# ═══════════════════════════════════════════════════════════════════
# CELL 2: Mount Google Drive
# ═══════════════════════════════════════════════════════════════════
#
# from google.colab import drive
# drive.mount('/content/drive')
#
# # Verify SFT data exists (produced by Day 1)
# !ls -la /content/drive/MyDrive/Colab_text_modelling_dp_01/data/processed/sft_data/math_*.jsonl
# !wc -l /content/drive/MyDrive/Colab_text_modelling_dp_01/data/processed/sft_data/math_*.jsonl
#
# # Create output directory
# !mkdir -p /content/drive/MyDrive/Colab_text_modelling_dp_01/lora_math

# ═══════════════════════════════════════════════════════════════════
# CELL 3: Upload & Run Training Script
# ═══════════════════════════════════════════════════════════════════
#
# # OPTION A: Upload day2_math_sft.py from sidebar file upload, then:
# !python day2_math_sft.py
#
# # OPTION B: If script is on Drive:
# !python /content/drive/MyDrive/Colab_text_modelling_dp_01/scripts/finetuning/day2_math_sft.py
#
# # OPTION C: If you want to resume from a checkpoint (after disconnect):
# # Just re-run the same command — the script auto-detects checkpoints!
# !python day2_math_sft.py

# ═══════════════════════════════════════════════════════════════════
# CELL 4: Inspect Results (run after training completes)
# ═══════════════════════════════════════════════════════════════════

import json
import os

def inspect_math_training():
    """Inspect the math LoRA training results."""
    base_dir = "/content/drive/MyDrive/Colab_text_modelling_dp_01/lora_math"

    # 1. Check training log
    log_path = os.path.join(base_dir, "training_log.json")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            report = json.load(f)
        print("=" * 50)
        print("TRAINING REPORT")
        print("=" * 50)
        print(f"Model:        {report['model']}")
        print(f"Train loss:   {report['results'].get('train_loss', 'N/A')}")
        print(f"Eval loss:    {report['results'].get('final_eval_loss', 'N/A')}")
        print(f"Runtime:      {report['results'].get('train_runtime_sec', 0)/60:.1f} min")
        print(f"Total steps:  {report['results'].get('total_steps', 'N/A')}")

        # Loss curve summary
        train_losses = report.get("loss_history", {}).get("train", [])
        eval_losses = report.get("loss_history", {}).get("eval", [])
        if train_losses:
            first = train_losses[0]["loss"]
            last = train_losses[-1]["loss"]
            print(f"\nLoss curve:   {first:.4f} -> {last:.4f} (Δ={last-first:.4f})")
        if eval_losses:
            best_eval = min(e["eval_loss"] for e in eval_losses)
            print(f"Best eval:    {best_eval:.4f}")
    else:
        print("No training log found yet.")

    # 2. Check sample outputs
    samples_path = os.path.join(base_dir, "sample_outputs.json")
    if os.path.exists(samples_path):
        with open(samples_path, "r") as f:
            samples = json.load(f)
        print("\n" + "=" * 50)
        print("SAMPLE OUTPUTS")
        print("=" * 50)
        for i, s in enumerate(samples):
            print(f"\n--- Sample {i+1} ---")
            print(f"Input:  {s['input'][:100]}...")
            print(f"Output: {s['output'][:200]}...")

    # 3. Check adapter files
    final_dir = os.path.join(base_dir, "final")
    if os.path.isdir(final_dir):
        files = os.listdir(final_dir)
        print(f"\nAdapter files in {final_dir}:")
        for f in sorted(files):
            fpath = os.path.join(final_dir, f)
            size_mb = os.path.getsize(fpath) / 1e6 if os.path.isfile(fpath) else 0
            print(f"  {f} ({size_mb:.1f} MB)")

    # 4. Check milestones
    for pct in [50, 75]:
        milestone_dir = os.path.join(base_dir, f"milestone_{pct}pct")
        if os.path.isdir(milestone_dir):
            print(f"\n  Milestone {pct}% checkpoint exists")

# Uncomment to run:
# inspect_math_training()

# ═══════════════════════════════════════════════════════════════════
# CELL 5: Plot Loss Curves (optional)
# ═══════════════════════════════════════════════════════════════════
#
# import matplotlib.pyplot as plt
#
# def plot_loss_curves():
#     log_path = "/content/drive/MyDrive/Colab_text_modelling_dp_01/lora_math/training_log.json"
#     with open(log_path, "r") as f:
#         report = json.load(f)
#
#     train_losses = report.get("loss_history", {}).get("train", [])
#     eval_losses = report.get("loss_history", {}).get("eval", [])
#
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#
#     if train_losses:
#         steps = [l["step"] for l in train_losses]
#         losses = [l["loss"] for l in train_losses]
#         ax.plot(steps, losses, label="Train Loss", alpha=0.7)
#
#     if eval_losses:
#         steps = [l["step"] for l in eval_losses]
#         losses = [l["eval_loss"] for l in eval_losses]
#         ax.plot(steps, losses, label="Eval Loss", marker="o", markersize=4)
#
#     ax.set_xlabel("Step")
#     ax.set_ylabel("Loss")
#     ax.set_title("Math LoRA Training Loss Curve")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig("/content/drive/MyDrive/Colab_text_modelling_dp_01/lora_math/loss_curve.png", dpi=150)
#     plt.show()
#
# # plot_loss_curves()

# ═══════════════════════════════════════════════════════════════════
# CELL 6: Download Adapter for Day 4
# ═══════════════════════════════════════════════════════════════════
#
# # Download the final adapter to upload to Account #3 for evaluation
# import shutil
#
# # Zip the adapter
# shutil.make_archive(
#     "/content/lora_math_final",
#     "zip",
#     "/content/drive/MyDrive/Colab_text_modelling_dp_01/lora_math/final"
# )
#
# from google.colab import files
# files.download("/content/lora_math_final.zip")
#
# print("Download lora_math_final.zip and upload to Account #3 for Day 4")
