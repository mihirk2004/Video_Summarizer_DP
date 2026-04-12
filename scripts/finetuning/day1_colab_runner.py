"""
=============================================================================
Day 1 — Colab Setup & Runner
=============================================================================
Run this FIRST in Google Colab to set up the environment and execute data prep.

Steps:
  Cell 1: Install dependencies
  Cell 2: Mount Drive & upload data
  Cell 3: Configure paths
  Cell 4: Run data preparation
  Cell 5: Inspect outputs
=============================================================================
"""

# ═══════════════════════════════════════════════════════════════════
# CELL 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════════
# Run this cell first. Restart runtime if prompted.
#
# !pip install -q google-generativeai

# ═══════════════════════════════════════════════════════════════════
# CELL 2: Mount Google Drive & Upload Data
# ═══════════════════════════════════════════════════════════════════
#
# from google.colab import drive
# drive.mount('/content/drive')
#
# # Create project directory on Drive
# !mkdir -p /content/drive/MyDrive/dp_project/results
# !mkdir -p /content/drive/MyDrive/dp_project/data/annotations
# !mkdir -p /content/drive/MyDrive/sft_data
#
# # OPTION A: Upload from local machine via Colab sidebar
# # Click the folder icon → Upload → select multimodal_inference.json
# # Then copy it to Drive:
# # !cp /content/multimodal_inference.json /content/drive/MyDrive/dp_project/results/
#
# # OPTION B: If files are already on Drive, verify they exist:
# !ls -la /content/drive/MyDrive/dp_project/results/multimodal_inference.json
# !ls /content/drive/MyDrive/dp_project/data/annotations/ | head -5

# ═══════════════════════════════════════════════════════════════════
# CELL 3: Configure & Set API Key
# ═══════════════════════════════════════════════════════════════════

import os

# Set your Gemini API key
os.environ["GEMINI_API_KEY"] = ""  # ← PUT YOUR KEY HERE

# Verify Drive mount
assert os.path.exists("/content/drive/MyDrive"), "Mount Google Drive first!"

# ═══════════════════════════════════════════════════════════════════
# CELL 4: Upload & Run the Data Preparation Script
# ═══════════════════════════════════════════════════════════════════
#
# # Upload day1_data_preparation.py to Colab, then:
# !python day1_data_preparation.py
#
# # Or if it's on Drive:
# !python /content/drive/MyDrive/dp_project/scripts/finetuning/day1_data_preparation.py

# ═══════════════════════════════════════════════════════════════════
# CELL 5: Inspect Outputs
# ═══════════════════════════════════════════════════════════════════

def inspect_outputs():
    """Quick inspection of generated JSONL files."""
    import json
    sft_dir = "/content/drive/MyDrive/Colab_text_modelling_dp_01/data/processed/sft_data"

    for fname in ["math_train.jsonl", "math_val.jsonl", "cs_train.jsonl", "cs_val.jsonl"]:
        fpath = os.path.join(sft_dir, fname)
        if not os.path.exists(fpath):
            print(f"❌ Missing: {fname}")
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"\n{'='*50}")
        print(f"📄 {fname}: {len(lines)} examples")
        print(f"{'='*50}")

        if lines:
            # Show first example
            item = json.loads(lines[0])
            msgs = item["messages"]
            print(f"System: {msgs[0]['content'][:80]}...")
            print(f"User:   {msgs[1]['content'][:100]}...")
            print(f"Asst:   {msgs[2]['content'][:150]}...")

            meta = item.get("_metadata", {})
            print(f"Meta:   segment={meta.get('segment_id')}, "
                  f"quality={meta.get('quality_score', 0):.3f}, "
                  f"method={meta.get('augmentation_method')}")

    # Show report
    report_path = os.path.join(sft_dir, "data_prep_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
        print(f"\n{'='*50}")
        print("📊 DATA PREP REPORT")
        print(f"{'='*50}")
        print(json.dumps(report, indent=2))

# Uncomment to run:
# inspect_outputs()

# ═══════════════════════════════════════════════════════════════════
# CELL 6: Download files for other accounts
# ═══════════════════════════════════════════════════════════════════
#
# # Download JSONL files to upload to Account #2 (CS training)
# from google.colab import files
# files.download('/content/drive/MyDrive/sft_data/cs_train.jsonl')
# files.download('/content/drive/MyDrive/sft_data/cs_val.jsonl')
#
# # Math files stay on this account's Drive for Day 2
# print("✅ Math files ready on this Drive for Day 2")
# print("📥 Download cs_*.jsonl and upload to Account #2 for Day 3")
