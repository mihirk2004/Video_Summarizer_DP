"""
Local dry-run test for Day 1 data preparation.
Tests all logic except Gemini API calls.
"""
import sys
sys.path.insert(0, r"d:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\scripts\finetuning")

# Monkey-patch Config for local paths
import day1_data_preparation as d1
d1.Config.INFERENCE_JSON = r"d:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\results\multimodal_inference.json"
d1.Config.ANNOTATIONS_DIR = r"d:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\data\annotations"
d1.Config.OUTPUT_DIR = r"d:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\data\sft_data"
d1.Config.GEMINI_API_KEY = ""  # Empty = skip augmentation (dry run)
d1.Config.MATH_SAMPLE_SIZE = 30  # Small sample for quick test
d1.Config.CS_SAMPLE_SIZE = 30

# Run
d1.main()
