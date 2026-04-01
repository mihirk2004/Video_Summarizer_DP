import os
import shutil
import glob

source_root = "data/processed"
target_dir = "data/annotations"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Find all *_annotated.json files inside lecture folders
json_files = glob.glob(os.path.join(source_root, "*", "*_annotated.json"))

for json_file in json_files:
    shutil.copy(json_file, target_dir)
    print(f"Copied: {json_file}")

print("✅ All annotation files copied successfully.")
