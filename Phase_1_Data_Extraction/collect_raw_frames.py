import os
import shutil
import re

PROCESSED_DIR = "data/processed"
RAW_FRAMES_DIR = "data/raw_frames"

os.makedirs(RAW_FRAMES_DIR, exist_ok=True)

for lecture_folder in os.listdir(PROCESSED_DIR):
    match = re.fullmatch(r"lecture_\d+", lecture_folder)
    if not match:
        continue

    frames_src = os.path.join(PROCESSED_DIR, lecture_folder, "frames")
    if not os.path.isdir(frames_src):
        print(f"Skipping (no frames): {lecture_folder}")
        continue

    lecture_dst = os.path.join(RAW_FRAMES_DIR, lecture_folder)
    os.makedirs(lecture_dst, exist_ok=True)

    print(f"Copying frames from {lecture_folder}/frames → raw_frames/{lecture_folder}")

    for file in os.listdir(frames_src):
        src_file = os.path.join(frames_src, file)
        dst_file = os.path.join(lecture_dst, file)

        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

print("✅ All frames copied to data/raw_frames successfully.")
