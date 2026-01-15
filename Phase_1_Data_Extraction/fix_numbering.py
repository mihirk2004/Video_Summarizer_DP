import os
import re

BASE_DIR = "data/processed"

OLD_START = 202
OLD_END = 248
NEW_START = 192

def pad(n):
    return f"{n:03d}"

# Step 1: collect folders
lecture_nums = []
for name in os.listdir(BASE_DIR):
    match = re.match(r"lecture_(\d+)$", name)
    if match:
        num = int(match.group(1))
        if OLD_START <= num <= OLD_END:
            lecture_nums.append(num)

# Sort descending (still important)
lecture_nums.sort(reverse=True)

# -------------------------
# PHASE 1: rename to temp
# -------------------------
for old_num in lecture_nums:
    old_folder = os.path.join(BASE_DIR, f"lecture_{pad(old_num)}")
    temp_folder = os.path.join(BASE_DIR, f"__tmp_{pad(old_num)}")

    print(f"[TEMP] lecture_{pad(old_num)} → __tmp_{pad(old_num)}")
    os.rename(old_folder, temp_folder)

# -------------------------
# PHASE 2: rename to final
# -------------------------
for old_num in lecture_nums:
    new_num = NEW_START + (old_num - OLD_START)

    temp_folder = os.path.join(BASE_DIR, f"__tmp_{pad(old_num)}")
    new_folder = os.path.join(BASE_DIR, f"lecture_{pad(new_num)}")

    print(f"[FINAL] __tmp_{pad(old_num)} → lecture_{pad(new_num)}")

    # Rename JSON inside folder
    for file in os.listdir(temp_folder):
        if re.match(rf"lecture_{pad(old_num)}.*\.json", file):
            old_json = os.path.join(temp_folder, file)
            new_json = os.path.join(
                temp_folder,
                file.replace(pad(old_num), pad(new_num))
            )
            print(f"   JSON: {file} → {os.path.basename(new_json)}")
            os.rename(old_json, new_json)

    os.rename(temp_folder, new_folder)

print("✅ Renumbering completed safely.")
