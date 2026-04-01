import os
import json
import re

ANNOTATION_DIR = "data/annotations"

# --------------------------------
# SUBJECT RULES
# --------------------------------
SUBJECT_RULES = [
    (1, 71, "Computer Science"),
    (72, 135, "Maths"),
    (136, 191, "Physics"),
    (192, 198, "Biology"),
    (199, 238, "Chemistry"),
]

def get_subject(lecture_num):
    for start, end, subject in SUBJECT_RULES:
        if start <= lecture_num <= end:
            return subject
    return None

def pad(n):
    return f"{n:03d}"

# ==================================================
# STEP 1: RENAME ALL lecture_XXX.json → lecture_XXX_annotated.json
# ==================================================
for filename in os.listdir(ANNOTATION_DIR):
    # Match lecture_XXX.json but NOT lecture_XXX_annotated.json
    match = re.fullmatch(r"lecture_(\d+)\.json", filename)
    if not match:
        continue

    lecture_num = match.group(1)
    old_path = os.path.join(ANNOTATION_DIR, filename)
    new_name = f"lecture_{lecture_num}_annotated.json"
    new_path = os.path.join(ANNOTATION_DIR, new_name)

    # Avoid overwriting if annotated already exists
    if os.path.exists(new_path):
        print(f"Skipping rename (already exists): {new_name}")
        continue

    print(f"[RENAME] {filename} → {new_name}")
    os.rename(old_path, new_path)

# ==================================================
# STEP 2: UPDATE SUBJECT METADATA
# ==================================================
for filename in os.listdir(ANNOTATION_DIR):
    if not filename.endswith(".json"):
        continue

    match = re.search(r"lecture_(\d+)", filename)
    if not match:
        print(f"Skipping (no lecture number found): {filename}")
        continue

    lecture_num = int(match.group(1))
    subject = get_subject(lecture_num)

    if subject is None:
        print(f"No subject rule for lecture {lecture_num} ({filename})")
        continue

    file_path = os.path.join(ANNOTATION_DIR, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "metadata" not in data:
        print(f"Skipping (no metadata): {filename}")
        continue

    old_subject = data["metadata"].get("subject", "N/A")
    data["metadata"]["subject"] = subject

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[SUBJECT] {filename}: {old_subject} → {subject}")

print("✅ Renaming + subject update completed successfully.")
