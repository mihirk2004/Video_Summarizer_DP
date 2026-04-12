import json

data = json.load(open(r'd:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\results\multimodal_inference.json', encoding='utf-8'))
segs = data['segments']

# Load subject mapping
import os, glob
annot_dir = r'd:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\data\annotations'
lec_subjects = {}
for f in glob.glob(os.path.join(annot_dir, 'lecture_*_annotated.json')):
    lid = os.path.basename(f).replace('_annotated.json', '')
    try:
        d = json.load(open(f, encoding='utf-8'))
        lec_subjects[lid] = d.get('metadata', {}).get('subject', 'Unknown')
    except:
        pass

# Filter quality >= 0.4
good = [s for s in segs if s.get('quality_score', 0) >= 0.4]
math_segs = [s for s in good if lec_subjects.get(s['lecture_id'], '') == 'Maths']
cs_segs = [s for s in good if lec_subjects.get(s['lecture_id'], '') == 'Computer Science']

print(f"Total quality >= 0.4: {len(good)}")
print(f"Math: {len(math_segs)}, CS: {len(cs_segs)}")

# Augmentation split
math_high = [s for s in math_segs if s['quality_score'] >= 0.85]
math_low = [s for s in math_segs if s['quality_score'] < 0.85]
cs_high = [s for s in cs_segs if s['quality_score'] >= 0.85]
cs_low = [s for s in cs_segs if s['quality_score'] < 0.85]

print(f"\n--- MATH ---")
print(f"  quality >= 0.85 (skip augment, use target_summary): {len(math_high)}")
print(f"  quality < 0.85 (needs augmentation): {len(math_low)}")
print(f"  With Equation tag (high): {sum(1 for s in math_high if 'Equation' in s.get('visual_tags',[]))}")
print(f"  With Equation tag (low): {sum(1 for s in math_low if 'Equation' in s.get('visual_tags',[]))}")

print(f"\n--- CS ---")
print(f"  quality >= 0.85 (skip augment, use target_summary): {len(cs_high)}")
print(f"  quality < 0.85 (needs augmentation): {len(cs_low)}")
print(f"  With Code tag (high): {sum(1 for s in cs_high if 'Computer_Code' in s.get('visual_tags',[]))}")
print(f"  With Code tag (low): {sum(1 for s in cs_low if 'Computer_Code' in s.get('visual_tags',[]))}")

# Quality distribution buckets
import collections
buckets = collections.Counter()
for s in good:
    subj = lec_subjects.get(s['lecture_id'], 'Other')
    if subj == 'Maths':
        bucket = 'Math'
    elif subj == 'Computer Science':
        bucket = 'CS'
    else:
        bucket = 'Other'
    qs = s['quality_score']
    if qs >= 0.85:
        buckets[f"{bucket}_high"] += 1
    elif qs >= 0.7:
        buckets[f"{bucket}_mid"] += 1
    elif qs >= 0.5:
        buckets[f"{bucket}_low"] += 1
    else:
        buckets[f"{bucket}_vlow"] += 1

print(f"\n--- Quality Buckets ---")
for k in sorted(buckets.keys()):
    print(f"  {k}: {buckets[k]}")

# Word count stats for proper max_length
import numpy as np
math_wc = [s['word_count'] for s in math_segs]
cs_wc = [s['word_count'] for s in cs_segs]
print(f"\n--- Word Counts ---")
print(f"  Math: mean={sum(math_wc)/len(math_wc):.0f}, min={min(math_wc)}, max={max(math_wc)}")
print(f"  CS: mean={sum(cs_wc)/len(cs_wc):.0f}, min={min(cs_wc)}, max={max(cs_wc)}")

# Target summary length
math_ts = [len(s.get('target_summary','').split()) for s in math_segs]
cs_ts = [len(s.get('target_summary','').split()) for s in cs_segs]
print(f"  Math target_summary: mean={sum(math_ts)/len(math_ts):.0f}, min={min(math_ts)}, max={max(math_ts)}")
print(f"  CS target_summary: mean={sum(cs_ts)/len(cs_ts):.0f}, min={min(cs_ts)}, max={max(cs_ts)}")
