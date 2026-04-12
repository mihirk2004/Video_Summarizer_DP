import json

data = json.load(open(r'd:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\results\multimodal_inference.json', encoding='utf-8'))
segs = data['segments']

print(f"Total segments in multimodal_inference.json: {len(segs)}")

# Fields
print(f"Fields: {list(segs[0].keys())}")

# Quality score stats
qs = [s['quality_score'] for s in segs if 'quality_score' in s]
print(f"\nQuality score stats:")
print(f"  Count: {len(qs)}")
print(f"  Min: {min(qs):.4f}")
print(f"  Max: {max(qs):.4f}")
print(f"  Mean: {sum(qs)/len(qs):.4f}")
above_04 = sum(1 for q in qs if q >= 0.4)
print(f"  >= 0.4: {above_04}")

# Tag distribution
import re
eq_segs = [s for s in segs if 'Equation' in s.get('visual_tags', [])]
code_segs = [s for s in segs if 'Computer_Code' in s.get('visual_tags', [])]
graph_segs = [s for s in segs if 'Graph_Chart' in s.get('visual_tags', [])]
print(f"\nEquation segments: {len(eq_segs)}")
print(f"Computer_Code segments: {len(code_segs)}")
print(f"Graph_Chart segments: {len(graph_segs)}")

# Check subject from annotations
annot_dir = r'd:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\data\annotations'
import os, glob
lec_subjects = {}
for f in glob.glob(os.path.join(annot_dir, 'lecture_*_annotated.json')):
    lid = os.path.basename(f).replace('_annotated.json', '')
    try:
        d = json.load(open(f, encoding='utf-8'))
        lec_subjects[lid] = d.get('metadata', {}).get('subject', 'Unknown')
    except:
        pass

# Cross-reference
math_segs = [s for s in segs if lec_subjects.get(s['lecture_id'], '') == 'Maths']
cs_segs = [s for s in segs if lec_subjects.get(s['lecture_id'], '') == 'Computer Science']
print(f"\nMaths segments (by annotation subject): {len(math_segs)}")
print(f"CS segments (by annotation subject): {len(cs_segs)}")

# Math + quality >= 0.4
math_good = [s for s in math_segs if s.get('quality_score', 0) >= 0.4]
cs_good = [s for s in cs_segs if s.get('quality_score', 0) >= 0.4]
print(f"Math segments with quality >= 0.4: {len(math_good)}")
print(f"CS segments with quality >= 0.4: {len(cs_good)}")

# Sample a math Equation segment
math_eq = [s for s in math_segs if 'Equation' in s.get('visual_tags', [])]
if math_eq:
    s = math_eq[0]
    print(f"\nSample Math+Equation segment: {s['segment_id']}")
    print(f"  final_summary[:200]: {s.get('final_summary','')[:200]}")
    print(f"  target_summary[:200]: {s.get('target_summary','')[:200]}")
    print(f"  quality_score: {s.get('quality_score')}")
