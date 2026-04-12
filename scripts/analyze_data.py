import json

data = json.load(open(r'd:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\data\processed\multimodal_dataset\multimodal_segments.json', encoding='utf-8'))
segs = data['segments']

eq_segs = [s for s in segs if 'Equation' in s.get('visual_tags', [])]
code_segs = [s for s in segs if 'Computer_Code' in s.get('visual_tags', [])]
graph_segs = [s for s in segs if 'Graph_Chart' in s.get('visual_tags', [])]

print(f"Segments with Equation tag: {len(eq_segs)}")
print(f"Segments with Computer_Code tag: {len(code_segs)}")
print(f"Segments with Graph_Chart tag: {len(graph_segs)}")

# Check quality_score
has_qs = sum(1 for s in segs if 'quality_score' in s)
print(f"Segments with quality_score field: {has_qs}")

# Sample equation segment
if eq_segs:
    s = eq_segs[0]
    print(f"\nSample Equation segment: {s['segment_id']}")
    print(f"  visual_tags: {s['visual_tags']}")
    print(f"  word_count: {s['word_count']}")
    print(f"  has target_summary: {'target_summary' in s}")
    print(f"  target_summary[:200]: {s.get('target_summary','')[:200]}")

# Segments in lecture 001-135
in_range = [s for s in segs if int(s['lecture_id'].split('_')[1]) <= 135]
print(f"\nSegments from lecture_001 to lecture_135: {len(in_range)}")

# Count by keyword heuristics for math
import re
math_kw = re.compile(r'equation|formula|integral|derivative|matrix|theorem|proof|mathematical|calculus|algebra|polynomial|logarithm|exponential|factorial|probability|permutation|combination', re.I)
code_kw = re.compile(r'function|variable|class|object|loop|array|pointer|struct|algorithm|program|code|compiler|syntax|operator|inheritance|polymorphism', re.I)

math_by_text = [s for s in segs if math_kw.search(s.get('raw_text', '') + ' '.join(s.get('visual_tags', [])))]
cs_by_text = [s for s in segs if code_kw.search(s.get('raw_text', '') + ' '.join(s.get('visual_tags', [])))]

print(f"\nMath segments (keyword match): {len(math_by_text)}")
print(f"CS segments (keyword match): {len(cs_by_text)}")

# Both
both = [s for s in segs if s in math_by_text and s in cs_by_text]
print(f"Both (overlap): {len(both)}")

# Equation tag lectures
eq_lecs = sorted(set(s['lecture_id'] for s in eq_segs))
print(f"\nLectures with Equation frames: {len(eq_lecs)}")
print(f"  Sample: {eq_lecs[:10]}")

code_lecs = sorted(set(s['lecture_id'] for s in code_segs))
print(f"Lectures with Code frames: {len(code_lecs)}")
print(f"  Sample: {code_lecs[:10]}")

# Fields available in a segment
print(f"\nFields in segment: {list(segs[0].keys())}")
