import json, os, glob

annot_dir = r'd:\Users\Mihir\Downloads\Documents\Mihir Codes\Dp_Project\data\annotations'
subjects = {}

for f in sorted(glob.glob(os.path.join(annot_dir, 'lecture_*_annotated.json'))):
    lid = os.path.basename(f).replace('_annotated.json', '')
    num = int(lid.split('_')[1])
    if num > 135:
        continue
    try:
        d = json.load(open(f, encoding='utf-8'))
        subj = d.get('metadata', {}).get('subject', 'Unknown')
        subjects.setdefault(subj, []).append(lid)
    except:
        subjects.setdefault('Error', []).append(lid)

for subj, lecs in sorted(subjects.items()):
    print(f"{subj}: {len(lecs)} lectures")
    print(f"  Sample: {lecs[:5]} ... {lecs[-3:]}")
