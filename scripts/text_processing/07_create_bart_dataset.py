#!/usr/bin/env python3
"""
Step 3: Create PyTorch Dataset for BART fine-tuning
Loads dataset_with_summaries.json → tokenizes → train/val/test split → saves

Usage:
    python scripts/text_processing/07_create_bart_dataset.py          # Full run
    python scripts/text_processing/07_create_bart_dataset.py --test   # Quick test
"""
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


# ======================================================================
# Dataset class
# ======================================================================
class LectureSummarizationDataset(Dataset):
    """PyTorch Dataset for lecture summarization"""

    def __init__(self, samples: List[Dict], tokenizer, max_input: int, max_target: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Tokenize input
        inputs = self.tokenizer(
            item['formatted_input'],
            max_length=self.max_input,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        # Tokenize target (summary)
        targets = self.tokenizer(
            item['pseudo_summary'],
            max_length=self.max_target,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        labels = targets['input_ids'].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels,
        }

        # Entity mask (for entity-aware loss)
        if 'entity_mask' in item:
            mask = item['entity_mask']
            # Pad or truncate to match target length
            if len(mask) < self.max_target:
                mask = mask + [0] * (self.max_target - len(mask))
            else:
                mask = mask[:self.max_target]
            result['entity_mask'] = torch.tensor(mask, dtype=torch.float)

        return result


# ======================================================================
# Dataset builder
# ======================================================================
class BARTDatasetBuilder:
    """Build tokenized datasets with train/val/test splits"""

    def __init__(self, config: Dict, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.logger = setup_logger("bart_dataset")

        # Paths
        self.data_dir = Path(config['paths']['data'].get(
            'bart_dataset', 'data/processed/bart_dataset'))

        # BART config
        bart = config.get('bart', {})
        self.model_name = bart.get('model_name', 'facebook/bart-large-cnn')
        self.max_input = bart.get('max_input_length', 1024)
        self.max_target = bart.get('max_target_length', 256)
        self.entity_loss_weight = bart.get('entity_loss_weight', 1.5)
        self.special_tokens = bart.get('special_tokens', [
            '[TRANSCRIPT]', '[VISUAL]', '[EQUATION]', '[DIAGRAM]',
            '[CODE]', '[SLIDE]', '[GRAPH]',
        ])

        # Split ratios
        self.train_ratio = config.get('data', {}).get('train_ratio', 0.7)
        self.val_ratio = config.get('data', {}).get('val_ratio', 0.15)
        self.test_ratio = config.get('data', {}).get('test_ratio', 0.15)
        self.seed = config.get('project', {}).get('random_seed', 42)

        # Tokenizer (loaded lazily)
        self._tokenizer = None
        self._nlp = None

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer

        self.logger.info(f"Loading tokenizer: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Add special tokens
        added = self._tokenizer.add_special_tokens({
            'additional_special_tokens': self.special_tokens
        })
        self.logger.info(f"Added {added} special tokens to tokenizer")

        # Save tokenizer for later use
        tok_path = self.data_dir / "tokenizer"
        self._tokenizer.save_pretrained(str(tok_path))
        self.logger.info(f"Tokenizer saved → {tok_path}")

    def _load_nlp(self):
        if self._nlp is not None:
            return
        import spacy

        # Priority: trained STEM NER > scispaCy > spaCy core
        ner_path = Path(self.config.get('paths', {}).get('models', {}).get(
            'ner', 'models/text/ner/final'))
        if ner_path.exists():
            try:
                self._nlp = spacy.load(str(ner_path))
                self.logger.info(f"Loaded trained STEM NER from {ner_path} ✓")
                return
            except Exception as e:
                self.logger.warning(f"Could not load trained NER ({e}), falling back")

        try:
            self._nlp = spacy.load("en_core_sci_sm")
            self.logger.info("Using en_core_sci_sm for entity detection")
        except OSError:
            self._nlp = spacy.load("en_core_web_sm")
            self.logger.info("Using en_core_web_sm for entity detection")

    # ------------------------------------------------------------------
    # Entity mask creation
    # ------------------------------------------------------------------
    def _create_entity_mask(self, summary: str) -> List[int]:
        """Create binary mask marking token positions that correspond to entities"""
        if self.entity_loss_weight <= 1.0:
            return []  # No mask needed

        self._load_nlp()

        # Tokenize summary to get offset mappings
        encoding = self._tokenizer(
            summary,
            max_length=self.max_target,
            truncation=True,
            return_offsets_mapping=True,
        )
        offsets = encoding['offset_mapping']

        # Find entity spans using NER
        doc = self._nlp(summary)
        entity_spans = []
        for ent in doc.ents:
            entity_spans.append((ent.start_char, ent.end_char))
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                entity_spans.append((chunk.start_char, chunk.end_char))

        # Map character spans to token positions
        mask = [0] * len(offsets)
        for char_start, char_end in entity_spans:
            for tok_idx, (tok_s, tok_e) in enumerate(offsets):
                if tok_s == tok_e == 0:
                    continue  # Special token
                if tok_s < char_end and tok_e > char_start:
                    mask[tok_idx] = 1

        return mask

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    def _stratified_split(self, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """Split by lecture_id to avoid data leakage"""
        lecture_ids = [s['lecture_id'] for s in segments]
        indices = np.arange(len(segments))

        # First split: train+val vs test
        gss1 = GroupShuffleSplit(n_splits=1, test_size=self.test_ratio,
                                random_state=self.seed)
        trainval_idx, test_idx = next(gss1.split(indices, groups=lecture_ids))

        # Second split: train vs val
        trainval_lectures = [lecture_ids[i] for i in trainval_idx]
        relative_val = self.val_ratio / (1 - self.test_ratio)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val,
                                random_state=self.seed)
        train_sub, val_sub = next(gss2.split(trainval_idx, groups=trainval_lectures))

        train_idx = trainval_idx[train_sub]
        val_idx = trainval_idx[val_sub]

        return {
            'train': [segments[i] for i in train_idx],
            'val': [segments[i] for i in val_idx],
            'test': [segments[i] for i in test_idx],
        }

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def build(self) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info("Building BART Dataset")
        self.logger.info("=" * 60)

        # Load data from Step 2
        input_path = self.data_dir / "dataset_with_summaries.json"
        assert input_path.exists(), f"Run 06_generate_pseudo_summaries.py first! Missing {input_path}"

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data['segments']
        if self.test_mode:
            segments = segments[:10]

        self._load_tokenizer()

        # Create entity masks
        if self.entity_loss_weight > 1.0:
            self.logger.info("Creating entity masks for entity-aware loss …")
            for seg in tqdm(segments, desc="Entity masks"):
                seg['entity_mask'] = self._create_entity_mask(seg['pseudo_summary'])

            # Log a sample mask for verification
            sample = next((s for s in segments if sum(s.get('entity_mask', [])) > 0), None)
            if sample:
                doc = self._nlp(sample['pseudo_summary'])
                ents = [ent.text for ent in doc.ents]
                chunks = [c.text for c in doc.noun_chunks if len(c.text.split()) <= 3]
                mask_sum = sum(sample['entity_mask'])
                self.logger.info(f"[Entity Mask Sample] segment: {sample['segment_id']}")
                self.logger.info(f"  Entities found: {ents[:10]}")
                self.logger.info(f"  Noun chunks: {chunks[:10]}")
                self.logger.info(f"  Mask tokens marked: {mask_sum}/{len(sample['entity_mask'])}")
        else:
            self.logger.info("Entity-aware loss disabled (weight=1.0)")

        # Split data (grouped by lecture)
        self.logger.info("Splitting data (stratified by lecture) …")
        splits = self._stratified_split(segments)

        self.logger.info(f"  Train: {len(splits['train'])}")
        self.logger.info(f"  Val:   {len(splits['val'])}")
        self.logger.info(f"  Test:  {len(splits['test'])}")

        # Tokenization stats
        self.logger.info("Checking tokenization lengths …")
        trunc_count = 0
        for seg in segments:
            tok = self._tokenizer(seg['formatted_input'], truncation=False)
            if len(tok['input_ids']) > self.max_input:
                trunc_count += 1

        trunc_pct = (trunc_count / len(segments)) * 100
        self.logger.info(f"Truncated inputs: {trunc_count}/{len(segments)} ({trunc_pct:.1f}%)")
        if trunc_pct > 10:
            self.logger.warning(f"⚠️  High truncation rate ({trunc_pct:.1f}%). "
                                "Consider increasing max_input_length.")

        # Create PyTorch datasets
        datasets = {}
        for split_name, split_data in splits.items():
            datasets[split_name] = LectureSummarizationDataset(
                split_data, self._tokenizer, self.max_input, self.max_target
            )

        # Save everything
        save_data = {
            'splits': splits,
            'tokenizer_path': str(self.data_dir / "tokenizer"),
            'max_input_length': self.max_input,
            'max_target_length': self.max_target,
            'special_tokens': self.special_tokens,
            'entity_loss_weight': self.entity_loss_weight,
            'statistics': {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test']),
                'total': len(segments),
                'truncation_pct': round(trunc_pct, 1),
                'n_lectures_train': len(set(s['lecture_id'] for s in splits['train'])),
                'n_lectures_val': len(set(s['lecture_id'] for s in splits['val'])),
                'n_lectures_test': len(set(s['lecture_id'] for s in splits['test'])),
            },
        }

        out_path = self.data_dir / "bart_tokenized_dataset.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(save_data, f)

        self.logger.info(f"Saved tokenized dataset → {out_path}")
        self.logger.info("✅ Dataset creation complete!")
        return save_data


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Create BART fine-tuning dataset")
    parser.add_argument("--test", action="store_true", help="Test mode (10 samples)")
    args = parser.parse_args()

    config = config_loader.load_all()
    builder = BARTDatasetBuilder(config, test_mode=args.test)
    result = builder.build()

    stats = result['statistics']
    print(f"\n{'=' * 50}")
    print("BART Dataset Created!")
    print(f"  Train: {stats['train']} samples ({stats['n_lectures_train']} lectures)")
    print(f"  Val:   {stats['val']} samples ({stats['n_lectures_val']} lectures)")
    print(f"  Test:  {stats['test']} samples ({stats['n_lectures_test']} lectures)")
    print(f"  Truncation: {stats['truncation_pct']}%")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
