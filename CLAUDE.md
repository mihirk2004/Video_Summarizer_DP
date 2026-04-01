# Lecture Video Summarization & Segmentation — Project Context

## Project Goal
Automate creation of structured technical documents (Markdown/PDF) from lecture videos by combining spoken transcripts with visual assets (equations, diagrams, code snippets) using multi-modal AI processing.

**Dataset**: 249 annotated lecture JSONs in `data/annotations/`

## Architecture Overview (4 Phases)

### Phase 1 — Preprocessing & Data Extraction ✅ DONE
- **FFmpeg**: Audio extraction from raw videos
- **Librosa**: Audio cleaning & noise reduction
- **OpenAI Whisper** (`small` model): Speech-to-text with word-level timestamps
- **Output**: Timestamped transcripts + cleaned audio in `data/processed/{lecture_id}/`

### Phase 2 — Content Analysis & Classification ✅ DONE
- **spaCy + scispaCy** (`en_core_sci_sm`): STEM NER for equations, concepts, methods
- **BERTopic** + Sentence-BERT (`all-MiniLM-L6-v2`): Topic segmentation with boundary detection
- **SBERT** (`all-mpnet-base-v2`): Semantic similarity embeddings
- **CLIP**: Frame concept detection (equation, diagram, code, gesture, slide)
- **YOLOv8**: Face/instructor detection
- **Trained models**: `models/text/ner/`, `models/text/topic_seg/`, `models/text/embeddings_SBERT/`

### Phase 3 — Fine-Tuning & Summarization ✅ DONE (evaluation remaining)
- **BART** (`facebook/bart-large-cnn`): Fine-tune for lecture summarization
- **FLAN-T5** (`google/flan-t5-base`): Pseudo-summary generation, boundary refinement, LLM enhancement
- **Hybrid Pipeline**: Topic-Seg → LLM Refine → BART Summarize → LLM Enhance

### Phase 4 — Multimodal Fusion & Document Generation 🔧 IN PROGRESS
- **CLIP** (`openai/clip-vit-base-patch32`): LoRA fine-tuned for lecture text↔image alignment
- **CLIP+BART Prefix Fusion**: Visual tokens prepended to BART encoder input
- **LLaVA** (`llava-1.5-7b-hf`): LoRA fine-tuned for frame captioning
- **Document Generation**: PDF/Markdown/HTML with MathJax
- **Scripts**: `11_prepare_multimodal_data.py` → `12_finetune_clip.py` → `13_precompute_clip_embeddings.py` → `14_train_clip_bart_fusion.py` → `15_multimodal_inference.py` → `16_generate_documents.py`
- **Models saved to**: `models/multimodal/{clip_finetuned,clip_bart_fusion,llava_lora}/`


## Data Structure

### Annotation JSONs (`data/annotations/lecture_XXX_annotated.json`)
152 files, each containing:
```json
{
  "video_id": "lecture_001",
  "metadata": { "subject": "Computer Science", "source": "YouTube", ... },
  "processing": {
    "audio_path": "data/processed/lecture_001/audio.wav",
    "transcript": {
      "text": "Full transcript...",
      "segments": [
        { "start": 0.0, "end": 2.74, "text": "Hello...", "words": [...] }
      ]
    },
    "frames": [
      { "frame_index": 0, "timestamp": 5.0, "concepts": ["equation"], "quality": "Good" }
    ]
  }
}
```
**Note**: Segments are Whisper-level (1-5 seconds each, ~1 sentence). No summary fields exist yet.

### Topic Segmentation Output (`models/text/topic_seg/`)
- `segmentation_results.json` (~13MB): Topic assignments, boundaries, lecture mapping
- `topic_representations.json`: Per-topic keyword representations
- `topic_model`: Saved BERTopic model (~407MB)

## Script Conventions
- Scripts numbered sequentially: `01_prepare_data.py`, `02_train_stem_ner.py`, etc.
- All scripts in `scripts/text_processing/`, utilities in `scripts/utils/`
- Config loaded via `ConfigLoader` singleton from `config/text_config.yaml` + `config/paths.yaml`
- Logging via `setup_logger()`, experiments via `log_experiment()`
- Models saved to `models/text/{model_name}/`
- Results/plots to `results/text_models/`

## Config Keys (text_config.yaml)
- `project.random_seed`: 42
- `data.train_ratio/val_ratio/test_ratio`: 0.7/0.15/0.15
- `training.device`: "cuda"
- `training.fp16`: false
- `training.gradient_accumulation_steps`: 1

## Hardware
- Local: 4GB GPU (CUDA)
- Alternative: Google Colab (T4 16GB)
- Must use gradient accumulation + fp16 for BART-large

## Key Dependencies
`torch>=2.0`, `transformers>=4.30`, `sentence-transformers>=2.2`, `spacy>=3.7`, `bertopic>=0.15`, `datasets`
