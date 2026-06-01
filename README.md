# Multimodal Video Lecture Summarizor 

Multimodal lecture video summarization system that turns a raw lecture video into a structured document with transcript-aware summaries, frame-aware visual context, and exportable Markdown, HTML, and PDF output.

The project combines speech transcription, topic segmentation, STEM NER, text summarization, visual frame classification, CLIP-based multimodal fusion, and document generation into one pipeline.

## What This Project Does

- Extracts audio from lecture videos and transcribes speech with Whisper.
- Samples and classifies key frames with a hybrid visual model.
- Segments transcripts into topic-aware chunks.
- Builds lecture summaries with BART and optional Mistral refinement.
- Fine-tunes CLIP on lecture text-image pairs.
- Trains a CLIP+BART fusion model for multimodal summarization.
- Generates final lecture notes in Markdown, HTML, and PDF.
- Serves everything through a Flask web app with live progress updates.

## End-to-End Flow

### Runtime pipeline

1. Upload a local video or provide a YouTube URL in the Flask app.
2. `app.py` launches `pipeline.LecturePipeline` in a background thread.
3. Phase 1 extracts audio, transcribes it, samples frames, and runs object/face detection.
4. Phase 2 classifies frames with the hybrid visual classifier.
5. Phase 3 creates transcript segments, scores them, and generates summaries.
6. Phase 4 builds the final document bundle and exposes it for download.

### Training/build pipeline

1. Prepare datasets from annotated lecture JSON files.
2. Train the STEM NER model.
3. Train BERTopic topic segmentation.
4. Train Sentence-BERT similarity embeddings.
5. Prepare BART data and pseudo-summaries.
6. Fine-tune BART for lecture summarization.
7. Prepare multimodal data and CLIP embeddings.
8. Fine-tune CLIP, then train CLIP+BART fusion.
9. Run multimodal inference and generate final documents.

### System Architecture

![System Diagram](Frontend%20Images/System%20Diagram.png)

## Model Inventory

| Model | Purpose | Built By | Output |
| --- | --- | --- | --- |
| Whisper `small` | Audio transcription with timestamps | `process_video.py`, `pipeline.py` | Transcript JSON in job output / processed data |
| YOLOv8 face detector (`yolov8n-face.pt`) | Face / instructor detection on frames | `process_video.py`, `pipeline.py` | Detection metadata in processing JSON |
| spaCy / scispaCy (`en_core_sci_sm`) | STEM NER for equations, concepts, methods | `scripts/text_processing/02_train_stem_ner.py` | `models/text/ner/final/` |
| BERTopic + SBERT (`all-MiniLM-L6-v2`) | Topic segmentation and boundary detection | `scripts/text_processing/03_train_topic_segmentation.py` | `models/text/topic_seg/results/segmentation_results.json` and `topic_model/` |
| Sentence-BERT (`all-mpnet-base-v2`) | Similarity embeddings for lecture segments | `scripts/text_processing/04_train_sentence_bert.py` | `models/text/embeddings_SBERT/final/` |
| FLAN-T5 (`google/flan-t5-large`) | Pseudo-summary generation for training data | `scripts/text_processing/06_generate_pseudo_summaries.py` | `data/processed/bart_dataset/dataset_with_summaries.json` |
| BART (`facebook/bart-large-cnn`) | Main lecture summarizer | `scripts/text_processing/08_train_bart_summarizer.py` | `models/text/bart_summarizer/best_model/` |
| Mistral-7B-Instruct-v0.3 | Summary refinement / hybrid inference | `scripts/inference/lecture_summarizer.py`, `09_llm_enhance_summaries.py`, `10_hybrid_inference.py` | Refined summaries in results files |
| CLIP (`openai/clip-vit-base-patch32`) | Lecture text-image alignment | `scripts/text_processing/12_finetune_clip.py` | `models/multimodal/clip_finetuned/best_model/` |
| CLIP+BART fusion | Multimodal summarization with visual prefix tokens | `scripts/text_processing/14_train_clip_bart_fusion.py` | `models/multimodal/clip_bart_fusion/best_model/` |
| LLaVA `1.5-7B` | Optional frame captioning | `scripts/text_processing/15_multimodal_inference.py` | Optional captions in multimodal results |

## Repository Layout

- `app.py` - Flask web app, upload flow, live progress, document viewer.
- `pipeline.py` - Main four-phase pipeline coordinator used by the web app.
- `process_video.py` - Core video processing utilities used in phase 1.
- `subject_router.py` - Lightweight subject router for CS vs Maths text models.
- `visual_classifier.py` - Hybrid ResNet50 + CLIP frame classifier.
- `scripts/text_processing/` - Training and inference scripts for each phase.
- `scripts/inference/lecture_summarizer.py` - Production summarization API used by the pipeline.
- `scripts/utils/` - Config loading, logging, evaluation, and data loading helpers.
- `config/` - YAML configuration for paths, text models, and summarization settings.
- `templates/` - Flask and generated document templates.
- `data/` - Annotations, processed datasets, frames, and multimodal artifacts.
- `models/` - Saved model weights and trained adapters.
- `results/` - Evaluation outputs and generated documents.

## Key Inputs and Outputs

### Inputs

- Lecture annotation JSON files in `data/annotations/`
- Raw lecture videos or YouTube URLs
- Optional classified frames in `data/frames_reorganized/`

### Outputs

- Per-job artifacts in `output/<job_id>/`
- Final documents in `results/documents/`
- Text model artifacts in `models/text/`
- Multimodal artifacts in `models/multimodal/`
- Processed datasets in `data/processed/`

Typical job output includes:

- `video_metadata.json`
- `frame_classifications.json`
- `segments.json`
- `document_data.json`
- extracted frame images under `output/<job_id>/frames/`

## Setup

### 1. Create a Python environment

Use Python 3.10 or newer. A virtual environment is recommended.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The project also expects external tools such as:

- `ffmpeg`
- `yt-dlp`
- A CUDA-capable GPU for the larger models, if available

### 3. Review config

The config loader merges:

- `config/paths.yaml`
- `config/text_config.yaml`
- `config/summarization_config.yaml`

Environment variables in those files are resolved by `scripts/utils/config_loader.py`.

## Run the Web App

Start the Flask UI:

```bash
python app.py
```

Then open the local server shown in the terminal. The app supports:

- local video upload
- YouTube URL download
- live progress updates via SSE
- document preview and download

## Frontend Screenshots

### Home Page

![Home Page](Frontend%20Images/Home%20Page.jpeg)

### Processing Pipeline

![Processing Pipeline](Frontend%20Images/Processing%20Pipeline.jpeg)

### Summary View

![Summary View](Frontend%20Images/Summary%20View.jpeg)

### Document View

![Document View](Frontend%20Images/Document%20View.jpeg)

## Build The Models

Run these scripts in order if you want to reproduce the training pipeline.

### Phase 1 - Data preparation

```bash
python scripts/text_processing/01_prepare_data.py
```

### Phase 2 - Text understanding

```bash
python scripts/text_processing/02_train_stem_ner.py
python scripts/text_processing/03_train_topic_segmentation.py
python scripts/text_processing/04_train_sentence_bert.py
```

### Phase 3 - Summarization data and BART

```bash
python scripts/text_processing/05_prepare_bart_data.py
python scripts/text_processing/06_generate_pseudo_summaries.py
python scripts/text_processing/07_create_bart_dataset.py
python scripts/text_processing/08_train_bart_summarizer.py
python scripts/text_processing/09_llm_enhance_summaries.py
python scripts/text_processing/10_hybrid_inference.py
```

### Phase 4 - Multimodal fusion

```bash
python scripts/text_processing/11_prepare_multimodal_data.py
python scripts/text_processing/12_finetune_clip.py
python scripts/text_processing/13_precompute_clip_embeddings.py
python scripts/text_processing/14_train_clip_bart_fusion.py
python scripts/text_processing/15_multimodal_inference.py
python scripts/text_processing/16_generate_documents.py
```

## Legacy / Batch Utilities

- `run_pipeline.py` downloads videos, runs the processing script, and prepares annotation export workflows.
- `Phase_1_Data_Extraction/` contains earlier phase-one scripts preserved for reference.
- `annotation_tools/` contains GUI and merger utilities for manual annotation work.

## Notes On Hardware

- The project was designed with a small local GPU in mind for lighter inference and Colab-class GPUs for training.
- BART-large, CLIP fusion, and LLaVA paths benefit from `fp16`, gradient accumulation, and model offloading.
- The pipeline automatically falls back to text-only summarization when multimodal assets are unavailable.

## Citation / Use

If you use this repository in a project or publication, describe it as a lecture video summarization pipeline that combines transcript processing, visual analysis, and multimodal document generation.
