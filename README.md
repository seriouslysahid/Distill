# Sarvam Teacher-Student Synthetic Dataset Pipeline

This repository contains a clean, robust, and reproducible pipeline for generating a high-quality, multilingual, speech-style, and reasoning-aware synthetic dataset using a teacher model (e.g., Sarvam-105B or Sarvam-2B) and preparing it for training a tiny Student Language Model (SLM).

## Features
- **Model Downloader & Verifier**: Fast download of model assets with a custom PyTorch meta-device check to verify architecture correctness and compatibility without causing memory/RAM crashes.
- **Distilabel Generation Pipeline**: A configurable pipeline supporting local execution (via HF Transformers), remote API execution, and a high-fidelity CPU Mock generator for fast development and validation.
- **Strict Structured Prompts**: Prompts that force structured JSON outputs, metadata annotation (task_type, language, style, difficulty, teacher), and concise reasoning rationales.
- **Advanced Cleaning Engine**: Near-duplicate filter (via Jaccard n-gram similarity), repetition loop checkers, template leakage scanners, and language-tag normalizers.
- **Balanced Mix Validator**: Checks and reports the dataset composition against the recommended SLM training target ratios.
- **Multilingual & Speech-Aware**: Focuses on Indian languages and natural speech/transcript style behavior (fillers, conversational flow, colloquialisms).

---

## Recommended Dataset Mix
- **35% Multilingual Chat & Instruction Following**
- **20% Speech / Transcript / Spoken-Dialogue style**
- **15% Translation and Paraphrase**
- **15% Reasoning and Problem Solving**
- **10% Summarization & Information Transformation**
- **5% Refusal & Safety**
- **5% Miscellaneous Utility**

---

## Installation

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Command Line Interface (CLI)

Use `main.py` as the entrypoint to coordinate the pipeline steps:

### 1. Download and Verify Teacher Model
To download the configurations and verify compatibility without loading hundreds of gigabytes of weights into memory:
```bash
python main.py download --config_only
```
To run the full weight download (if disk space and hardware permit):
```bash
python main.py download
```

### 2. Generate Synthetic Dataset
To run the data generator:
```bash
# Using the fast, high-fidelity mock generator (recommended for local testing/CPU)
python main.py generate --backend mock --num_samples 200

# Using a local Hugging Face model (requires local GPU)
python main.py generate --backend transformers --num_samples 200
```

### 3. Clean and Export Dataset
To run the cleaning filters, deduplication, and export to JSONL, Parquet, and preview:
```bash
python main.py clean
```

### 4. Run the Full Pipeline
To run all three steps in sequence (download, generation, cleaning/export) with a single command:
```bash
python main.py all --backend mock --num_samples 200
```

---

## Configurations (`config.yaml`)

Edit `config.yaml` to adjust pipeline settings:
- `model.teacher_id`: The Hugging Face repo ID (e.g. `sarvamai/sarvam-105b` or `sarvamai/sarvam-2b-v0.5`).
- `generation.backend`: Select generation backend (`mock` or `transformers`).
- `generation.num_samples`: The number of samples to generate.
- `cleaning.near_dup_threshold`: Jaccard similarity threshold for near-duplicate filtering (default `0.85`).
- `cleaning.max_response_length`: Maximum character length of response to optimize for tiny SLMs.
- `cleaning.allowed_languages`: List of allowed ISO-639-1 language codes.

---

## Output Artifacts

All results are exported to the `./dataset` directory:
1. `dataset/raw_dataset.jsonl`: Raw generations directly from the pipeline.
2. `dataset/cleaned_dataset.jsonl`: Cleaned, filtered, and deduplicated dataset.
3. `dataset/cleaned_dataset.parquet`: Binary training-ready format.
4. `dataset/dataset_preview.json`: A small preview file containing 3 samples per task category.
5. `dataset/summary_report.md`: Markdown summary report with statistics and mix distribution analysis.
