"""
Dataset Cleaning, Deduplication, and Validation Module.
Cleans raw synthetic data via length filters, loop detectors, template leakage checks, and language tag normalization.
Performs pairwise Jaccard near-duplicate comparison scaled using multiprocessing across CPU cores.
"""

import os
import sys
import json
import re
import yaml
import argparse
from pathlib import Path
import binascii
from collections import Counter
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Force UTF-8 stdout/stderr on Windows to prevent UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

def load_config(config_path="config.yaml"):
    """Load configuration from YAML."""
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file '{config_path}' not found.")
        sys.exit(1)
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse configuration file '{config_path}': {e}")
        sys.exit(1)

def get_char_ngrams(text, n=4):
    """Generate character n-grams from text for similarity checks."""
    text = re.sub(r'\s+', '', text.lower()) # strip whitespace
    if len(text) < n:
        return set([text])
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def detect_repetition(text, n=4, threshold=3):
    """Detect repetitive loops of n-grams in text."""
    words = re.findall(r'\w+', text.lower())
    if len(words) < n:
        return False
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    for ngram, count in counts.items():
        if count > threshold:
            return True
    return False

def contains_template_leakage(text):
    """Check if the text contains common template indicators or system markers."""
    patterns = [
        r'\[insert\b',
        r'\{prompt\}',
        r'\[your name\]',
        r'<system>',
        r'as an AI language model',
        r'according to the prompt',
        r'in the given text',
        r'\[placeholder\]'
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def normalize_lang_tag(lang):
    """Normalize language tags to standard ISO-639-1."""
    if not lang:
        return "en"
    lang = lang.strip().lower()
    mapping = {
        "hindi": "hi", "hin": "hi",
        "telugu": "te", "tel": "te",
        "tamil": "ta", "tam": "ta",
        "english": "en", "eng": "en",
        "marathi": "mr", "mar": "mr",
        "bengali": "bn", "ben": "bn",
        "gujarati": "gu", "guj": "gu",
        "kannada": "kn", "kan": "kn",
        "malayalam": "ml", "mal": "ml",
        "oriya": "or", "ori": "or", "odia": "or",
        "punjabi": "pa", "pan": "pa",
        "assamese": "as", "asm": "as"
    }
    return mapping.get(lang, lang[:2])

# Global variables for worker processes to access pre-computed n-grams efficiently without copy overhead
_global_ngrams = []

def init_worker(ngrams_list):
    """Initialize global n-grams list in each worker process."""
    global _global_ngrams
    _global_ngrams = ngrams_list

def check_duplicate_worker(index_range, threshold):
    """Check a range of samples for duplicates against all preceding samples."""
    duplicates = []
    for idx in index_range:
        current_ngrams = _global_ngrams[idx]
        is_dup = False
        for j in range(idx):
            sim = jaccard_similarity(current_ngrams, _global_ngrams[j])
            if sim > threshold:
                is_dup = True
                break
        if is_dup:
            duplicates.append(idx)
    return duplicates


def compute_signatures_chunk(chunk_data):
    """Worker function to compute MinHash signatures for a chunk of n-gram sets."""
    # chunk_data is a list of (sample, ngrams)
    signatures = []
    num_hashes = 32
    for _, ngrams in chunk_data:
        sig = []
        for i in range(num_hashes):
            min_val = 0xffffffff
            for ngram in ngrams:
                # CRC32 hash (stable across processes)
                val = binascii.crc32(f"{ngram}:{i}".encode('utf-8'))
                if val < min_val:
                    min_val = val
            sig.append(min_val)
        signatures.append(sig)
    return signatures

def process_single_sample_validation(sample_data):
    """Validates structure, length, repetition, and normalizes fields for a single raw sample."""
    sample, allowed_langs, min_len, max_len, repetition_n, repetition_thresh = sample_data
    
    # Unwrap split-level nesting if present
    if len(sample) == 1 and isinstance(list(sample.values())[0], dict):
        sample = list(sample.values())[0]

    required_fields = ["instruction", "response", "task_type", "language"]
    if not all(field in sample and sample[field] is not None for field in required_fields):
        return None, "missing_fields"

    instruction = str(sample["instruction"]).strip()
    response = str(sample["response"]).strip()
    task_type = str(sample["task_type"]).strip()
    lang = normalize_lang_tag(str(sample["language"]))

    sample["instruction"] = instruction
    sample["response"] = response
    sample["language"] = lang

    if lang not in allowed_langs:
        return None, f"invalid_language_{lang}"

    if len(response) < min_len:
        return None, "too_short"
    if len(response) > max_len:
        return None, "too_long"

    if contains_template_leakage(instruction) or contains_template_leakage(response):
        return None, "template_leakage"

    if detect_repetition(response, n=repetition_n, threshold=repetition_thresh):
        return None, "repetition_loop"

    if task_type == "reasoning":
        rationale = sample.get("rationale")
        if not rationale or not str(rationale).strip() or len(str(rationale).strip()) < 5:
            return None, "missing_reasoning_rationale"

    # Compute and attach character n-grams for deduplication stage
    combined_text = instruction + " " + response
    char_ngrams = get_char_ngrams(combined_text, n=4)

    return (sample, char_ngrams), "valid"

def clean_and_validate(config):
    raw_path = Path(config["dataset"]["raw_file"])
    if not raw_path.exists():
        print(f"[ERROR] Raw dataset file {raw_path} does not exist. Please run generation first.")
        return False

    cfg_clean = config["cleaning"]
    allowed_langs = cfg_clean.get("allowed_languages", ["en", "hi", "te", "ta"])
    near_dup_thresh = cfg_clean.get("near_dup_threshold", 0.85)
    max_len = cfg_clean.get("max_response_length", 1500)
    min_len = cfg_clean.get("min_response_length", 10)
    repetition_thresh = cfg_clean.get("repetition_threshold", 3)
    repetition_n = cfg_clean.get("repetition_ngram", 4)
    num_workers = cfg_clean.get("num_workers", 4)
    
    print("=== Parallel Cleaning and Deduplication ===")
    print(f"Using {num_workers} CPU cores for parallel processing.")
    print(f"Loading raw dataset from {raw_path}...")
    
    raw_samples = []
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        raw_samples.append(json.loads(line))
                    except json.JSONDecodeError as decode_err:
                        print(f"[WARNING] JSON decode error at line {line_idx}: {decode_err}")
                        print(f"  Snippet of skipped line: {line.strip()[:100]}...")
    except Exception as e:
        print(f"[ERROR] Failed to read raw dataset file '{raw_path}': {e}")
        return False
                    
    total_raw = len(raw_samples)
    print(f"Total raw samples loaded: {total_raw}")
    
    if total_raw == 0:
        print("[ERROR] No valid raw samples loaded from file. Cannot clean dataset.")
        return False

    # Step 1: Parallel validation and n-gram precomputation
    print("\n[Stage 1/2] Running parallel field validation and n-gram extraction...")
    validated_data = []
    skipped_stats = Counter()
    
    # Package arguments for worker pool
    validation_args = [
        (s, allowed_langs, min_len, max_len, repetition_n, repetition_thresh) 
        for s in raw_samples
    ]
    
    # Process validations in parallel using worker pool
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_sample_validation, validation_args, chunksize=100))
        
    for sample_res, status in results:
        if sample_res is not None:
            validated_data.append(sample_res) # tuple of (sample, ngrams)
        else:
            skipped_stats[status] += 1
            
    num_validated = len(validated_data)
    print(f"Validated samples: {num_validated} (Filtered out {total_raw - num_validated})")
    
    if num_validated == 0:
        print("[ERROR] No samples passed structural validation! Deduplication aborted.")
        return False
        
    # Step 2: Parallel MinHash LSH Near-Deduplication
    print("\n[Stage 2/2] Running parallel MinHash LSH near-deduplication...")
    
    # Chunk data for parallel signature calculation
    sig_chunk_size = max(1, num_validated // (num_workers * 2))
    chunks = [validated_data[i : i + sig_chunk_size] for i in range(0, num_validated, sig_chunk_size)]
    
    print(f"Computing MinHash signatures in parallel using {num_workers} workers...")
    signatures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_signatures_chunk, chunk) for chunk in chunks]
        for future in futures:
            signatures.extend(future.result())
            
    # LSH Banding parameters: 32 hashes, 8 bands of size 4
    num_hashes = 32
    band_size = 4
    num_bands = num_hashes // band_size
    buckets = [{} for _ in range(num_bands)]
    
    print("Grouping signatures into LSH bands...")
    candidate_pairs = set()
    for idx in range(num_validated):
        sig = signatures[idx]
        for b in range(num_bands):
            band_sig = tuple(sig[b * band_size : (b + 1) * band_size])
            if band_sig in buckets[b]:
                for prev_idx in buckets[b][band_sig]:
                    candidate_pairs.add((prev_idx, idx))
                buckets[b][band_sig].append(idx)
            else:
                buckets[b][band_sig] = [idx]
                
    print(f"Found {len(candidate_pairs)} candidate pairs. Verifying with exact Jaccard similarity...")
    duplicate_indices = set()
    samples_list = [item[0] for item in validated_data]
    
    # Verify candidate pairs using exact Jaccard calculation
    for prev_idx, idx in sorted(candidate_pairs, key=lambda x: (x[1], x[0])):
        if prev_idx in duplicate_indices or idx in duplicate_indices:
            continue
        sim = jaccard_similarity(validated_data[prev_idx][1], validated_data[idx][1])
        if sim > near_dup_thresh:
            duplicate_indices.add(idx)
            
    # Filter out duplicate indices
    cleaned_samples = [
        samples_list[i] for i in range(num_validated)
        if i not in duplicate_indices
    ]
    
    skipped_stats["near_duplicate"] = len(duplicate_indices)
    
    print("\n--- Skip Summary ---")
    for reason, count in sorted(skipped_stats.items()):
        print(f"Skipped due to {reason}: {count}")
    print(f"Total cleaned samples kept: {len(cleaned_samples)}")
    
    if not cleaned_samples:
        print("[ERROR] No samples passed the deduplication filters!")
        return False
        
    # Write cleaned samples to JSONL
    cleaned_path = Path(config["dataset"]["cleaned_file"])
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cleaned_path, "w", encoding="utf-8") as f:
            for sample in cleaned_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"[OK] Cleaned dataset saved to: {cleaned_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save cleaned dataset to '{cleaned_path}': {e}")
        return False
    
    # Write to Parquet
    parquet_path = Path(config["dataset"]["parquet_file"])
    try:
        df = pd.DataFrame(cleaned_samples)
        df.to_parquet(parquet_path, index=False)
        print(f"[OK] Cleaned dataset saved to Parquet: {parquet_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save Parquet format: {e}. PyArrow or pandas output error.")
        
    # Write a small preview file (up to 3 samples per task type)
    preview_path = Path(config["dataset"]["preview_file"])
    preview_samples = []
    task_groups = {}
    for sample in cleaned_samples:
        t = sample["task_type"]
        if t not in task_groups:
            task_groups[t] = []
        if len(task_groups[t]) < 3:
            task_groups[t].append(sample)
            preview_samples.append(sample)
            
    try:
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(preview_samples, f, ensure_ascii=False, indent=2)
        print(f"[OK] Preview file saved to: {preview_path}")
    except Exception as e:
        print(f"[WARNING] Failed to write preview file: {e}")

    # Generate the Markdown summary report
    summary_path = Path(config["dataset"]["summary_file"])
    task_dist = Counter(s["task_type"] for s in cleaned_samples)
    lang_dist = Counter(s["language"] for s in cleaned_samples)
    difficulty_dist = Counter(s["difficulty"] for s in cleaned_samples)
    
    total_samples = len(cleaned_samples)
    target_mix = config["dataset"]["target_mix"]
    
    report_content = f"""# Synthetic Dataset Summary Report

This dataset was generated using Sarvam AI teacher architectures and processed for training a tiny, high-performance Student Language Model (SLM).

## General Statistics
- **Total Cleaned Samples**: {total_samples}
- **Raw Samples Evaluated**: {total_raw}
- **Filtered Out Samples**: {total_raw - total_samples} (Near-duplicates, repetitions, length-bounds, style-breaks)

## Task Distribution Analysis

| Task Type | Count | Percentage | Target Ratio | Status |
| :--- | :--- | :--- | :--- | :--- |
"""
    for t_type, target_ratio in target_mix.items():
        count = task_dist.get(t_type, 0)
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        diff = percentage - (target_ratio * 100)
        status = "Balanced" if abs(diff) < 5 else ("Low" if diff < 0 else "High")
        report_content += f"| {t_type} | {count} | {percentage:.1f}% | {target_ratio*100:.1f}% | {status} |\n"

    report_content += "\n## Language Distribution\n\n| Language | Count | Percentage |\n| :--- | :--- | :--- |\n"
    for lang, count in lang_dist.most_common():
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        report_content += f"| {lang.upper()} | {count} | {percentage:.1f}% |\n"

    report_content += "\n## Difficulty Distribution\n\n| Difficulty | Count | Percentage |\n| :--- | :--- | :--- |\n"
    for diff, count in difficulty_dist.most_common():
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        report_content += f"| {diff} | {count} | {percentage:.1f}% |\n"

    report_content += f"""
## Cleaning Filters Applied
- **Near-Deduplication Jaccard Threshold**: {near_dup_thresh} (on character 4-grams)
- **Response Length Bounds**: {min_len} - {max_len} characters
- **Repetitive Loop Check**: N-gram count check on {repetition_n}-grams with threshold {repetition_thresh}
- **Template Leakage Check**: Automated scan for placeholders, prompt leakage, and assistant prefixes.
- **Multiprocessing Workers**: {num_workers} parallel CPU cores
"""
    
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"[OK] Dataset summary report saved to: {summary_path}")
    except Exception as e:
        print(f"[WARNING] Failed to write dataset summary report: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Clean and Validate Generated Dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    clean_and_validate(config)

if __name__ == "__main__":
    main()
