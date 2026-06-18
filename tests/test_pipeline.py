"""
Unit Testing Suite for the Synthetic Dataset Cleaning & Deduplication Module.
Verifies logic correctness, data integrity filters, and format normalization rules.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path to ensure clean import of local modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from clean_dataset import (
    get_char_ngrams,
    jaccard_similarity,
    detect_repetition,
    contains_template_leakage,
    normalize_lang_tag,
    process_single_sample_validation
)

def test_get_char_ngrams():
    """Verify n-gram tokenization, whitespace stripping, and case insensitivity."""
    text1 = "Hello World"
    # Stripped text: "helloworld" (10 chars)
    # 4-grams (7 total): "hell", "ello", "llow", "lowo", "owor", "worl", "orld"
    ngrams1 = get_char_ngrams(text1, n=4)
    assert len(ngrams1) == 7
    assert "hell" in ngrams1
    assert "orld" in ngrams1
    
    # Check case insensitivity
    text2 = "HELLO WORLD"
    ngrams2 = get_char_ngrams(text2, n=4)
    assert ngrams1 == ngrams2
    
    # Test short string
    assert get_char_ngrams("abc", n=4) == {"abc"}

def test_jaccard_similarity():
    """Verify Jaccard similarity mathematical bounds [0.0, 1.0]."""
    set1 = {"a", "b", "c", "d"}
    set2 = {"c", "d", "e", "f"}
    # intersection: {"c", "d"} (size 2)
    # union: {"a", "b", "c", "d", "e", "f"} (size 6)
    # similarity: 2/6 = 0.3333333333333333
    assert abs(jaccard_similarity(set1, set2) - 0.33333) < 0.001
    
    # Disjoint
    assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0
    
    # Identical
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    
    # Empty sets
    assert jaccard_similarity(set(), {"a"}) == 0.0
    assert jaccard_similarity({"a"}, set()) == 0.0
    assert jaccard_similarity(set(), set()) == 0.0

def test_detect_repetition():
    """Verify word-level n-gram repetition detection."""
    # Healthy string without repetition loops
    assert not detect_repetition("The quick brown fox jumps over the lazy dog.", n=4, threshold=3)
    
    # String with repeating 4-grams exceeding threshold 3
    repeating_text = "I love tea. I love tea. I love tea. I love tea. I love tea."
    # Words: ["i", "love", "tea", "i", "love", "tea", "i", "love", "tea", "i", "love", "tea", "i", "love", "tea"]
    # n-gram ("i", "love", "tea", "i") occurs 4 times
    assert detect_repetition(repeating_text, n=4, threshold=3)
    
    # Short string
    assert not detect_repetition("no rep", n=4, threshold=3)

def test_contains_template_leakage():
    """Verify template placeholder detection."""
    assert not contains_template_leakage("Write a letter to Aarav.")
    assert contains_template_leakage("Write a letter to [Insert Name].")
    assert contains_template_leakage("Here is the answer to the {prompt}.")
    assert contains_template_leakage("As an AI language model, I cannot fulfill this.")
    assert contains_template_leakage("Replace [placeholder] with the value.")

def test_normalize_lang_tag():
    """Verify language tag normalization to standard ISO-639-1."""
    assert normalize_lang_tag("hindi") == "hi"
    assert normalize_lang_tag("HIN") == "hi"
    assert normalize_lang_tag("TELUGU") == "te"
    assert normalize_lang_tag("Tamil") == "ta"
    assert normalize_lang_tag("English") == "en"
    assert normalize_lang_tag("Odia") == "or"
    assert normalize_lang_tag("unknown") == "un"
    assert normalize_lang_tag("") == "en"

def test_process_single_sample_validation():
    """Verify the comprehensive validation rules and unwrapping logic."""
    allowed_langs = ["en", "hi", "te", "ta"]
    params = (allowed_langs, 10, 1500, 4, 3) # allowed, min, max, ngram, threshold
    
    # 1. Happy Path (Valid sample)
    valid_sample = {
        "instruction": "Explain photosynthesis.",
        "response": "Photosynthesis is the process plant leaves use to convert sunlight to energy.",
        "task_type": "multilingual_chat",
        "language": "en",
        "difficulty": "medium",
        "style": "formal"
    }
    res, status = process_single_sample_validation((valid_sample, *params))
    assert status == "valid"
    assert res is not None
    assert res[0]["language"] == "en" # Normalized
    assert len(res[1]) > 0 # Has n-grams computed
    
    # 2. Wrapped Split format (e.g. {"train": {...}})
    wrapped_sample = {"train": valid_sample.copy()}
    res, status = process_single_sample_validation((wrapped_sample, *params))
    assert status == "valid"
    assert res is not None
    assert res[0]["instruction"] == "Explain photosynthesis."
    
    # 3. Missing fields check
    invalid_sample_1 = {"instruction": "Hello", "language": "en"}
    res, status = process_single_sample_validation((invalid_sample_1, *params))
    assert status == "missing_fields"
    assert res is None
    
    # 4. Too short response check
    invalid_sample_2 = valid_sample.copy()
    invalid_sample_2["response"] = "Short"
    res, status = process_single_sample_validation((invalid_sample_2, *params))
    assert status == "too_short"
    
    # 5. Invalid language check
    invalid_sample_3 = valid_sample.copy()
    invalid_sample_3["language"] = "french" # Not in allowed list
    res, status = process_single_sample_validation((invalid_sample_3, *params))
    assert status == "invalid_language_fr"
    
    # 6. Template leakage check
    invalid_sample_4 = valid_sample.copy()
    invalid_sample_4["instruction"] = "Write to [Insert Name]."
    res, status = process_single_sample_validation((invalid_sample_4, *params))
    assert status == "template_leakage"
    
    # 7. Repetitive loop check
    invalid_sample_5 = valid_sample.copy()
    invalid_sample_5["response"] = "Plant leaves convert sunlight Plant leaves convert sunlight Plant leaves convert sunlight Plant leaves convert sunlight Plant leaves convert sunlight."
    res, status = process_single_sample_validation((invalid_sample_5, *params))
    assert status == "repetition_loop"
    
    # 8. Reasoning without rationale check
    invalid_sample_6 = valid_sample.copy()
    invalid_sample_6["task_type"] = "reasoning"
    invalid_sample_6["rationale"] = None
    res, status = process_single_sample_validation((invalid_sample_6, *params))
    assert status == "missing_reasoning_rationale"
