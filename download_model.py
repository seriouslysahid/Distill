import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Force enable hf_transfer for Rust-based high-speed parallel downloads
# This environment variable MUST be set before importing huggingface_hub
try:
    import hf_transfer
    hf_transfer_enabled = True
except ImportError:
    hf_transfer_enabled = False

import argparse
import yaml
import torch
import shutil
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows to prevent UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from huggingface_hub import snapshot_download, HfApi
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights

def get_free_space_gb(path):
    """Return free disk space in Gigabytes."""
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML."""
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Using defaults.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def verify_meta_load(model_id, trust_remote=True):
    """Verify that the model architecture can load on the PyTorch meta device."""
    print(f"\nVerifying model architecture definition for '{model_id}' on meta device...")
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote)
        
        # Load model structure with accelerate's empty weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote)
            
        print("[OK] Model structure successfully verified and loaded on 'meta' device!")
        print(f"  Architecture: {config.architectures}")
        print(f"  Vocabulary size: {config.vocab_size}")
        if hasattr(config, "num_hidden_layers"):
            print(f"  Layers: {config.num_hidden_layers}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to verify model structure: {e}")
        return False

def verify_tokenizer(model_id, trust_remote=True):
    """Verify tokenizer loads and can encode/decode text."""
    print(f"\nVerifying tokenizer for '{model_id}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)
        test_text = "नमस्ते, आप कैसे हैं? How can I help you today?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print("[OK] Tokenizer successfully loaded!")
        print(f"  Test text: '{test_text}'")
        print(f"  Encoded token length: {len(tokens)}")
        print(f"  Decoded back: '{decoded}'")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to verify tokenizer: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and Verify Hugging Face Teacher Models")
    parser.add_argument("--model_id", type=str, help="Hugging Face Model ID to download")
    parser.add_argument("--verify_only", action="store_true", help="Only verify already downloaded or remote model files")
    parser.add_argument("--config_only", action="store_true", help="Only download configuration and tokenizer files (skip weights)")
    parser.add_argument("--local_dir", type=str, help="Directory to save downloaded files")
    args = parser.parse_args()

    # Load defaults from config
    cfg = load_config()
    model_id = args.model_id or cfg.get("model", {}).get("teacher_id", "sarvamai/sarvam-2b-v0.5")
    local_dir = args.local_dir or cfg.get("model", {}).get("local_dir", "./models/teacher")
    
    # Redirect HF cache to prevent home directory disk exhaustion
    hf_cache_dir = os.path.abspath(os.path.join(local_dir, ".cache"))
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_MODULES_CACHE"] = os.path.join(hf_cache_dir, "modules")
    
    # Retrieve HF Token if available
    token_var = cfg.get("model", {}).get("token_env_var", "HF_TOKEN")
    hf_token = os.environ.get(token_var, None)
    
    print(f"=== Model Downloader & Verifier ===")
    print(f"Target Model ID: {model_id}")
    print(f"Local Storage Directory: {local_dir}")
    if hf_transfer_enabled and not args.config_only:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("Rust-based High-Speed Downloader (hf_transfer): ENABLED")
    else:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print("Rust-based High-Speed Downloader (hf_transfer): DISABLED (using standard downloader to support ignore patterns)")
        
    if hf_token:
        print("Hugging Face API Token: Detected in environment variables.")
    else:
        print("Hugging Face API Token: Not set (gated models may fail).")

    # If verifying only, skip downloading
    if args.verify_only:
        print("\nSkipping downloads. Verifying model and tokenizer directly...")
        tokenizer_ok = verify_tokenizer(model_id)
        model_ok = verify_meta_load(model_id)
        if tokenizer_ok and model_ok:
            print("\n[SUCCESS] Verification successful!")
            sys.exit(0)
        else:
            print("\n[ERROR] Verification failed!")
            sys.exit(1)

    # Estimate required space (approximate)
    api = HfApi(token=hf_token)
    try:
        model_info = api.model_info(model_id)
        total_size_bytes = sum(getattr(f, "size", 0) or 0 for f in model_info.siblings)
        total_size_gb = total_size_bytes / (1024**3)
        print(f"Remote model folder size: {total_size_gb:.2f} GB")
    except Exception as e:
        print(f"Warning: Could not fetch remote model size info: {e}")
        total_size_gb = 5.0  # fallback estimate for 2B
        if "105b" in model_id.lower():
            total_size_gb = 210.0 # fallback estimate for 105B

    # Disk space check
    dest_path = Path(local_dir).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    free_gb = get_free_space_gb(dest_path.parent)
    print(f"Free disk space on target drive: {free_gb:.2f} GB")

    if not args.config_only and free_gb < total_size_gb:
        print(f"\n[WARNING] Not enough disk space. Requires {total_size_gb:.2f} GB but only {free_gb:.2f} GB is free.")
        print("Switching to '--config_only' mode to fetch tokenizer and configuration files only.")
        args.config_only = True

    # Snapshots download options
    ignore_patterns = []
    if args.config_only:
        print("\nDownloading tokenizer, configs, and safetensors metadata (skipping model weights)...")
        # Ignore main weights files to save bandwidth and space
        ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.gguf"]
    else:
        print(f"\nDownloading snapshot for '{model_id}' to '{local_dir}'...")

    try:
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            ignore_patterns=ignore_patterns,
            token=hf_token,
            max_workers=8 if hf_transfer_enabled else 4
        )
        print(f"[OK] Download completed! Files stored at: {downloaded_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        print("Note: If the model is gated, ensure your HF_TOKEN environment variable is set correctly.")
        print("We will attempt to proceed with online verification using the model ID directly.")

    # Verification
    # We verify against the local directory if downloaded, else fall back to the Hugging Face Hub model ID
    verification_target = local_dir if os.path.exists(local_dir) and os.listdir(local_dir) else model_id
    
    tokenizer_ok = verify_tokenizer(verification_target)
    model_ok = verify_meta_load(verification_target)

    if tokenizer_ok and model_ok:
        print("\n[SUCCESS] Verification successful!")
        sys.exit(0)
    else:
        print("\n[ERROR] Verification failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
