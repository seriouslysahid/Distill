import os
import sys
import subprocess
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# Check for numpy binary incompatibility (NumPy 2.x vs Pandas/PyArrow built on NumPy 1.x)
try:
    import pandas as pd
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        print("\n" + "="*80)
        print("[WARNING] NumPy binary incompatibility detected (likely upgraded to NumPy 2.x by vLLM).")
        print("Restoring NumPy <2 to prevent binary incompatibility with pandas/pyarrow...")
        print("="*80 + "\n")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2"], check=True)
            print("\n[OK] NumPy successfully downgraded. Re-executing command...\n")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as run_err:
            print(f"[ERROR] Failed to downgrade NumPy: {run_err}")
            print("Please run manually: pip install \"numpy<2\"")
            sys.exit(1)
except ImportError:
    pass

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
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_command(command_list, description):
    """Run a command as a subprocess and stream output with premium terminal styling."""
    border = "=" * 80
    print(f"\n{border}")
    print(f"  RUNNING: {description.upper()}")
    print(f"  Command: {' '.join(command_list)}")
    print(f"{border}\n")
    try:
        # Run process and pipe output directly to stdout/stderr
        result = subprocess.run(
            command_list,
            check=True,
            text=True
        )
        print(f"\n{border}")
        print(f"  [OK] {description} completed successfully.")
        print(f"{border}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{border}")
        print(f"  [ERROR] {description} failed with exit code {e.returncode}.")
        print(f"{border}\n")
        return False
    except Exception as e:
        print(f"\n{border}")
        print(f"  [ERROR] Executing {description}: {e}")
        print(f"{border}\n")
        return False

def download_step(args, config):
    """Run the model downloading and verification step."""
    cmd = [sys.executable, "download_model.py"]
    if args.config_only:
        cmd.append("--config_only")
    if args.verify_only:
        cmd.append("--verify_only")
    if args.model_id:
        cmd.extend(["--model_id", args.model_id])
    if args.local_dir:
        cmd.extend(["--local_dir", args.local_dir])
        
    return run_command(cmd, "Model download & verification")

def cleanup_zombie_gpus():
    """Find and terminate zombie processes holding GPU memory to prevent OOM."""
    import os
    import sys
    border = "=" * 80
    print(f"\n{border}")
    print("  [GPU HEALTH] Running active process cleanup...")
    
    current_pid = os.getpid()
    parent_pid = os.getppid()
    killed_any = False
    
    # Method 1: Using psutil to scan process list for orphaned python/vllm instances
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                pid = proc.info['pid']
                if pid == current_pid or pid == parent_pid:
                    continue
                cmdline = proc.info['cmdline']
                if cmdline:
                    cmd_str = " ".join(cmdline).lower()
                    if ("python" in cmd_str or "vllm" in cmd_str or "distilabel" in cmd_str) and \
                       ("generate_dataset" in cmd_str or "main.py" in cmd_str or "vllm" in cmd_str or "multiprocessing" in cmd_str):
                        print(f"  [GPU HEALTH] Terminating process {pid} ({proc.info['name']}): {' '.join(cmdline)[:80]}...")
                        proc.kill()
                        killed_any = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        print(f"  [GPU HEALTH] Process scanning failed: {e}")

    # Method 2: Query nvidia-smi as a fallback
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True
        )
        pids = [int(line.strip()) for line in out.strip().split("\n") if line.strip()]
        for pid in pids:
            if pid != current_pid and pid != parent_pid:
                print(f"  [GPU HEALTH] Terminating GPU app process {pid} via nvidia-smi...")
                try:
                    os.kill(pid, 9)
                    killed_any = True
                except Exception as kill_err:
                    print(f"    Could not terminate process {pid}: {kill_err}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  [GPU HEALTH] nvidia-smi query failed: {e}")

    if killed_any:
        print("  [GPU HEALTH] Orphaned processes terminated. Waiting 2 seconds for VRAM release...")
        import time
        time.sleep(2)
    else:
        print("  [GPU HEALTH] No orphaned GPU or vLLM processes detected.")
    print(f"{border}\n")

def generate_step(args, config):
    """Run the synthetic generation step."""
    cleanup_zombie_gpus()
    cmd = [sys.executable, "generate_dataset.py"]
    if args.backend:
        cmd.extend(["--backend", args.backend])
    if args.num_samples:
        cmd.extend(["--num_samples", str(args.num_samples)])
    if args.tensor_parallel_size is not None:
        cmd.extend(["--tensor_parallel_size", str(args.tensor_parallel_size)])
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])
        
    return run_command(cmd, "Synthetic dataset generation")

def clean_step(args, config):
    """Run the cleaning, deduplication, and export step."""
    cmd = [sys.executable, "clean_dataset.py"]
    return run_command(cmd, "Dataset cleaning, deduplication & export")

def main():
    parser = argparse.ArgumentParser(
        description="Sarvam-105B Teacher-Student Synthetic Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  download   Download model configs, tokenizer, and verify meta-load
  generate   Run Distilabel data generation pipeline
  clean      Run cleaning, deduplication, and export
  all        Run the entire pipeline in sequence
"""
    )
    
    # Subcommands
    parser.add_argument("command", choices=["download", "generate", "clean", "all"], help="Command to execute")
    
    # Optional overrides
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model_id", type=str, help="Override Hugging Face Model ID")
    parser.add_argument("--local_dir", type=str, help="Override local download path")
    parser.add_argument("--backend", type=str, choices=["mock", "transformers", "vllm"], help="Override generation backend")
    parser.add_argument("--num_samples", type=int, help="Override number of samples to generate")
    parser.add_argument("--tensor_parallel_size", type=int, help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--quantization", type=str, help="Quantization format (e.g. fp8)")
    
    # Download specific overrides
    parser.add_argument("--config_only", action="store_true", help="Download tokenizer and configuration files only")
    parser.add_argument("--verify_only", action="store_true", help="Skip downloading, run checks only")

    args = parser.parse_args()
    config = load_config(args.config)

    # Redirect HF cache to the configured local storage directory to prevent home disk exhaustion
    local_dir = args.local_dir or config.get("model", {}).get("local_dir", "./models/teacher")
    hf_cache_dir = os.path.abspath(os.path.join(local_dir, ".cache"))
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_MODULES_CACHE"] = os.path.join(hf_cache_dir, "modules")

    success = False
    if args.command == "download":
        success = download_step(args, config)
    elif args.command == "generate":
        success = generate_step(args, config)
    elif args.command == "clean":
        success = clean_step(args, config)
    elif args.command == "all":
        print("=== Running Full Pipeline ===")
        # 1. Download & Verify (defaulting to config_only if running locally for safety)
        if not args.verify_only and not args.model_id:
            # If running all without specific model overrides, let's enforce config_only to verify architecture
            # unless a real backend is selected or the user specifies otherwise
            backend = args.backend or config.get("generation", {}).get("backend", "mock")
            if backend == "mock":
                args.config_only = True
        
        if download_step(args, config):
            # 2. Generate
            if generate_step(args, config):
                # 3. Clean & Export
                if clean_step(args, config):
                    print("\n🎉 Full pipeline executed successfully! Cleaned dataset artifacts are ready.")
                    success = True
                    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
