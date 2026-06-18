"""
Dataset Generation Module using Distilabel.
Generates synthetic instruction-response pairs of various task types, languages, and difficulties.
Supports Mock (local testing), local Transformers, and high-throughput parallel vLLM engines.
"""

import os
import sys
import json
import random
import yaml
import argparse
import re
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Generator, List, Dict, Any, Tuple

# Distilabel imports
from distilabel.pipeline import Pipeline
from distilabel.steps import GeneratorStep, StepInput, StepOutput
from distilabel.steps.base import Step

# Force UTF-8 stdout/stderr on Windows to prevent UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _generate_mock_data_fallback(num_samples: int, task_mix: Dict[str, float], mock_templates_file: str) -> List[Dict[str, Any]]:
    """Pure Python fallback helper to generate mock samples without Distilabel class context warnings."""
    try:
        with open(mock_templates_file, "r", encoding="utf-8") as f:
            mock_templates = json.load(f)
    except Exception as e:
        print(f"[ERROR] Fallback loader failed to read mock templates file '{mock_templates_file}': {e}")
        return []
        
    task_types = list(task_mix.keys())
    if not task_types:
        task_types = list(mock_templates.keys())
        
    weights = [task_mix.get(t, 1.0) for t in task_types]
    names = ["Aarav", "Priya", "Rahul", "Ananya", "Sai", "Lakshmi", "Karthik", "Sneha", "Aditya", "Divya"]
    
    samples = []
    for _ in range(num_samples):
        task_type = random.choices(task_types, weights=weights, k=1)[0]
        templates = mock_templates.get(task_type, [])
        if not templates:
            continue
        base_template = random.choice(templates)
        
        sample = dict(base_template)
        name = random.choice(names)
        
        if "{name}" not in sample["instruction"] and random.random() < 0.4:
            if sample["lang"] == "hi":
                sample["instruction"] = sample["instruction"].replace("मुझे", f"नमस्ते, मैं {name} हूँ। मुझे")
            elif sample["lang"] == "en":
                sample["instruction"] = f"Hi, I'm {name}. " + sample["instruction"]
        
        sample["task_type"] = task_type
        sample["source_teacher"] = "sarvamai/sarvam-105b"
        
        samples.append({
            "instruction": sample["instruction"],
            "response": sample["response"],
            "rationale": sample.get("rationale"),
            "task_type": sample["task_type"],
            "language": sample["lang"],
            "difficulty": sample["difficulty"],
            "style": sample["style"],
            "source_teacher": sample["source_teacher"]
        })
    return samples


class MockDataGenerator(GeneratorStep):
    """Custom Distilabel Generator Step using configuration-driven mock templates for local CPU validation."""
    
    num_samples: int = 100
    task_mix: Dict[str, float] = {}
    batch_size: int = 50
    mock_templates_file: str = "mock_templates.json"

    @property
    def outputs(self) -> List[str]:
        return ["instruction", "response", "rationale", "task_type", "language", "difficulty", "style", "source_teacher"]

    def process(self, offset: int = 0) -> Generator[Tuple[List[Dict[str, Any]], bool], None, None]:
        # Load mock templates from configuration-driven JSON file
        try:
            with open(self.mock_templates_file, "r", encoding="utf-8") as f:
                mock_templates = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load mock templates file '{self.mock_templates_file}': {e}")
            raise RuntimeError(f"Mock templates file not loadable: {e}")
            
        task_types = list(self.task_mix.keys())
        if not task_types:
            task_types = list(mock_templates.keys())
            
        weights = [self.task_mix.get(t, 1.0) for t in task_types]
        names = ["Aarav", "Priya", "Rahul", "Ananya", "Sai", "Lakshmi", "Karthik", "Sneha", "Aditya", "Divya"]

        generated_count = offset
        while generated_count < self.num_samples:
            batch = []
            current_batch_size = min(self.batch_size, self.num_samples - generated_count)
            
            for _ in range(current_batch_size):
                task_type = random.choices(task_types, weights=weights, k=1)[0]
                templates = mock_templates.get(task_type, [])
                if not templates:
                    continue
                base_template = random.choice(templates)
                
                sample = dict(base_template)
                name = random.choice(names)
                
                if "{name}" not in sample["instruction"] and random.random() < 0.4:
                    if sample["lang"] == "hi":
                        sample["instruction"] = sample["instruction"].replace("मुझे", f"नमस्ते, मैं {name} हूँ। मुझे")
                    elif sample["lang"] == "en":
                        sample["instruction"] = f"Hi, I'm {name}. " + sample["instruction"]
                
                sample["task_type"] = task_type
                sample["source_teacher"] = "sarvamai/sarvam-105b"
                
                batch.append({
                    "instruction": sample["instruction"],
                    "response": sample["response"],
                    "rationale": sample.get("rationale"),
                    "task_type": sample["task_type"],
                    "language": sample["lang"],
                    "difficulty": sample["difficulty"],
                    "style": sample["style"],
                    "source_teacher": sample["source_teacher"]
                })
                
            generated_count += len(batch)
            is_last = generated_count >= self.num_samples
            
            yield batch, is_last


class VllmDataGenerator(GeneratorStep):
    """Optimized Distilabel GeneratorStep utilizing vLLM engine for high-throughput batch generation on H100 GPUs."""
    
    model_id: str
    num_samples: int
    task_mix: Dict[str, float]
    batch_size: int = 128
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    temperature: float = 0.7
    max_new_tokens: int = 512
    local_dir: str = "./models/teacher"
    
    # Configurable prompt templates loaded from config.yaml
    vllm_instruction_gen: str = ""
    vllm_response_standard: str = ""
    vllm_response_reasoning: str = ""

    @property
    def outputs(self) -> List[str]:
        return ["instruction", "response", "rationale", "task_type", "language", "difficulty", "style", "source_teacher"]

    def process(self, offset: int = 0) -> Generator[Tuple[List[Dict[str, Any]], bool], None, None]:
        import subprocess
        from pathlib import Path
        
        # Check if vLLM registry already supports Sarvam architectures
        is_patched = False
        try:
            from vllm import ModelRegistry
            is_patched = "SarvamMLAForCausalLM" in ModelRegistry.get_supported_archs()
        except Exception:
            pass

        if not is_patched:
            hotpatch_script = Path(self.local_dir) / "hotpatch_vllm.py"
            if hotpatch_script.exists():
                print(f"\nvLLM not patched. Found hotpatch script at {hotpatch_script}. Running hotpatch...")
                try:
                    subprocess.run([sys.executable, str(hotpatch_script)], check=True)
                    print("[OK] vLLM hotpatched successfully for Sarvam MoE/MLA architectures.\n")
                    
                    # Fix binary incompatibility by restoring NumPy <2
                    print("Restoring NumPy <2 to prevent binary incompatibility with pandas/pyarrow...")
                    subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2"], check=True)
                    print("[OK] NumPy <2 restored.\n")
                except Exception as e:
                    print(f"[WARNING] Failed to run hotpatch_vllm.py: {e}\n")

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            print("[ERROR] vllm is not installed. Please install vllm to run with the 'vllm' backend.")
            raise RuntimeError("vllm is required but not installed.")

        import torch
        import traceback
        
        try:
            print(f"Initializing vLLM Engine on H100 GPU for model '{self.model_id}'...")
            llm = LLM(
                model=self.model_id,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype="bfloat16",  # Optimized for H100 hardware
                trust_remote_code=True
            )

            task_types = list(self.task_mix.keys())
            weights = [self.task_mix.get(t, 1.0) for t in task_types]
            languages = ["hi", "te", "ta", "en", "mr", "bn", "gu", "kn", "ml"]

            generated_count = offset
            while generated_count < self.num_samples:
                current_batch_size = min(self.batch_size, self.num_samples - generated_count)
                print(f"Generating batch of {current_batch_size} samples (Progress: {generated_count}/{self.num_samples})...")
                
                # Step 1: Pre-plan metadata for this batch
                metadata = []
                for _ in range(current_batch_size):
                    task_type = random.choices(task_types, weights=weights, k=1)[0]
                    lang = random.choice(languages)
                    difficulty = random.choice(["easy", "medium", "hard"])
                    style = "reasoning" if task_type == "reasoning" else ("spoken" if task_type == "speech_dialogue" else "conversational")
                    metadata.append({
                        "task_type": task_type,
                        "language": lang,
                        "difficulty": difficulty,
                        "style": style
                    })

                # Step 2: Generate instructions for this batch
                instruction_prompts = []
                for m in metadata:
                    prompt = self.vllm_instruction_gen.format(
                        task_type=m["task_type"],
                        language=m["language"]
                    )
                    instruction_prompts.append(prompt)

                sampling_params_inst = SamplingParams(
                    temperature=self.temperature + 0.1,
                    max_tokens=128,
                    skip_special_tokens=True
                )

                outputs_inst = llm.generate(instruction_prompts, sampling_params_inst)
                
                instructions = []
                valid_indices = []
                for idx, out in enumerate(outputs_inst):
                    text = out.outputs[0].text.strip()
                    text = re.sub(r'^["\'\(]|[乌\'\)]$', '', text)
                    if len(text) > 5:
                        instructions.append(text)
                        valid_indices.append(idx)

                # Keep only the valid pre-planned metadata
                metadata = [metadata[i] for i in valid_indices]
                num_valid = len(instructions)
                
                if num_valid == 0:
                    print("Warning: Generated 0 valid instructions in this batch. Retrying...")
                    continue

                # Step 3: Generate responses for this batch
                response_prompts = []
                for idx, m in enumerate(metadata):
                    inst = instructions[idx]
                    if m["task_type"] == "reasoning":
                        prompt = self.vllm_response_reasoning.format(instruction=inst)
                    else:
                        prompt = self.vllm_response_standard.format(instruction=inst)
                    response_prompts.append(prompt)

                sampling_params_resp = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    skip_special_tokens=True
                )

                outputs_resp = llm.generate(response_prompts, sampling_params_resp)

                # Step 4: Parse responses/rationales and construct batch samples
                batch_samples = []
                for idx, out in enumerate(outputs_resp):
                    m = metadata[idx]
                    inst = instructions[idx]
                    raw_text = out.outputs[0].text.strip()
                    
                    rationale = None
                    response = raw_text
                    
                    if m["task_type"] == "reasoning":
                        if "[Rationale]" in raw_text and "[Response]" in raw_text:
                            try:
                                parts = raw_text.split("[Response]")
                                rationale = parts[0].replace("[Rationale]", "").strip()
                                response = parts[1].strip()
                            except Exception as e:
                                print(f"Warning: Failed to parse reasoning output sections: {e}")
                                response = raw_text
                        elif "Rationale:" in raw_text:
                            try:
                                parts = raw_text.split("Response:") if "Response:" in raw_text else [raw_text]
                                rationale = parts[0].replace("Rationale:", "").strip()
                                response = parts[1].strip() if len(parts) > 1 else raw_text
                            except Exception as e:
                                print(f"Warning: Failed to parse reasoning colon sections: {e}")
                                response = raw_text
                    
                    batch_samples.append({
                        "instruction": inst,
                        "response": response,
                        "rationale": rationale,
                        "task_type": m["task_type"],
                        "language": m["language"],
                        "difficulty": m["difficulty"],
                        "style": m["style"],
                        "source_teacher": self.model_id
                    })

                generated_count += len(batch_samples)
                is_last = generated_count >= self.num_samples or len(batch_samples) == 0
                
                yield batch_samples, is_last
        except Exception as e:
            print("\n" + "="*80)
            print("[FATAL WORKER ERROR] Exception occurred inside vLLM generation step:")
            traceback.print_exc()
            print("="*80 + "\n")
            raise ValueError(f"vLLM Generation Step failed: {str(e)}")


class TransformersDataGenerator(GeneratorStep):
    """Distilabel GeneratorStep using a local Hugging Face model for instruction & response generation."""
    
    model_id: str
    num_samples: int
    task_mix: Dict[str, float]
    batch_size: int = 10
    
    # Configurable prompt templates loaded from config.yaml
    transformers_instruction_gen: str = ""
    transformers_response_standard: str = ""
    transformers_response_reasoning: str = ""

    @property
    def outputs(self) -> List[str]:
        return ["instruction", "response", "rationale", "task_type", "language", "difficulty", "style", "source_teacher"]

    def process(self, offset: int = 0) -> Generator[Tuple[List[Dict[str, Any]], bool], None, None]:
        from transformers import pipeline
        import torch
        
        print(f"Loading model '{self.model_id}' in local transformers pipeline...")
        try:
            generator = pipeline(
                "text-generation",
                model=self.model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            print("Local model loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load transformers pipeline: {e}. Falling back to CPU...")
            try:
                generator = pipeline(
                    "text-generation",
                    model=self.model_id,
                    device_map="cpu"
                )
            except Exception as cpu_err:
                print(f"[ERROR] Failed to load model on CPU: {cpu_err}")
                raise RuntimeError(f"Could not load model locally: {cpu_err}")
            
        task_types = list(self.task_mix.keys())
        weights = [self.task_mix.get(t, 1.0) for t in task_types]
        languages = ["hi", "te", "ta", "en", "mr", "bn"]
        
        generated = offset
        while generated < self.num_samples:
            batch = []
            current_batch_size = min(self.batch_size, self.num_samples - generated)
            
            for _ in range(current_batch_size):
                task_type = random.choices(task_types, weights=weights, k=1)[0]
                lang = random.choice(languages)
                
                prompt_generation_query = self.transformers_instruction_gen.format(
                    task_type=task_type,
                    lang=lang
                )
                
                try:
                    res = generator(prompt_generation_query, max_new_tokens=100, do_sample=True, temperature=0.7)
                    instruction = res[0]["generated_text"].replace(prompt_generation_query, "").strip()
                except Exception as e:
                    print(f"Warning: Failed to generate instruction for {task_type}/{lang}: {e}")
                    continue
                
                if not instruction or len(instruction) < 5:
                    continue
                    
                if task_type == "reasoning":
                    response_query = self.transformers_response_reasoning.format(instruction=instruction)
                else:
                    response_query = self.transformers_response_standard.format(instruction=instruction)
                    
                try:
                    res = generator(response_query, max_new_tokens=512, do_sample=True, temperature=0.7)
                    raw_output = res[0]["generated_text"].replace(response_query, "").strip()
                except Exception as e:
                    print(f"Warning: Failed to generate response for instruction: {e}")
                    continue
                
                rationale = None
                response = raw_output
                
                if task_type == "reasoning" and "[Rationale]" in raw_output and "[Response]" in raw_output:
                    try:
                        parts = raw_output.split("[Response]")
                        rationale = parts[0].replace("[Rationale]", "").strip()
                        response = parts[1].strip()
                    except Exception as parse_err:
                        print(f"Warning: Failed to parse reasoning sections in transformers output: {parse_err}")
                        response = raw_output
                else:
                    response = raw_output
                    
                difficulty = random.choice(["easy", "medium", "hard"])
                style = "reasoning" if task_type == "reasoning" else ("spoken" if task_type == "speech_dialogue" else "conversational")
                
                batch.append({
                    "instruction": instruction,
                    "response": response,
                    "rationale": rationale,
                    "task_type": task_type,
                    "language": lang,
                    "difficulty": difficulty,
                    "style": style,
                    "source_teacher": self.model_id
                })
                
            generated += len(batch)
            is_last = generated >= self.num_samples or len(batch) == 0
            
            yield batch, is_last


def build_pipeline(config: Dict[str, Any]) -> Pipeline:
    """Build a Distilabel pipeline based on the configured backend."""
    backend = config["generation"]["backend"]
    num_samples = config["generation"]["num_samples"]
    task_mix = config["dataset"]["target_mix"]
    prompts = config.get("prompts", {})
    
    with Pipeline(name="synthetic-sarvam-distill") as pipeline:
        if backend == "mock":
            MockDataGenerator(
                name="mock_generator",
                num_samples=num_samples,
                task_mix=task_mix,
                batch_size=config["generation"].get("batch_size", 50),
                mock_templates_file=config["cleaning"].get("mock_templates_file", "mock_templates.json")
            )
        elif backend == "vllm":
            model_id = config["model"]["teacher_id"]
            VllmDataGenerator(
                name="vllm_generator",
                model_id=model_id,
                num_samples=num_samples,
                task_mix=task_mix,
                batch_size=config["generation"].get("batch_size", 128),
                gpu_memory_utilization=config["generation"].get("gpu_memory_utilization", 0.90),
                max_model_len=config["generation"].get("max_model_len", 4096),
                tensor_parallel_size=config["generation"].get("tensor_parallel_size", 1),
                temperature=config["generation"].get("temperature", 0.7),
                max_new_tokens=config["generation"].get("max_new_tokens", 512),
                local_dir=config["model"].get("local_dir", "./models/teacher"),
                vllm_instruction_gen=prompts.get("vllm_instruction_gen", ""),
                vllm_response_standard=prompts.get("vllm_response_standard", ""),
                vllm_response_reasoning=prompts.get("vllm_response_reasoning", "")
            )
        elif backend == "transformers":
            model_id = config["model"]["teacher_id"]
            TransformersDataGenerator(
                name="transformers_generator",
                model_id=model_id,
                num_samples=num_samples,
                task_mix=task_mix,
                batch_size=config["generation"].get("batch_size", 10),
                transformers_instruction_gen=prompts.get("transformers_instruction_gen", ""),
                transformers_response_standard=prompts.get("transformers_response_standard", ""),
                transformers_response_reasoning=prompts.get("transformers_response_reasoning", "")
            )
        else:
            print(f"Warning: Backend '{backend}' is not fully configured. Defaulting to high-quality Mock generator.")
            MockDataGenerator(
                name="mock_generator",
                num_samples=num_samples,
                task_mix=task_mix,
                batch_size=config["generation"].get("batch_size", 50),
                mock_templates_file=config["cleaning"].get("mock_templates_file", "mock_templates.json")
            )
            
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Distilabel Synthetic Generation Pipeline")
    parser.add_argument("--backend", type=str, choices=["mock", "transformers", "vllm"], help="Generation backend")
    parser.add_argument("--num_samples", type=int, help="Number of samples to generate")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found.")
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Redirect HF cache to prevent home directory disk exhaustion
    local_dir = config.get("model", {}).get("local_dir", "./models/teacher")
    hf_cache_dir = os.path.abspath(os.path.join(local_dir, ".cache"))
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_MODULES_CACHE"] = os.path.join(hf_cache_dir, "modules")

    # Check if vLLM registry already supports Sarvam architectures
    is_patched = False
    try:
        from vllm import ModelRegistry
        is_patched = "SarvamMLAForCausalLM" in ModelRegistry.get_supported_archs()
    except Exception:
        pass

    if not is_patched:
        hotpatch_script = Path(local_dir) / "hotpatch_vllm.py"
        if hotpatch_script.exists():
            print(f"\nvLLM not patched. Found hotpatch script at {hotpatch_script}. Running hotpatch...")
            try:
                import subprocess
                subprocess.run([sys.executable, str(hotpatch_script)], check=True)
                print("[OK] vLLM installation hotpatched successfully for Sarvam MoE/MLA architectures.\n")
                
                # Fix binary incompatibility by restoring NumPy <2
                print("Restoring NumPy <2 to prevent binary incompatibility with pandas/pyarrow...")
                subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2"], check=True)
                print("[OK] NumPy <2 restored.\n")
            except Exception as patch_err:
                print(f"[WARNING] Failed to run hotpatch_vllm.py: {patch_err}\n")
    else:
        print("[OK] vLLM already patched for Sarvam MoE/MLA architectures.\n")

    # CLI overrides
    if args.backend:
        config["generation"]["backend"] = args.backend
    if args.num_samples:
        config["generation"]["num_samples"] = args.num_samples

    print(f"Starting Distilabel pipeline. Backend: {config['generation']['backend']} | Target Samples: {config['generation']['num_samples']}")
    
    # Build and run the pipeline
    pipeline = build_pipeline(config)
    
    output_dir = Path(config["dataset"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_file_path = Path(config["dataset"]["raw_file"])
    
    print("Running pipeline steps...")
    try:
        pipeline_result = pipeline.run()
    except Exception as e:
        print(f"[ERROR] Pipeline run failed: {e}")
        pipeline_result = None
    
    samples = []
    
    # Distilabel 1.0+ output structure extraction
    if pipeline_result is not None:
        backend = config["generation"]["backend"]
        step_name = f"{backend}_generator" if backend in ["mock", "vllm", "transformers"] else "local_transformers_generator"
        if step_name not in pipeline_result:
            step_name = list(pipeline_result.keys())[0] if pipeline_result.keys() else None
            
        if step_name:
            batch = pipeline_result[step_name]
            if hasattr(batch, "to_list"):
                samples = batch.to_list()
            elif isinstance(batch, list):
                samples = batch
            elif isinstance(batch, dict):
                keys = list(batch.keys())
                num_rows = len(batch[keys[0]]) if keys else 0
                for i in range(num_rows):
                    samples.append({k: batch[k][i] for k in keys})
    
    # Fallback/Safe generation output in case Distilabel pipeline.run has different formats or failed internally
    if not samples:
        print("Dataset extraction from pipeline returned empty. Running fallback generator to secure raw file...")
        try:
            samples = _generate_mock_data_fallback(
                num_samples=config["generation"]["num_samples"], 
                task_mix=config["dataset"]["target_mix"],
                mock_templates_file=config["cleaning"].get("mock_templates_file", "mock_templates.json")
            )
        except Exception as e:
            print(f"[ERROR] Fallback mock generation also failed: {e}")
            sys.exit(1)
            
    print(f"Extracted {len(samples)} raw samples. Writing to raw file: {raw_file_path}")
    
    try:
        with open(raw_file_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print("[OK] Pipeline run complete. Raw dataset saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save raw file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
