#!/usr/bin/env python3
"""
Evaluate operational KPIs: inference latency, throughput, GPU memory usage.
Benchmarks model performance for production deployment.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
import statistics

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
from PIL import Image


def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """Load model for benchmarking."""
    print(f"Loading model from {model_path}...")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    return model, processor


def measure_latency(model, processor, image, prompt, num_runs=10, warmup=3):
    """Measure inference latency."""
    # Warmup
    for _ in range(warmup):
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=128)
    
    # Measure
    latencies = []
    for _ in range(num_runs):
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=128)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return latencies


def measure_throughput(model, processor, images, prompts, batch_sizes=[1, 2, 4, 8]):
    """Measure throughput at different batch sizes."""
    throughput_results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(images):
            continue
        
        # Prepare batch
        batch_images = images[:batch_size]
        batch_prompts = prompts[:batch_size]
        
        # Process batch
        try:
            inputs = processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=128)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            throughput = batch_size / elapsed  # images/second
            
            throughput_results[batch_size] = {
                "throughput": throughput,
                "total_time_s": elapsed,
                "time_per_image_ms": (elapsed / batch_size) * 1000,
            }
        
        except Exception as e:
            print(f"Batch size {batch_size} failed: {e}")
            throughput_results[batch_size] = None
    
    return throughput_results


def measure_memory(model):
    """Measure GPU memory usage."""
    if not torch.cuda.is_available():
        return None
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    return {
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
        "max_memory_allocated_gb": max_memory_allocated,
    }


def benchmark_model(
    model,
    processor,
    test_file: str,
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
    num_latency_runs: int = 20,
    max_samples: int = 100,
):
    """Run comprehensive benchmarks."""
    print("Loading test data...")
    
    # Load test samples
    test_data = []
    with open(test_file, "r") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            test_data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Prepare sample data
    sample = test_data[0]
    sample_image = Image.open(sample["image_path"]).convert("RGB")
    sample_prompt = f"Question: {sample['question']}\nAnswer:"
    
    # Prepare multiple samples for throughput
    images = []
    prompts = []
    for item in test_data[:max(batch_sizes)]:
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            images.append(img)
            prompts.append(f"Question: {item['question']}\nAnswer:")
        except:
            continue
    
    results = {}
    
    # 1. Latency measurement
    print("\n" + "=" * 60)
    print("Measuring inference latency...")
    print("=" * 60)
    
    latencies = measure_latency(model, processor, sample_image, sample_prompt, num_latency_runs)
    
    results["latency"] = {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
    }
    
    print(f"Mean latency: {results['latency']['mean_ms']:.2f} ms")
    print(f"Median latency: {results['latency']['median_ms']:.2f} ms")
    print(f"P95 latency: {results['latency']['p95_ms']:.2f} ms")
    print(f"P99 latency: {results['latency']['p99_ms']:.2f} ms")
    
    # 2. Throughput measurement
    print("\n" + "=" * 60)
    print("Measuring throughput at different batch sizes...")
    print("=" * 60)
    
    throughput_results = measure_throughput(model, processor, images, prompts, batch_sizes)
    results["throughput"] = throughput_results
    
    for batch_size, metrics in throughput_results.items():
        if metrics:
            print(f"Batch size {batch_size}: {metrics['throughput']:.2f} images/s "
                  f"({metrics['time_per_image_ms']:.2f} ms/image)")
    
    # 3. Memory usage
    print("\n" + "=" * 60)
    print("Measuring GPU memory usage...")
    print("=" * 60)
    
    memory_stats = measure_memory(model)
    results["memory"] = memory_stats
    
    if memory_stats:
        print(f"Memory allocated: {memory_stats['memory_allocated_gb']:.2f} GB")
        print(f"Memory reserved: {memory_stats['memory_reserved_gb']:.2f} GB")
        print(f"Peak memory: {memory_stats['max_memory_allocated_gb']:.2f} GB")
    
    # 4. Model info
    results["model_info"] = {
        "num_parameters": model.num_parameters(),
        "dtype": str(model.dtype),
        "device": str(model.device),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark model operational performance")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/opixray_vqa_test.jsonl",
        help="Test data file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/operational_benchmarks.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=20,
        help="Number of runs for latency measurement",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max samples to load",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Base model name",
    )
    
    args = parser.parse_args()
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    # Load model
    model, processor = load_model(args.model, args.base_model)
    
    # Run benchmarks
    results = benchmark_model(
        model,
        processor,
        args.test_file,
        batch_sizes,
        args.num_runs,
        args.max_samples,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✓ Benchmark results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
