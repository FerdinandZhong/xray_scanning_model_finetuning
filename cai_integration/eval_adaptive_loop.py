#!/usr/bin/env python3
"""
Adaptive loop evaluation experiment E_adapt (CAI wrapper).

Compares single-pass pipeline vs uncertainty-gated adaptive re-analysis.
Target: >= 5% recall gain on ambiguous items at fixed precision.

Environment Variables:
- MODEL_PATH:        Path to merged VLM model (default: /home/cdsw/models/qwen3vl-2b-pgrav-merged)
- YOLO_MODEL:        Path to class-agnostic YOLO (default: runs/detect/class_agnostic_xray/weights/best.pt)
- CATEGORY_HINTS:    Path to category hints JSON (default: data/category_hints.json)
- TEST_DATA:         Path to test JSONL (default: data/pgrav_vqa/pgrav_test.jsonl)
- OUTPUT_DIR:        Path for results (default: /home/cdsw/test_results/adaptive_loop)
- ENTROPY_THRESHOLD: Logprob entropy threshold (default: 1.5)
- NUM_SAMPLES:       Number of samples (default: 200)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path("/home/cdsw/cai_integration")))
from utils import PROJECT_ROOT, get_venv_python, run_in_venv, check_gpu, validate_script_exists, check_output_exists


def main():
    print("=" * 60)
    print("Adaptive Loop Evaluation (E_adapt)")
    print("=" * 60)

    model_path = os.getenv("MODEL_PATH", "/home/cdsw/models/qwen3vl-2b-pgrav-merged")
    yolo_model = os.getenv("YOLO_MODEL", "runs/detect/class_agnostic_xray/weights/best.pt")
    category_hints = os.getenv("CATEGORY_HINTS", "data/category_hints.json")
    test_data = os.getenv("TEST_DATA", "data/pgrav_vqa/pgrav_test.jsonl")
    output_dir = os.getenv("OUTPUT_DIR", "/home/cdsw/test_results/adaptive_loop")
    entropy_threshold = os.getenv("ENTROPY_THRESHOLD", "1.5")
    num_samples = os.getenv("NUM_SAMPLES", "200")

    print(f"  VLM model:   {model_path}")
    print(f"  YOLO model:  {yolo_model}")
    print(f"  Hints:       {category_hints}")
    print(f"  Test data:   {test_data}")
    print(f"  Threshold:   {entropy_threshold}")
    print(f"  Samples:     {num_samples}")
    print()

    # Validate prerequisites
    if not Path(model_path).exists():
        print(f"Error: VLM model not found: {model_path}")
        sys.exit(1)
    yolo_path = PROJECT_ROOT / yolo_model
    if not check_output_exists(str(yolo_path), min_size_mb=1.0):
        print(f"Error: YOLO model not found: {yolo_path}")
        sys.exit(1)
    hints_path = PROJECT_ROOT / category_hints
    if not hints_path.exists():
        print(f"Error: Category hints not found: {hints_path}")
        print("   Ensure build_category_hints job has completed")
        sys.exit(1)

    check_gpu()
    script = validate_script_exists("evaluation/eval_adaptive_loop.py")
    venv_python = get_venv_python()

    cmd = [
        venv_python, "-u", str(script),
        "--model-path", model_path,
        "--yolo-model", yolo_model,
        "--category-hints", category_hints,
        "--test-data", test_data,
        "--output-dir", output_dir,
        "--entropy-threshold", entropy_threshold,
        "--num-samples", num_samples,
    ]
    run_in_venv(cmd)

    output_path = Path(output_dir)
    if output_path.exists():
        print(f"  Results saved to: {output_path}")
    print("Adaptive loop evaluation completed")


if __name__ == "__main__":
    main()
