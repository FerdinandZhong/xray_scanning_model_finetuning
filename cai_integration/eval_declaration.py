#!/usr/bin/env python3
"""
End-to-end declaration matching experiment E7 (CAI wrapper).

Runs the full XScan-Agent pipeline: X-ray image + declaration -> mismatch report.
Measures declaration match F1, undeclared recall, and latency.

Environment Variables:
- MODEL_PATH:        Path to merged VLM model (default: /home/cdsw/models/qwen3vl-2b-pgrav-merged)
- YOLO_MODEL:        Path to class-agnostic YOLO (default: runs/detect/class_agnostic_xray/weights/best.pt)
- CATEGORY_HINTS:    Path to category hints JSON (default: data/category_hints.json)
- DECLARATION_DATA:  Path to declaration benchmark (default: data/declaration_benchmark.json)
- OUTPUT_DIR:        Path for results (default: /home/cdsw/test_results/declaration_matching)
- NUM_SAMPLES:       Number of samples (default: "" = all)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path("/home/cdsw/cai_integration")))
from utils import PROJECT_ROOT, get_venv_python, run_in_venv, check_gpu, validate_script_exists, check_output_exists


def main():
    print("=" * 60)
    print("Declaration Matching Evaluation (E7)")
    print("=" * 60)

    model_path = os.getenv("MODEL_PATH", "/home/cdsw/models/qwen3vl-2b-pgrav-merged")
    yolo_model = os.getenv("YOLO_MODEL", "runs/detect/class_agnostic_xray/weights/best.pt")
    category_hints = os.getenv("CATEGORY_HINTS", "data/category_hints.json")
    declaration_data = os.getenv("DECLARATION_DATA", "data/declaration_benchmark.json")
    output_dir = os.getenv("OUTPUT_DIR", "/home/cdsw/test_results/declaration_matching")
    num_samples = os.getenv("NUM_SAMPLES", "")

    print(f"  VLM model:        {model_path}")
    print(f"  YOLO model:       {yolo_model}")
    print(f"  Category hints:   {category_hints}")
    print(f"  Declaration data: {declaration_data}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Samples:          {num_samples if num_samples else 'all'}")
    print()

    # Validate prerequisites
    if not Path(model_path).exists():
        print(f"Error: VLM model not found: {model_path}")
        sys.exit(1)
    yolo_path = PROJECT_ROOT / yolo_model
    if not check_output_exists(str(yolo_path), min_size_mb=1.0):
        print(f"Error: YOLO model not found: {yolo_path}")
        sys.exit(1)

    check_gpu()
    script = validate_script_exists("evaluation/eval_declaration_matching.py")
    venv_python = get_venv_python()

    cmd = [
        venv_python, "-u", str(script),
        "--model-path", model_path,
        "--yolo-model", yolo_model,
        "--category-hints", category_hints,
        "--declaration-data", declaration_data,
        "--output-dir", output_dir,
    ]
    if num_samples:
        cmd.extend(["--num-samples", num_samples])

    run_in_venv(cmd)

    output_path = Path(output_dir)
    if output_path.exists():
        print(f"  Results saved to: {output_path}")
    print("Declaration matching evaluation completed")


if __name__ == "__main__":
    main()
