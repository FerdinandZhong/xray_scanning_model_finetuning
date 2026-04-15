#!/usr/bin/env python3
"""
Logprob calibration experiment E_cal (CAI wrapper).

Validates that VLM token logprobs are a useful uncertainty signal.
Plots entropy vs accuracy, computes ECE. Target: ECE < 0.15.

Environment Variables:
- MODEL_PATH:  Path to merged VLM model (default: /home/cdsw/models/qwen3vl-2b-pgrav-merged)
- TEST_DATA:   Path to test JSONL (default: data/pgrav_vqa/pgrav_val.jsonl)
- OUTPUT_DIR:  Path for results (default: /home/cdsw/test_results/calibration)
- NUM_SAMPLES: Number of samples to evaluate (default: 500)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, get_venv_python, run_in_venv, check_gpu, validate_script_exists


def main():
    print("=" * 60)
    print("Logprob Calibration Experiment (E_cal)")
    print("=" * 60)

    model_path = os.getenv("MODEL_PATH", "/home/cdsw/models/qwen3vl-2b-pgrav-merged")
    test_data = os.getenv("TEST_DATA", "data/pgrav_vqa/pgrav_val.jsonl")
    output_dir = os.getenv("OUTPUT_DIR", "/home/cdsw/test_results/calibration")
    num_samples = os.getenv("NUM_SAMPLES", "500")

    print(f"  Model:       {model_path}")
    print(f"  Test data:   {test_data}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Samples:     {num_samples}")
    print()

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("   Ensure Phase B training and merge_lora have completed")
        sys.exit(1)

    check_gpu()
    script = validate_script_exists("evaluation/calibration_analysis.py")
    venv_python = get_venv_python()

    cmd = [
        venv_python, "-u", str(script),
        "--model-path", model_path,
        "--test-data", test_data,
        "--output-dir", output_dir,
        "--num-samples", num_samples,
    ]
    run_in_venv(cmd)

    output_path = Path(output_dir)
    if output_path.exists():
        result_files = list(output_path.glob("*"))
        print(f"  Results saved to: {output_path}")
        for f in result_files:
            print(f"    {f.name}")
    print("Calibration experiment completed")


if __name__ == "__main__":
    main()
