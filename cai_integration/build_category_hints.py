#!/usr/bin/env python3
"""
Build static category hints lookup from STCray + HiXray annotations (CAI wrapper).

Generates a JSON file mapping categories to descriptions, visual cues, and
confusable pairs for use in Phase C inference-time retrieval-as-text.

Environment Variables:
- STCRAY_DIR:      Path to processed STCray data (default: data/stcray_processed)
- HIXRAY_DIR:      Path to processed HiXray data (default: data/hixray_processed)
- OUTPUT_PATH:     Path for category hints JSON (default: data/category_hints.json)
- FORCE_REPROCESS: Force re-generation (default: false)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, get_venv_python, run_in_venv, should_skip, write_done_marker, validate_script_exists


def main():
    print("=" * 60)
    print("Build Category Hints Lookup")
    print("=" * 60)

    stcray_dir = os.getenv("STCRAY_DIR", "data/stcray_processed")
    hixray_dir = os.getenv("HIXRAY_DIR", "data/hixray_processed")
    output_path = os.getenv("OUTPUT_PATH", "data/category_hints.json")

    output_file = PROJECT_ROOT / output_path

    print(f"  STCray dir:  {stcray_dir}")
    print(f"  HiXray dir:  {hixray_dir}")
    print(f"  Output:      {output_path}")
    print()

    if not os.getenv("FORCE_REPROCESS", "false").lower() == "true":
        if output_file.exists():
            import json
            hints = json.loads(output_file.read_text())
            print(f"  Category hints already exist: {len(hints)} entries")
            print("  Skipping. Set FORCE_REPROCESS=true to regenerate.")
            return

    script = validate_script_exists("scripts/build_category_hints.py")
    venv_python = get_venv_python()

    cmd = [
        venv_python, "-u", str(script),
        "--stcray-dir", stcray_dir,
        "--hixray-dir", hixray_dir,
        "--output", output_path,
    ]
    run_in_venv(cmd)

    # Verify
    if output_file.exists():
        import json
        hints = json.loads(output_file.read_text())
        print(f"  Generated {len(hints)} category entries")
        write_done_marker("build_category_hints", str(output_file.parent))
        print("Category hints built successfully")
    else:
        print(f"Error: Output not found: {output_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
