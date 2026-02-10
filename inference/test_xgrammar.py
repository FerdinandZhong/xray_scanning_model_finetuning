#!/usr/bin/env python3
"""
Test script for XGrammar guided generation with vLLM.
Verifies that structured JSON output is generated correctly.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from inference.vllm_engine import create_inference_engine
from inference.json_parser import validate_structured_output


def test_structured_generation(
    model_path: str,
    image_paths: List[str],
    use_vllm: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Test structured JSON generation on sample images.
    
    Args:
        model_path: Path to fine-tuned model
        image_paths: List of test image paths
        use_vllm: Whether to use vLLM (required for XGrammar)
        verbose: Print detailed output
    
    Returns:
        Test results with success rate and examples
    """
    print("=" * 70)
    print("XGrammar Structured Output Test")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Engine: {'vLLM (XGrammar enabled)' if use_vllm else 'Transformers (no XGrammar)'}")
    print(f"Test images: {len(image_paths)}")
    print()
    
    # Initialize engine
    print("Initializing inference engine...")
    try:
        engine = create_inference_engine(
            model_path=model_path,
            use_vllm=use_vllm,
        )
        print("✓ Engine initialized\n")
    except Exception as e:
        print(f"✗ Failed to initialize engine: {e}")
        return {"success": False, "error": str(e)}
    
    # Test each image
    results = []
    valid_json_count = 0
    valid_schema_count = 0
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Testing: {Path(image_path).name}")
        
        try:
            # Generate structured output
            output = engine.generate_structured(
                image_path=image_path,
                prompt="List all items detected in this X-ray scan in JSON format.",
                max_tokens=500,
                temperature=0.7,
            )
            
            # Check if output is valid JSON (should always be with XGrammar)
            if isinstance(output, dict):
                valid_json_count += 1
                print("  ✓ Valid JSON output")
                
                # Validate schema
                is_valid, errors = validate_structured_output(output)
                if is_valid:
                    valid_schema_count += 1
                    print("  ✓ Valid schema")
                    
                    # Extract item count
                    item_count = output.get("total_count", 0)
                    print(f"  ✓ Detected {item_count} items")
                    
                    if verbose:
                        print(f"  Output: {json.dumps(output, indent=2)}")
                else:
                    print(f"  ✗ Schema validation failed:")
                    for error in errors:
                        print(f"    - {error}")
                    
                    if verbose:
                        print(f"  Output: {json.dumps(output, indent=2)}")
            else:
                print(f"  ✗ Not a valid dict (type: {type(output)})")
                if verbose:
                    print(f"  Raw output: {output}")
            
            results.append({
                "image": Path(image_path).name,
                "output": output,
                "valid_json": isinstance(output, dict),
                "valid_schema": is_valid if isinstance(output, dict) else False,
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "image": Path(image_path).name,
                "error": str(e),
                "valid_json": False,
                "valid_schema": False,
            })
        
        print()
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Total images tested: {len(image_paths)}")
    print(f"Valid JSON outputs: {valid_json_count}/{len(image_paths)} ({valid_json_count/len(image_paths)*100:.1f}%)")
    print(f"Valid schema outputs: {valid_schema_count}/{len(image_paths)} ({valid_schema_count/len(image_paths)*100:.1f}%)")
    
    if use_vllm:
        if valid_json_count == len(image_paths):
            print("\n✓ SUCCESS: XGrammar guided generation working correctly!")
            print("  All outputs are valid JSON (100% success rate)")
        else:
            print("\n✗ WARNING: Some outputs are not valid JSON")
            print("  XGrammar should guarantee 100% valid JSON")
            print("  Check vLLM version and configuration")
    else:
        print("\n⚠ Note: Using Transformers engine (no XGrammar)")
        print("  Valid JSON rate may be lower without guided generation")
    
    if valid_schema_count == len(image_paths):
        print("✓ All outputs match the expected schema")
    else:
        print(f"⚠ {len(image_paths) - valid_schema_count} outputs have schema issues")
        print("  Review item names, confidence ranges, and location values")
    
    print()
    
    return {
        "success": True,
        "total": len(image_paths),
        "valid_json": valid_json_count,
        "valid_schema": valid_schema_count,
        "json_rate": valid_json_count / len(image_paths),
        "schema_rate": valid_schema_count / len(image_paths),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test XGrammar structured JSON generation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Test image paths"
    )
    parser.add_argument(
        "--use-transformers",
        action="store_true",
        help="Use Transformers engine (no XGrammar, for comparison)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each test"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Save test results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Verify images exist
    image_paths = []
    for img_path in args.images:
        if Path(img_path).exists():
            image_paths.append(img_path)
        else:
            print(f"Warning: Image not found: {img_path}")
    
    if not image_paths:
        print("Error: No valid image paths provided")
        return 1
    
    # Run test
    use_vllm = not args.use_transformers
    results = test_structured_generation(
        model_path=args.model,
        image_paths=image_paths,
        use_vllm=use_vllm,
        verbose=args.verbose,
    )
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.save_results}")
    
    # Exit with success code
    if results.get("success") and results.get("json_rate", 0) == 1.0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
