#!/usr/bin/env python3
"""
Split VQA dataset into train/val/test sets with stratification.
Ensures balanced distribution of prohibited item categories.
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


def load_vqa_dataset(input_file: Path) -> List[Dict[str, Any]]:
    """Load VQA dataset from JSONL file."""
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def stratify_by_categories(data: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """Group data by category presence for stratified splitting."""
    stratified = defaultdict(list)
    
    for item in data:
        metadata = item.get("metadata", {})
        categories = metadata.get("categories", [])
        
        if not categories:
            key = "no_prohibited"
        else:
            # Use sorted tuple of categories as key
            key = tuple(sorted(set(categories)))
        
        stratified[str(key)].append(item)
    
    return stratified


def split_stratified(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Split data with stratification."""
    random.seed(seed)
    
    # Stratify by categories
    stratified = stratify_by_categories(data)
    
    train_data = []
    val_data = []
    test_data = []
    
    print(f"Found {len(stratified)} unique category combinations")
    
    for key, items in stratified.items():
        # Shuffle items in this stratum
        random.shuffle(items)
        
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])
    
    # Final shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


def write_jsonl(data: List[Dict[str, Any]], output_file: Path):
    """Write data to JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def print_statistics(data: List[Dict[str, Any]], split_name: str):
    """Print statistics for a data split."""
    print(f"\n{split_name} Set Statistics:")
    print(f"  Total samples: {len(data)}")
    
    # Count by question type
    question_types = defaultdict(int)
    for item in data:
        qtype = item.get("metadata", {}).get("question_type", "unknown")
        question_types[qtype] += 1
    
    print(f"  Question types:")
    for qtype, count in sorted(question_types.items()):
        print(f"    - {qtype}: {count}")
    
    # Count prohibited items
    with_prohibited = sum(
        1 for item in data
        if item.get("metadata", {}).get("categories", [])
    )
    mismatches = sum(
        1 for item in data
        if not item.get("metadata", {}).get("match_declaration", True)
    )
    
    print(f"  Samples with prohibited items: {with_prohibited}")
    print(f"  Declaration mismatches: {mismatches}")


def main():
    parser = argparse.ArgumentParser(
        description="Split VQA dataset into train/val/test sets"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input VQA JSONL file",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="data/opixray_vqa_train.jsonl",
        help="Output file for training set",
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default="data/opixray_vqa_val.jsonl",
        help="Output file for validation set",
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default="data/opixray_vqa_test.jsonl",
        help="Output file for test set",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        return 1
    
    input_file = Path(args.input)
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please run data preparation scripts first:")
        print("  1. python data/create_vqa_pairs.py")
        print("  2. python data/declaration_simulator.py")
        return 1
    
    print(f"Loading dataset from {input_file}...")
    data = load_vqa_dataset(input_file)
    print(f"Loaded {len(data)} samples")
    
    print(f"\nSplitting with ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    train_data, val_data, test_data = split_stratified(
        data,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    
    # Write splits
    print(f"\nWriting splits...")
    write_jsonl(train_data, Path(args.train_output))
    write_jsonl(val_data, Path(args.val_output))
    write_jsonl(test_data, Path(args.test_output))
    
    # Print statistics
    print_statistics(train_data, "Training")
    print_statistics(val_data, "Validation")
    print_statistics(test_data, "Test")
    
    print(f"\n✓ Dataset split complete!")
    print(f"  - Training: {args.train_output} ({len(train_data)} samples)")
    print(f"  - Validation: {args.val_output} ({len(val_data)} samples)")
    print(f"  - Test: {args.test_output} ({len(test_data)} samples)")
    
    print("\nNext step: python training/train_local.py --config configs/train_local.yaml")


if __name__ == "__main__":
    main()
