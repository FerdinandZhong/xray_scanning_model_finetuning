#!/usr/bin/env python3
"""
Split VQA dataset into train/val/test sets.
Useful after generating VQA for a random sample of images.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def load_vqa_dataset(input_file: Path) -> List[Dict]:
    """Load VQA dataset from JSONL file."""
    data = []
    with open(input_file) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def write_vqa_dataset(data: List[Dict], output_file: Path):
    """Write VQA dataset to JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple:
    """
    Split dataset into train and validation sets.
    
    Args:
        data: List of VQA samples
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.2)
        random_seed: Random seed for reproducibility
    
    Returns:
        (train_data, val_data)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio - 1.0) > 0.01:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must equal 1.0")
    
    # Set random seed
    random.seed(random_seed)
    
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    total = len(shuffled_data)
    train_size = int(total * train_ratio)
    
    # Split
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:]
    
    return train_data, val_data


def count_question_types(dataset: List[Dict]) -> Dict[str, int]:
    """Count question types in dataset."""
    types = {}
    for item in dataset:
        qtype = item.get("metadata", {}).get("question_type", "unknown")
        types[qtype] = types.get(qtype, 0) + 1
    return types


def print_statistics(data: List[Dict], split_name: str):
    """Print statistics for a dataset split."""
    print(f"\n{split_name} Statistics:")
    print(f"  Total samples: {len(data)}")
    
    # Count unique images
    unique_images = len(set(item.get("image_id") or item.get("image_path") for item in data))
    print(f"  Unique images: {unique_images}")
    
    # Question types
    types = count_question_types(data)
    print(f"  Question types:")
    for qtype, count in sorted(types.items()):
        print(f"    - {qtype}: {count} ({count/len(data)*100:.1f}%)")
    
    # Check for structured questions
    structured_count = types.get("structured_list", 0)
    if structured_count > 0:
        print(f"  Structured JSON: {structured_count} ({structured_count/len(data)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Split VQA dataset into train/val sets"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input VQA JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for split datasets (default: data)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="stcray_vqa_sampled",
        help="Prefix for output files (default: stcray_vqa_sampled)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Load dataset
    print("=" * 60)
    print("VQA Dataset Splitter")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Loading VQA dataset...")
    
    data = load_vqa_dataset(input_path)
    print(f"✓ Loaded {len(data)} VQA samples")
    
    # Split dataset
    print(f"\nSplitting dataset...")
    print(f"  Train ratio: {args.train_ratio} ({args.train_ratio*100:.0f}%)")
    print(f"  Val ratio: {args.val_ratio} ({args.val_ratio*100:.0f}%)")
    print(f"  Random seed: {args.random_seed}")
    
    train_data, val_data = split_dataset(
        data,
        args.train_ratio,
        args.val_ratio,
        args.random_seed
    )
    
    # Save splits
    output_dir = Path(args.output_dir)
    train_file = output_dir / f"{args.output_prefix}_train.jsonl"
    val_file = output_dir / f"{args.output_prefix}_val.jsonl"
    
    print(f"\nSaving splits...")
    write_vqa_dataset(train_data, train_file)
    write_vqa_dataset(val_data, val_file)
    
    print(f"✓ Training set: {len(train_data)} samples → {train_file}")
    print(f"✓ Validation set: {len(val_data)} samples → {val_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Split Summary")
    print("="*60)
    print(f"Total samples: {len(data)}")
    print(f"Training: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"Validation: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"Random seed: {args.random_seed}")
    
    # Detailed statistics
    print_statistics(train_data, "Training")
    print_statistics(val_data, "Validation")
    
    # Question type distribution comparison
    print("\n" + "="*60)
    print("Question Type Distribution")
    print("="*60)
    
    train_types = count_question_types(train_data)
    val_types = count_question_types(val_data)
    
    print(f"{'Type':<20} {'Train':<15} {'Val':<15}")
    print("-" * 50)
    all_types = sorted(set(train_types.keys()) | set(val_types.keys()))
    for qtype in all_types:
        train_count = train_types.get(qtype, 0)
        val_count = val_types.get(qtype, 0)
        train_pct = f"{train_count} ({train_count/len(train_data)*100:.1f}%)" if train_data else "0"
        val_pct = f"{val_count} ({val_count/len(val_data)*100:.1f}%)" if val_data else "0"
        print(f"{qtype:<20} {train_pct:<15} {val_pct:<15}")
    
    print("\n✓ Dataset split complete!")
    print(f"\nNext step: Update training config to use these files:")
    print(f"  train_file: {train_file}")
    print(f"  eval_file: {val_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
