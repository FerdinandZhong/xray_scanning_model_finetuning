#!/usr/bin/env python3
"""
Download STCray dataset from HuggingFace.
Dataset: Naoufel555/STCray-Dataset
- 46,642 X-ray images with captions
- 21 threat categories
- Real-world airport scanner data
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm


def download_stcray(
    output_dir: str = "data/stcray",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """
    Download STCray dataset from HuggingFace.
    
    Args:
        output_dir: Output directory for dataset
        cache_dir: HuggingFace cache directory
        max_samples: Maximum samples per split (for testing)
    """
    print("=" * 60)
    print("Downloading STCray Dataset from HuggingFace")
    print("Dataset: Naoufel555/STCray-Dataset")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from HuggingFace
    print("\nLoading dataset from HuggingFace...")
    try:
        dataset = load_dataset(
            "Naoufel555/STCray-Dataset",
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify HuggingFace access (some datasets require authentication)")
        print("3. Try: huggingface-cli login")
        return False
    
    print(f"Dataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Process each split
    for split_name in dataset.keys():
        print(f"\n{'-'*60}")
        print(f"Processing '{split_name}' split...")
        print(f"{'-'*60}")
        
        split_data = dataset[split_name]
        
        # Limit samples if specified
        if max_samples and len(split_data) > max_samples:
            print(f"Limiting to {max_samples} samples (testing mode)")
            split_data = split_data.select(range(max_samples))
        
        # Create output directories
        split_dir = output_path / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Process samples
        annotations = []
        
        for idx, item in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
            # Save image
            image = item['image']
            image_filename = f"{idx:06d}.jpg"
            image_path = images_dir / image_filename
            image.save(image_path)
            
            # Collect annotation information
            # Use relative path for portability
            relative_path = f"{split_name}/images/{image_filename}"
            annotation = {
                "image_id": idx,
                "image_filename": image_filename,
                "image_path": relative_path,  # Relative path for portability
                "image_path_absolute": str(image_path),  # Keep absolute as backup
                "caption": item.get('caption', ''),
                "categories": item.get('categories', []),
                "bboxes": item.get('bboxes', []),
            }
            
            # Add any other fields from the dataset
            for key, value in item.items():
                if key not in ['image', 'caption', 'categories', 'bboxes']:
                    annotation[key] = value
            
            annotations.append(annotation)
        
        # Save annotations to JSON
        ann_file = split_dir / "annotations.json"
        with open(ann_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✓ Saved {len(annotations)} samples")
        print(f"  - Images: {images_dir}")
        print(f"  - Annotations: {ann_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"Dataset location: {output_path}")
    
    # Print structure
    print("\nDataset structure:")
    for split_name in dataset.keys():
        split_dir = output_path / split_name
        if split_dir.exists():
            num_images = len(list((split_dir / "images").glob("*.jpg")))
            print(f"  {split_name}/")
            print(f"    - images/ ({num_images} files)")
            print(f"    - annotations.json")
    
    print("\nNext step: Generate VQA pairs with LLM")
    print("  python data/llm_vqa_generator.py \\")
    print(f"    --annotations {output_path}/train/annotations.json \\")
    print(f"    --images-dir {output_path}/train/images \\")
    print("    --output data/stcray_vqa_train.jsonl")
    
    return True


def verify_dataset(dataset_dir: str):
    """Verify downloaded dataset structure."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_path}")
        return False
    
    print("Verifying dataset structure...")
    
    all_good = True
    for split_name in ["train", "test"]:
        split_dir = dataset_path / split_name
        images_dir = split_dir / "images"
        ann_file = split_dir / "annotations.json"
        
        if not split_dir.exists():
            print(f"  ✗ Missing split: {split_name}")
            all_good = False
            continue
        
        if not images_dir.exists():
            print(f"  ✗ Missing images directory: {images_dir}")
            all_good = False
            continue
        
        if not ann_file.exists():
            print(f"  ✗ Missing annotations file: {ann_file}")
            all_good = False
            continue
        
        # Count images
        num_images = len(list(images_dir.glob("*.jpg")))
        
        # Load annotations
        with open(ann_file) as f:
            annotations = json.load(f)
        
        print(f"  ✓ {split_name}: {num_images} images, {len(annotations)} annotations")
        
        if num_images != len(annotations):
            print(f"    ⚠ Warning: Image count mismatch")
    
    if all_good:
        print("\n✓ Dataset verification passed!")
        return True
    else:
        print("\n✗ Dataset verification failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download STCray dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/stcray",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split (for testing)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing dataset instead of downloading",
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.output_dir)
    else:
        success = download_stcray(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
        )
        
        if success:
            verify_dataset(args.output_dir)


if __name__ == "__main__":
    main()
