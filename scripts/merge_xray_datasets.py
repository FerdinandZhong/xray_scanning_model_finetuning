#!/usr/bin/env python3
"""
Merge multiple X-ray datasets into a single YOLO format dataset.

Combines luggage_xray (threats) + cargoxray (cargo items) for larger training set.
Handles different class sets by creating unified class mapping.
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


def load_dataset_config(yaml_path: Path) -> Dict:
    """Load dataset configuration from data.yaml."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def create_class_mapping(datasets: List[Tuple[str, Dict]]) -> Tuple[List[str], Dict[str, Dict[int, int]]]:
    """
    Create unified class list and mapping from old to new class IDs.
    
    Returns:
        unified_classes: List of all unique class names
        mappings: Dict[dataset_name -> Dict[old_class_id -> new_class_id]]
    """
    all_classes = set()
    
    # Collect all unique classes
    for dataset_name, config in datasets:
        classes = config.get('names', [])
        all_classes.update(classes)
    
    # Create unified sorted class list (threats first for easier interpretation)
    threats = ['blade', 'dagger', 'knife', 'scissors', 'SwissArmyKnife']
    unified_classes = []
    
    # Add threats first
    for threat in threats:
        if threat in all_classes:
            unified_classes.append(threat)
            all_classes.remove(threat)
    
    # Add remaining classes alphabetically
    unified_classes.extend(sorted(all_classes))
    
    # Create mappings for each dataset
    mappings = {}
    for dataset_name, config in datasets:
        old_classes = config.get('names', [])
        mapping = {}
        for old_id, old_class in enumerate(old_classes):
            new_id = unified_classes.index(old_class)
            mapping[old_id] = new_id
        mappings[dataset_name] = mapping
    
    return unified_classes, mappings


def convert_label_file(label_path: Path, class_mapping: Dict[int, int], output_path: Path):
    """Convert label file with new class IDs."""
    if not label_path.exists():
        return
    
    with open(label_path) as f:
        lines = f.readlines()
    
    converted_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            old_class_id = int(parts[0])
            new_class_id = class_mapping.get(old_class_id, old_class_id)
            converted_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(converted_lines)


def merge_datasets(
    dataset_paths: List[Path],
    output_path: Path,
    dataset_names: List[str] = None
):
    """
    Merge multiple YOLO datasets into one.
    
    Args:
        dataset_paths: List of paths to dataset roots (containing data.yaml)
        output_path: Output directory for merged dataset
        dataset_names: Optional names for datasets (for prefix)
    """
    print("="*70)
    print("MERGING X-RAY DATASETS")
    print("="*70)
    
    if dataset_names is None:
        dataset_names = [f"dataset{i}" for i in range(len(dataset_paths))]
    
    # Load all dataset configs
    datasets = []
    for name, path in zip(dataset_names, dataset_paths):
        yaml_path = path / "data.yaml"
        if not yaml_path.exists():
            print(f"⚠️  Warning: {yaml_path} not found, skipping {name}")
            continue
        config = load_dataset_config(yaml_path)
        datasets.append((name, config))
        print(f"✓ Loaded {name}: {len(config.get('names', []))} classes")
    
    if not datasets:
        print("❌ Error: No valid datasets found")
        return 1
    
    # Create unified class mapping
    print("\nCreating unified class mapping...")
    unified_classes, mappings = create_class_mapping(datasets)
    print(f"✓ Unified classes: {len(unified_classes)}")
    print(f"  Classes: {unified_classes}")
    
    # Print mappings
    print("\nClass ID mappings:")
    for dataset_name, mapping in mappings.items():
        dataset_config = next(config for name, config in datasets if name == dataset_name)
        old_classes = dataset_config['names']
        print(f"\n  {dataset_name}:")
        for old_id, new_id in sorted(mapping.items()):
            old_name = old_classes[old_id] if old_id < len(old_classes) else f"class_{old_id}"
            new_name = unified_classes[new_id]
            if old_id != new_id:
                print(f"    {old_name} ({old_id} → {new_id})")
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "valid").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "valid").mkdir(parents=True, exist_ok=True)
    
    # Merge datasets
    print("\nMerging datasets...")
    stats = {
        'train_images': 0,
        'train_labels': 0,
        'val_images': 0,
        'val_labels': 0
    }
    
    for dataset_name, (name, config) in zip(dataset_names, datasets):
        src_path = dataset_paths[dataset_names.index(name)]
        class_mapping = mappings[name]
        
        print(f"\n  Processing {name}...")
        
        # Process train split
        train_img_dir = src_path / config.get('train', 'images/train')
        train_label_dir = src_path / config.get('train', 'images/train').replace('images', 'labels')
        
        if train_img_dir.exists():
            for img_path in train_img_dir.glob("*.jpg"):
                # Copy image with prefix to avoid name conflicts
                new_name = f"{name}_{img_path.name}"
                shutil.copy2(img_path, output_path / "images" / "train" / new_name)
                stats['train_images'] += 1
                
                # Convert and copy label
                label_path = train_label_dir / (img_path.stem + ".txt")
                if label_path.exists():
                    output_label = output_path / "labels" / "train" / (new_name.replace('.jpg', '.txt'))
                    convert_label_file(label_path, class_mapping, output_label)
                    stats['train_labels'] += 1
        
        # Process validation split
        val_img_dir = src_path / config.get('val', 'images/valid')
        val_label_dir = src_path / config.get('val', 'images/valid').replace('images', 'labels')
        
        if val_img_dir.exists():
            for img_path in val_img_dir.glob("*.jpg"):
                new_name = f"{name}_{img_path.name}"
                shutil.copy2(img_path, output_path / "images" / "valid" / new_name)
                stats['val_images'] += 1
                
                label_path = val_label_dir / (img_path.stem + ".txt")
                if label_path.exists():
                    output_label = output_path / "labels" / "valid" / (new_name.replace('.jpg', '.txt'))
                    convert_label_file(label_path, class_mapping, output_label)
                    stats['val_labels'] += 1
        
        print(f"    Train: {stats['train_images']} images, {stats['train_labels']} labels")
        print(f"    Val:   {stats['val_images']} images, {stats['val_labels']} labels")
    
    # Create merged data.yaml
    merged_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'nc': len(unified_classes),
        'names': unified_classes,
        'source_datasets': [name for name, _ in datasets],
        'merged_date': '2026-02-20'
    }
    
    # Add threat categories if any threats exist
    threats_in_unified = [c for c in unified_classes if c in ['blade', 'dagger', 'knife', 'scissors', 'SwissArmyKnife']]
    if threats_in_unified:
        merged_config['threats'] = threats_in_unified
    
    with open(output_path / "data.yaml", 'w') as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
    
    print("\n" + "="*70)
    print("MERGE COMPLETE")
    print("="*70)
    print(f"\nOutput: {output_path}")
    print(f"  Train: {stats['train_images']} images, {stats['train_labels']} labels")
    print(f"  Val:   {stats['val_images']} images, {stats['val_labels']} labels")
    print(f"  Classes: {len(unified_classes)}")
    print(f"  Config: {output_path / 'data.yaml'}")
    print()
    
    # Print warnings
    if len(unified_classes) > 20:
        print("⚠️  WARNING: Large number of classes ({}) may reduce accuracy".format(len(unified_classes)))
        print("   Consider keeping datasets separate for task-specific models")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple X-ray YOLO datasets"
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['luggage_xray_yolo', 'cargoxray_yolo'],
        help='Dataset names (must have data.yaml in data/{name}/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/merged_xray_yolo',
        help='Output directory for merged dataset'
    )
    
    args, _ = parser.parse_known_args()
    
    # Convert dataset names to paths
    project_root = Path(__file__).parent.parent
    dataset_paths = [project_root / "data" / name for name in args.datasets]
    output_path = project_root / args.output
    
    # Verify datasets exist
    for name, path in zip(args.datasets, dataset_paths):
        if not (path / "data.yaml").exists():
            print(f"❌ Error: Dataset not found: {path / 'data.yaml'}")
            return 1
    
    print(f"\nDatasets to merge:")
    for name, path in zip(args.datasets, dataset_paths):
        config = load_dataset_config(path / "data.yaml")
        print(f"  - {name}: {config.get('nc', 0)} classes")
    
    print(f"\nOutput: {output_path}")
    print()
    
    # Ask for confirmation
    response = input("Proceed with merge? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled")
        return 0
    
    # Merge
    return merge_datasets(dataset_paths, output_path, args.datasets)


if __name__ == '__main__':
    exit(main())
