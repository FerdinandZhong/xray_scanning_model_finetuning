#!/usr/bin/env python3
"""
Download sample X-ray baggage scanning images from the internet.

This script helps you collect a few simpler X-ray images for testing RolmOCR.
"""

import os
import argparse
from pathlib import Path
import requests
from urllib.parse import urlparse


# Sample X-ray image URLs (publicly available, legal to use for testing)
SAMPLE_XRAY_URLS = [
    # Add your URLs here - these should be public domain or licensed for reuse
    # Example sources:
    # - Government security agency sample images
    # - Academic research datasets (OPIXray, SIXray public samples)
    # - Open image databases
    
    # Placeholder - replace with actual URLs
    # "https://example.com/sample_xray_knife.jpg",
    # "https://example.com/sample_xray_gun.jpg",
]


def download_image(url: str, output_dir: Path, filename: str = None) -> Path:
    """
    Download an image from URL.
    
    Args:
        url: URL of the image
        output_dir: Directory to save the image
        filename: Optional custom filename
    
    Returns:
        Path to downloaded image
    """
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Determine filename
        if filename is None:
            # Extract from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            # Ensure it has an extension
            if not filename or '.' not in filename:
                filename = f"xray_image_{hash(url) % 10000}.jpg"
        
        # Save file
        output_path = output_dir / filename
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  ✓ Saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  ✗ Error downloading {url}: {e}")
        return None


def create_annotation_template(image_path: Path, annotations_dir: Path) -> Path:
    """
    Create an annotation template JSON file for manual annotation.
    
    Args:
        image_path: Path to the image
        annotations_dir: Directory to save annotations
    
    Returns:
        Path to annotation template file
    """
    annotation_template = {
        "image_filename": image_path.name,
        "image_path": str(image_path.relative_to(image_path.parent.parent)),
        "image_path_absolute": str(image_path.absolute()),
        "categories": [],  # TODO: Fill in detected categories (e.g., ["Knife", "Gun"])
        "items": [],  # TODO: Fill in specific items with details
        "notes": "Manual annotation - please fill in categories and items",
        "source": "web_download",
        "annotator": "manual"
    }
    
    # Save annotation template
    annotation_file = annotations_dir / f"{image_path.stem}_annotation.json"
    
    import json
    with open(annotation_file, 'w') as f:
        json.dump(annotation_template, f, indent=2)
    
    print(f"  Created annotation template: {annotation_file}")
    return annotation_file


def main():
    parser = argparse.ArgumentParser(
        description="Download sample X-ray images for testing"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/test_xrays',
        help='Output directory for downloaded images (default: data/test_xrays)'
    )
    parser.add_argument(
        '--urls',
        type=str,
        nargs='+',
        help='List of image URLs to download'
    )
    parser.add_argument(
        '--create-templates',
        action='store_true',
        help='Create annotation template files'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    images_dir = output_dir / 'images'
    annotations_dir = output_dir / 'annotations'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"X-ray Image Download")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    # Get URLs to download
    urls_to_download = args.urls if args.urls else SAMPLE_XRAY_URLS
    
    if not urls_to_download:
        print("No URLs provided!")
        print()
        print("Usage:")
        print("  1. Provide URLs directly:")
        print("     python3 scripts/download_sample_xrays.py --urls https://example.com/xray1.jpg https://example.com/xray2.jpg")
        print()
        print("  2. Or edit this script and add URLs to SAMPLE_XRAY_URLS list")
        print()
        print("Recommended sources:")
        print("  - OPIXray dataset samples (public)")
        print("  - SIXray dataset samples (academic use)")
        print("  - Government security agency sample images")
        print("  - Google Images (filter by 'labeled for reuse')")
        return
    
    # Download images
    downloaded_images = []
    for url in urls_to_download:
        image_path = download_image(url, images_dir)
        if image_path:
            downloaded_images.append(image_path)
    
    print(f"\n✓ Downloaded {len(downloaded_images)} images")
    
    # Create annotation templates
    if args.create_templates:
        print(f"\n{'='*70}")
        print(f"Creating Annotation Templates")
        print(f"{'='*70}\n")
        
        for image_path in downloaded_images:
            create_annotation_template(image_path, annotations_dir)
        
        print(f"\n✓ Created {len(downloaded_images)} annotation templates")
        print(f"\nNext steps:")
        print(f"  1. Open annotation files in {annotations_dir}/")
        print(f"  2. Fill in the 'categories' and 'items' fields")
        print(f"  3. Run test with: python3 test_rolmocr.py --image <image_path> --annotations <annotation_file>")
    
    print(f"\n{'='*70}")
    print(f"Download Complete!")
    print(f"{'='*70}")
    print(f"Images: {images_dir}/")
    print(f"Annotations: {annotations_dir}/")
    print()


if __name__ == "__main__":
    main()
