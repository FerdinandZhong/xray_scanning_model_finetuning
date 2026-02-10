#!/usr/bin/env python3
"""
Test YOLO X-ray Detection Model with OpenAI-compatible API.

Picks a random image from STCray dataset and queries the YOLO API server.
"""

import os
import json
import random
import argparse
from pathlib import Path
import requests


def get_random_stcray_image(image_dir: str):
    """Get a random X-ray image from STCray dataset."""
    image_path = Path(image_dir)
    
    if not image_path.exists():
        print(f"Error: Image directory not found: {image_dir}")
        exit(1)
    
    # Find all JPG images recursively
    all_images = list(image_path.rglob("*.jpg")) + list(image_path.rglob("*.JPG"))
    
    if not all_images:
        print(f"Error: No images found in {image_dir}")
        exit(1)
    
    # Pick random image
    random_image = random.choice(all_images)
    return random_image


def query_yolo_openai_format(api_base_url: str, image_path: Path, api_key: str = None):
    """
    Query YOLO model using OpenAI-compatible API format.
    
    Uses requests library to send multipart/form-data with image file.
    This mimics how OpenAI clients send files to vision models.
    
    Args:
        api_base_url: Base URL of the API server
        image_path: Path to X-ray image file
        api_key: Optional API key for authentication
    """
    url = f"{api_base_url}/v1/chat/completions"
    
    print(f"\n{'='*70}")
    print(f"Testing YOLO Detection API (OpenAI-compatible)")
    print(f"{'='*70}")
    print(f"API URL: {url}")
    print(f"Image: {image_path}")
    print(f"Image size: {image_path.stat().st_size / 1024:.1f} KB")
    print(f"\nSending request...")
    
    # Prepare headers
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    # Send image as multipart form data
    with open(image_path, 'rb') as f:
        files = {'file': (image_path.name, f, 'image/jpeg')}
        
        try:
            response = requests.post(url, files=files, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Error: Cannot connect to API server at {api_base_url}")
            print(f"   Make sure the server is running:")
            print(f"   For local: ./scripts/serve_yolo_api.sh --model <model_path>")
            print(f"   For remote: Check the endpoint URL and network connectivity")
            exit(1)
        except requests.exceptions.HTTPError as e:
            print(f"\n❌ HTTP Error: {e}")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            exit(1)
    
    # Parse OpenAI-compatible response
    result = response.json()
    
    print(f"\n{'='*70}")
    print(f"OpenAI-Compatible Response")
    print(f"{'='*70}")
    print(f"Response ID: {result.get('id', 'N/A')}")
    print(f"Model: {result.get('model', 'N/A')}")
    print(f"Created: {result.get('created', 'N/A')}")
    
    # Extract detection results from message content
    if result.get('choices') and len(result['choices']) > 0:
        message_content = result['choices'][0]['message']['content']
        
        # Parse JSON from content
        detections = json.loads(message_content)
        
        print(f"\n{'='*70}")
        print(f"Detection Results")
        print(f"{'='*70}")
        print(f"Total detections: {detections['total_count']}")
        print(f"Has concealed items: {detections['has_concealed_items']}")
        
        if detections['items']:
            print(f"\nDetected Items:")
            for i, item in enumerate(detections['items'], 1):
                print(f"\n  {i}. {item['name']}")
                print(f"     Confidence: {item['confidence']:.1%}")
                print(f"     Location: {item['location']}")
                if 'bbox' in item:
                    bbox = item['bbox']
                    print(f"     BBox (normalized): [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")
        else:
            print(f"\n  No threats detected.")
        
        print(f"\n{'='*70}")
        print(f"Raw JSON Output:")
        print(f"{'='*70}")
        print(json.dumps(detections, indent=2))
        
        return detections
        
    else:
        print(f"\n❌ Unexpected response format:")
        print(json.dumps(result, indent=2))
        return None


def check_health(api_base_url: str, api_key: str = None):
    """Check if API server is healthy."""
    url = f"{api_base_url}/health"
    
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        health_data = response.json()
        print(f"✓ API server is healthy")
        print(f"  Model loaded: {health_data.get('model_loaded', False)}")
        print(f"  Backend: {health_data.get('backend', 'unknown')}")
        return True
    except Exception as e:
        print(f"⚠ Warning: Cannot reach API health endpoint: {e}")
        print(f"  Attempting to query anyway...")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test YOLO X-ray Detection API with random STCray image"
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default="https://ml-9132483a-8f3.gr-docpr.a465-9q4k.cloudera.site/namespaces/serving-default/endpoints/rolmocr/openai",
        help='Base URL of the API server (default: Cloudera AI endpoint)'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/stcray_raw/STCray_TestSet/Images',
        help='Directory containing STCray images (default: data/stcray_raw/STCray_TestSet/Images)'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Specific image path (if provided, ignores --image-dir and picks this image)'
    )
    
    args = parser.parse_args()
    
    # Get JWT token from environment
    api_key = os.environ.get("JWT_TOKEN")
    if not api_key:
        print("⚠ Warning: JWT_TOKEN environment variable not set")
        print("  If the API requires authentication, set it with:")
        print("  export JWT_TOKEN=your-token-here")
        api_key = None
    
    print(f"\n{'='*70}")
    print(f"YOLO X-ray Detection Test")
    print(f"{'='*70}")
    print(f"API Base URL: {args.base_url}")
    print(f"Authentication: {'JWT Token' if api_key else 'None'}")
    
    # Check if API server is running
    check_health(args.base_url, api_key)
    
    # Get image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"\n❌ Error: Image not found: {args.image}")
            exit(1)
        print(f"Using specified image: {image_path}")
    else:
        # Try primary image directory
        if not Path(args.image_dir).exists():
            # Fallback to test dataset
            fallback_dir = "data/yolo_dataset_test/images/val"
            if Path(fallback_dir).exists():
                print(f"⚠ Primary image directory not found: {args.image_dir}")
                print(f"  Using fallback: {fallback_dir}")
                args.image_dir = fallback_dir
            else:
                print(f"\n❌ Error: Image directory not found: {args.image_dir}")
                print(f"  Fallback directory also not found: {fallback_dir}")
                print(f"\n  Please specify a valid directory with --image-dir or a specific image with --image")
                exit(1)
        
        image_path = get_random_stcray_image(args.image_dir)
    
    # Query API
    detections = query_yolo_openai_format(args.base_url, image_path, api_key)
    
    if detections:
        print(f"\n{'='*70}")
        print(f"✓ Test completed successfully!")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"✗ Test completed with errors")
        print(f"{'='*70}\n")
        exit(1)


if __name__ == "__main__":
    main()
