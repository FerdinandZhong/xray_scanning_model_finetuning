#!/usr/bin/env python3
"""
Test script for deployed YOLO X-ray Detection API.

Usage:
    python scripts/test_deployed_yolo_api.py path/to/xray_image.jpg
    python scripts/test_deployed_yolo_api.py  # Auto-finds test image
"""

import requests
import json
import sys
from pathlib import Path

BASE_URL = "https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site"

def test_health():
    """Test health endpoint."""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    url = f"{BASE_URL}/health"
    
    try:
        response = requests.get(url, timeout=10)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()
        
        return response.status_code == 200
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        print()
        return False

def test_detect(image_path: str):
    """Test direct detection endpoint."""
    print("=" * 60)
    print("Testing /v1/detect Endpoint")
    print("=" * 60)
    
    url = f"{BASE_URL}/v1/detect"
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    print(f"Image: {image_path}")
    print(f"Sending POST request to {url}...")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": ("xray.jpg", f, "image/jpeg")}
            response = requests.post(url, files=files, timeout=30)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Success!")
            print(f"  Detections: {result['num_detections']}")
            print(f"  Threat: {result['is_threat']}")
            
            if result['threats_detected']:
                print(f"  Threats: {', '.join(result['threats_detected'])}")
            
            if result['detections']:
                print(f"\n  Objects found:")
                for det in result['detections']:
                    bbox = det['bbox']
                    print(f"    - {det['label']}: {det['confidence']:.2%} at ({bbox['x_min']:.0f}, {bbox['y_min']:.0f}, {bbox['x_max']:.0f}, {bbox['y_max']:.0f})")
            else:
                print(f"\n  No objects detected (empty bag or low confidence)")
            
            print()
            return True
        else:
            print(f"❌ Error: {response.text}")
            print()
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        print()
        return False

def test_chat_completions(image_path: str):
    """Test OpenAI-compatible endpoint."""
    print("=" * 60)
    print("Testing /v1/chat/completions Endpoint")
    print("=" * 60)
    
    url = f"{BASE_URL}/v1/chat/completions"
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    print(f"Image: {image_path}")
    print(f"Sending POST request to {url}...")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": ("xray.jpg", f, "image/jpeg")}
            response = requests.post(url, files=files, timeout=30)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            completion = response.json()
            
            print(f"\n✓ Success!")
            print(f"  Model: {completion['model']}")
            print(f"  ID: {completion['id']}")
            print(f"  Created: {completion['created']}")
            
            # Parse detection result from message content
            detection_result = json.loads(completion['choices'][0]['message']['content'])
            
            print(f"\n  Detection Results:")
            print(f"    Detections: {detection_result['num_detections']}")
            print(f"    Threat: {detection_result['is_threat']}")
            
            if detection_result['threats_detected']:
                print(f"    Threats: {', '.join(detection_result['threats_detected'])}")
            
            if detection_result['detections']:
                print(f"\n    Objects:")
                for det in detection_result['detections']:
                    print(f"      - {det['label']}: {det['confidence']:.2%}")
            
            print()
            return True
        else:
            print(f"❌ Error: {response.text}")
            print()
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        print()
        return False

if __name__ == "__main__":
    print()
    print("🚀 Testing YOLO X-ray Detection API")
    print(f"   Base URL: {BASE_URL}")
    print()
    
    # Test health
    health_ok = test_health()
    
    if not health_ok:
        print("❌ Health check failed! Is the application running?")
        print("   Check CAI UI: Applications → xray-yolo-detection-api")
        sys.exit(1)
    
    # Get test image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a test image
        test_paths = [
            "data/luggage_xray_yolo/test/images/xray_001.jpg",
            "data/cargoxray/test/images/P00001.png",
            "test_images/sample_xray.jpg"
        ]
        image_path = None
        for path in test_paths:
            if Path(path).exists():
                image_path = path
                print(f"Auto-detected test image: {image_path}")
                print()
                break
        
        if not image_path:
            print("❌ No test image provided and none found automatically.")
            print()
            print("Usage:")
            print(f"  python {sys.argv[0]} path/to/xray_image.jpg")
            print()
            print("Or place an image at one of these locations:")
            for path in test_paths:
                print(f"  - {path}")
            print()
            sys.exit(1)
    
    # Test both endpoints
    detect_ok = test_detect(image_path)
    chat_ok = test_chat_completions(image_path)
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Health Check:        {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"  /v1/detect:          {'✓ PASS' if detect_ok else '✗ FAIL'}")
    print(f"  /v1/chat/completions: {'✓ PASS' if chat_ok else '✗ FAIL'}")
    print("=" * 60)
    
    if health_ok and detect_ok and chat_ok:
        print("✅ All tests passed!")
        print()
        print("Next steps:")
        print("  - View API docs: https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/docs")
        print("  - Integrate into your application")
        print("  - Train on X-ray dataset for better accuracy")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
