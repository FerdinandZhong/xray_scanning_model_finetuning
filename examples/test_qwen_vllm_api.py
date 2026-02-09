#!/usr/bin/env python3
"""
Test script to verify Qwen2.5-VL vLLM server is working correctly.
Run this after starting the vLLM server with: bash scripts/start_qwen_vllm_server.sh
"""

import base64
import os
import sys
from pathlib import Path

def test_vllm_server():
    """Test vLLM server with a sample image."""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ Error: openai package not installed")
        print("Install with: pip install openai>=1.12.0")
        return False
    
    # Configuration
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    model = os.getenv("MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    
    print("=" * 60)
    print("Testing Qwen2.5-VL vLLM Server")
    print("=" * 60)
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print()
    
    # Initialize client
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require a key
        base_url=api_base,
    )
    
    # Test 1: List models
    print("Test 1: List available models")
    print("-" * 60)
    try:
        models = client.models.list()
        print(f"✓ Server is running")
        print(f"Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        print()
        print("Make sure vLLM server is running:")
        print("  bash scripts/start_qwen_vllm_server.sh")
        return False
    print()
    
    # Test 2: Generate VQA
    print("Test 2: Generate VQA for sample image")
    print("-" * 60)
    
    # Find a sample image
    sample_images = [
        "data/stcray/train/images/000000.jpg",
        "data/stcray/test/images/000000.jpg",
        "data/opixray/images/P00001.jpg",
    ]
    
    image_path = None
    for path in sample_images:
        if Path(path).exists():
            image_path = path
            break
    
    if not image_path:
        print("⚠ No sample image found. Skipping VQA test.")
        print("Expected image at: data/stcray/train/images/000000.jpg")
        return True
    
    print(f"Using image: {image_path}")
    
    # Encode image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Create prompt
    prompt = """You are an X-ray security analyst. Analyze this X-ray baggage scan image.

Task: Generate 2 Question-Answer pairs for training a VQA model.

Requirements:
1. Focus on ITEM RECOGNITION ONLY (no risk assessment)
2. Question types: general, specific
3. Use natural, conversational language

Output Format (JSON array only):
[
  {"question": "What items are visible in this X-ray scan?", "answer": "...", "question_type": "general"},
  {"question": "Is there a gun in this scan?", "answer": "...", "question_type": "specific"}
]"""
    
    try:
        print("Sending request to vLLM server...")
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ],
            }],
            max_tokens=1024,
            temperature=0.7,
        )
        
        print("✓ Response received!")
        print()
        print("Generated VQA pairs:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print()
        
    except Exception as e:
        print(f"❌ Failed to generate VQA: {e}")
        return False
    
    print()
    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Generate test dataset (100 images):")
    print("     bash scripts/generate_vqa_qwen_api.sh")
    print()
    print("  2. Or use Python script directly:")
    print("     python data/llm_vqa_generator.py \\")
    print("       --annotations data/stcray/train/annotations.json \\")
    print("       --images-dir data/stcray/train/images \\")
    print("       --output data/stcray_vqa_test.jsonl \\")
    print("       --model Qwen/Qwen2.5-VL-7B-Instruct \\")
    print("       --max-images 100")
    
    return True


if __name__ == "__main__":
    success = test_vllm_server()
    sys.exit(0 if success else 1)
