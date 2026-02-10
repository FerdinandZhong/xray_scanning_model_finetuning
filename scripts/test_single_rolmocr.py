#!/usr/bin/env python3
"""
Test RolmOCR on a single image and show detailed output.
"""

import argparse
import base64
import json
import os
from pathlib import Path

from openai import OpenAI
from PIL import Image


def encode_image_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_single_image(
    client: OpenAI,
    image_path: Path,
    model_id: str,
    show_raw: bool = True
):
    """Test RolmOCR on a single image with detailed output."""
    
    print(f"\n{'='*80}")
    print(f"Testing RolmOCR on: {image_path.name}")
    print(f"{'='*80}\n")
    
    # Load and display image info
    img = Image.open(image_path)
    print(f"Image size: {img.size[0]}x{img.size[1]}")
    print(f"Image format: {img.format}")
    print(f"Image mode: {img.mode}\n")
    
    # Encode image
    print("Encoding image to base64...")
    base64_image = encode_image_base64(image_path)
    print(f"Base64 length: {len(base64_image)} characters\n")
    
    # Test 1: Simple query without structured output
    print("="*80)
    print("Test 1: Simple Query (No Structured Output)")
    print("="*80)
    print("\nPrompt: 'What objects do you see in this X-ray image?'\n")
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What objects do you see in this X-ray image? List them."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=2048
        )
        
        content = response.choices[0].message.content
        print("Response:")
        print("-" * 80)
        print(content)
        print("-" * 80)
        
        if show_raw:
            print("\nRaw response object:")
            print(json.dumps(response.model_dump(), indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: With structured output (JSON schema)
    print("\n" + "="*80)
    print("Test 2: Structured Output (JSON Schema)")
    print("="*80)
    print("\nJSON Schema:")
    
    json_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "confidence": {"type": "number"},
                        "location": {"type": "string"}
                    },
                    "required": ["name"]
                }
            },
            "summary": {"type": "string"}
        },
        "required": ["items"]
    }
    
    print(json.dumps(json_schema, indent=2))
    print()
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert X-ray image analyzer. Identify all objects visible in the scan."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this cargo X-ray image and list all visible items. Focus on major objects like textiles, auto parts, tools, toys, shoes, bags, etc."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=2048,
            extra_body={
                "guided_json": json_schema
            }
        )
        
        content = response.choices[0].message.content
        print("Response:")
        print("-" * 80)
        print(content)
        print("-" * 80)
        
        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            print("\nParsed JSON:")
            print(json.dumps(parsed, indent=2))
            
            if "items" in parsed:
                print(f"\nDetected {len(parsed['items'])} items:")
                for i, item in enumerate(parsed['items'], 1):
                    print(f"  {i}. {item.get('name', 'unknown')} "
                          f"(confidence: {item.get('confidence', 'N/A')})")
        except json.JSONDecodeError:
            print("\nWarning: Response is not valid JSON")
        
        if show_raw:
            print("\nRaw response object:")
            print(json.dumps(response.model_dump(), indent=2, default=str))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Different prompt style
    print("\n" + "="*80)
    print("Test 3: Direct Cargo Category Query")
    print("="*80)
    print("\nPrompt: 'What type of cargo is this? Is it textiles, tools, auto parts, toys, or something else?'\n")
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What type of cargo is visible in this X-ray scan? Choose from: textiles, tools, auto parts, toys, shoes, bags, bicycle, car wheels, clothes, fabrics, lamps, office supplies, tableware, spare parts, or unknown. Just give me one or two words."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=50
        )
        
        content = response.choices[0].message.content
        print("Response:")
        print("-" * 80)
        print(content)
        print("-" * 80)
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test RolmOCR on a single image with detailed output"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://ml-9132483a-8f3.gr-docpr.a465-9q4k.cloudera.site/namespaces/serving-default/endpoints/rolmocr/openai/v1",
        help="Base URL for OpenAI-compatible API"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="reducto/RolmOCR",
        help="Model ID"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to image file"
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Show raw API response"
    )
    
    args = parser.parse_args()
    
    # Get JWT token
    jwt_token = os.getenv("JWT_TOKEN")
    if not jwt_token:
        print("Error: JWT_TOKEN environment variable not set")
        return 1
    
    # Validate image
    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        return 1
    
    # Create client
    client = OpenAI(
        base_url=args.base_url,
        api_key=jwt_token
    )
    
    # Test
    test_single_image(client, args.image, args.model_id, args.show_raw)
    
    print(f"\n{'='*80}")
    print("Testing complete!")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
