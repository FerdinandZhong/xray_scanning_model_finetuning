#!/usr/bin/env python3
"""
Test pre-trained Qwen2.5-VL-7B on X-ray images to evaluate if fine-tuning is needed.
This helps determine the baseline performance before fine-tuning.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import os

def test_pretrained_model(
    test_images: List[str],
    annotations_file: str,
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    api_base: str = "http://localhost:8000/v1",
    max_samples: int = 100,
):
    """Test pre-trained model on X-ray images."""
    from openai import OpenAI
    
    print("=" * 60)
    print("Testing Pre-trained Qwen2.5-VL-7B")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"API Base: {api_base}")
    print(f"Max samples: {max_samples}")
    print()
    
    # Load annotations
    with open(annotations_file) as f:
        annotations = json.load(f)
    
    # Limit samples
    annotations = annotations[:max_samples]
    
    # Initialize client
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_base,
    )
    
    # Test prompt (simple, generic)
    def create_prompt():
        return """You are an X-ray security analyst. Analyze this X-ray baggage scan image.

Question: What prohibited items (weapons, sharp objects, explosives) can you identify in this scan? 
If you see any, describe them and their locations. If the scan is clean, say "No prohibited items detected."

Answer concisely in 2-3 sentences."""
    
    results = []
    correct = 0
    total = 0
    
    print("Testing on samples...")
    for idx, ann in enumerate(annotations):
        image_path = ann.get('image_path', '')
        if not Path(image_path).exists():
            # Try relative path
            image_path = Path(annotations_file).parent.parent / ann.get('image_path', '')
        
        if not Path(image_path).exists():
            print(f"⚠ Image not found: {image_path}")
            continue
        
        ground_truth_categories = ann.get('categories', [])
        
        # Encode image
        import base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Call model
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": create_prompt()},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ],
                }],
                max_tokens=256,
                temperature=0.3,
            )
            
            answer = response.choices[0].message.content
            
            # Simple evaluation: check if detected categories
            detected = []
            answer_lower = answer.lower()
            for cat in ground_truth_categories:
                # Simple keyword matching
                cat_keywords = cat.lower().replace('_', ' ').split()
                if any(keyword in answer_lower for keyword in cat_keywords):
                    detected.append(cat)
            
            # Calculate accuracy
            if ground_truth_categories:
                # Has items - check if detected any
                if detected or "no prohibited" not in answer_lower:
                    correct += 1
            else:
                # Clean scan - check if said clean
                if "no prohibited" in answer_lower or "clean" in answer_lower:
                    correct += 1
            
            total += 1
            
            results.append({
                "image_id": ann.get('image_id'),
                "ground_truth": ground_truth_categories,
                "detected": detected,
                "answer": answer,
            })
            
            # Print progress
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(annotations)}, Accuracy: {correct/total*100:.1f}%")
        
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue
    
    # Final results
    print()
    print("=" * 60)
    print("Pre-trained Model Evaluation Results")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Simple accuracy: {correct/total*100:.1f}%")
    print()
    
    # Show some examples
    print("Sample outputs:")
    print("-" * 60)
    for i, result in enumerate(results[:5]):
        print(f"\nExample {i+1}:")
        print(f"Ground truth: {result['ground_truth']}")
        print(f"Model answer: {result['answer'][:150]}...")
        print()
    
    # Save results
    output_file = "results/pretrained_evaluation.json"
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "total_samples": total,
            "accuracy": correct/total,
            "results": results
        }, f, indent=2)
    
    print(f"✓ Full results saved to: {output_file}")
    print()
    print("=" * 60)
    print("Interpretation:")
    print("=" * 60)
    accuracy = correct/total*100
    
    if accuracy >= 85:
        print(f"✓ EXCELLENT ({accuracy:.1f}%): Pre-trained model works well!")
        print("  → Fine-tuning may provide marginal improvements")
        print("  → Consider testing more samples before deciding")
    elif accuracy >= 70:
        print(f"⚠ GOOD ({accuracy:.1f}%): Pre-trained model is decent")
        print("  → Fine-tuning will likely provide 10-15% improvement")
        print("  → RECOMMENDED for production use")
    elif accuracy >= 50:
        print(f"⚠ MODERATE ({accuracy:.1f}%): Pre-trained model struggles")
        print("  → Fine-tuning will likely provide 20-30% improvement")
        print("  → STRONGLY RECOMMENDED")
    else:
        print(f"❌ POOR ({accuracy:.1f}%): Pre-trained model insufficient")
        print("  → Fine-tuning is ESSENTIAL for this task")
    
    print()
    print("Next steps:")
    if accuracy < 85:
        print("  1. Generate VQA dataset: bash scripts/generate_vqa_qwen_api.sh")
        print("  2. Fine-tune model: bash scripts/train.sh")
        print("  3. Compare fine-tuned vs pre-trained performance")
    else:
        print("  1. Test on more samples to confirm performance")
        print("  2. Consider fine-tuning if you need >90% accuracy")


def main():
    parser = argparse.ArgumentParser(description="Test pre-trained Qwen2.5-VL on X-ray images")
    parser.add_argument("--annotations", type=str, required=True, help="Annotations JSON file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to test")
    
    args = parser.parse_args()
    
    test_pretrained_model(
        test_images=[],
        annotations_file=args.annotations,
        model=args.model,
        api_base=args.api_base,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
