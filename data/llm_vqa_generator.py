#!/usr/bin/env python3
"""
LLM-based VQA dataset generator for X-ray images.
Uses GPT-4V or Claude to generate high-quality, diverse VQA pairs.
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm


class LLMVQAGenerator:
    """Generate VQA pairs using vision-capable LLMs."""
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        use_local: bool = False,
        api_base: Optional[str] = None,
    ):
        """
        Initialize LLM VQA generator.
        
        Args:
            model: Model name (claude-3-5-sonnet-20241022, gpt-4o, Qwen/Qwen2.5-VL-7B-Instruct, etc.)
            api_key: API key (or use environment variable)
            use_local: Use local inference (load model weights directly)
            api_base: API base URL for OpenAI-compatible APIs (e.g., vLLM server)
        """
        self.model = model
        self.use_local = use_local
        
        # Determine provider
        if use_local and "qwen" in model.lower():
            # Local Qwen model loading
            self.provider = "local_qwen"
            print("Initializing local Qwen2.5-VL model...")
            self._init_local_qwen(model)
        elif "claude" in model.lower():
            # Anthropic API
            self.provider = "anthropic"
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            print(f"Initialized {self.provider} with model: {model}")
        elif "gemini" in model.lower():
            # Google Gemini API
            self.provider = "gemini"
            import google.generativeai as genai
            
            api_key_value = api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key_value:
                raise ValueError("GOOGLE_API_KEY environment variable or api_key parameter required for Gemini")
            
            genai.configure(api_key=api_key_value)
            
            # Configure custom endpoint if provided (for AI Gateway)
            if api_base:
                print(f"Initialized Gemini with model: {model}")
                print(f"Custom endpoint: {api_base}")
                print(f"Note: Gemini SDK may use default endpoint. For custom endpoints, ensure API key is configured for that endpoint.")
            else:
                print(f"Initialized Gemini with model: {model}")
            
            self.client = genai.GenerativeModel(
                model_name=model,
                generation_config={"temperature": 0.7, "max_output_tokens": 2000}
            )
        else:
            # OpenAI-compatible API (includes GPT, vLLM with Qwen, etc.)
            self.provider = "openai"
            import openai
            
            # Use custom API base if provided (for vLLM, local servers, etc.)
            openai_api_base = api_base or os.getenv("OPENAI_API_BASE")
            openai_api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
            
            self.client = openai.OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            
            if openai_api_base:
                print(f"Initialized OpenAI-compatible API with model: {model}")
                print(f"API Base: {openai_api_base}")
            else:
                print(f"Initialized OpenAI with model: {model}")
    
    def _init_local_qwen(self, model: str):
        """Initialize local Qwen2.5-VL model."""
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        # Map model names to HuggingFace IDs
        model_mapping = {
            "qwen2.5-vl-2b": "Qwen/Qwen2.5-VL-2B-Instruct",
            "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
        }
        
        model_id = model_mapping.get(model.lower(), model)
        
        # Load model and processor
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.qwen_processor = AutoProcessor.from_pretrained(model_id)
        self.device = self.qwen_model.device
        
        print(f"✓ Loaded {model_id} on {self.device}")
    
    def _create_prompt(self, ground_truth: Dict, num_pairs: int) -> str:
        """Create prompt for LLM to generate VQA pairs."""
        categories = ground_truth.get('categories', [])
        bboxes = ground_truth.get('bboxes', [])
        caption = ground_truth.get('caption', '')
        
        # Format categories nicely
        if categories:
            categories_str = ', '.join(categories)
        else:
            categories_str = 'None (clean scan)'
        
        prompt = f"""You are an expert X-ray security analyst. Analyze this X-ray baggage scan image.

Ground Truth Information:
- Threat items present: {categories_str}
- Number of threat items: {len(categories)}
- Number of bounding boxes: {len(bboxes)}
{f"- Caption: {caption}" if caption else ""}

Task: Generate {num_pairs} diverse Question-Answer pairs for training a Visual Question Answering (VQA) model.

Requirements:
1. Focus on ITEM RECOGNITION ONLY
   - DO mention: item names, locations, descriptions
   - DO NOT mention: risk levels, actions, recommendations, "inspection required"

2. Question types (vary them):
   - General: "What items are visible in this X-ray scan?"
   - Specific: "Is there a gun/knife/explosive in this scan?"
   - Location: "Where are the threat items located?"
   - Count: "How many prohibited items are in this scan?"
   - Detailed: "Describe all items visible with their locations"
   - Occlusion: "Are any items concealed or overlapping?"

3. Answer guidelines:
   - Use natural, conversational language
   - Be specific about item names (use ground truth categories)
   - Include spatial information (upper-left, center, lower-right, etc.)
   - Mention if items are partially visible, concealed, or overlapping
   - For clean scans, say "No prohibited items detected"
   - Keep answers focused and concise (2-4 sentences)

4. Accuracy requirements:
   - ONLY mention items present in ground truth
   - Do NOT hallucinate items not in the list
   - Be precise about item names
   - Use location descriptions that match the image

5. Diversity:
   - Vary question phrasing
   - Mix question types
   - Include both positive and negative examples (for specific questions)

Output Format (JSON array only, no other text):
[
  {{
    "question": "What threat items can you identify in this X-ray scan?",
    "answer": "I can see a handgun in the upper-left area and a knife in the center portion of the baggage.",
    "question_type": "general"
  }},
  {{
    "question": "Is there an explosive device visible in this scan?",
    "answer": "No, there is no explosive device in this scan.",
    "question_type": "specific"
  }},
  ...
]

CRITICAL: 
- Output ONLY the JSON array
- No markdown code blocks
- No additional text or explanations
- Ensure valid JSON syntax"""
        
        return prompt
    
    def _call_claude(self, image_data: str, prompt: str) -> str:
        """Call Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }],
        )
        return response.content[0].text
    
    def _call_openai(self, image_data: str, prompt: str) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
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
            max_tokens=2000,
        )
        return response.choices[0].message.content
    
    def _call_gemini(self, image_data: str, prompt: str) -> str:
        """Call Google Gemini API."""
        from PIL import Image
        import io
        import base64
        
        # Decode base64 image to PIL Image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate content with image and text
        response = self.client.generate_content([prompt, image])
        
        return response.text
    
    def _call_local_qwen(self, image_path: str, prompt: str) -> str:
        """Call local Qwen2.5-VL model."""
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare messages in Qwen2.5-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ]
        
        # Prepare for inference
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.qwen_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode
        output_text = self.qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0]
    
    def _parse_response(self, response: str) -> List[Dict]:
        """Parse LLM response into VQA pairs."""
        # Remove markdown code blocks if present
        json_match = re.search(
            r'```(?:json)?\s*(\[.*?\])\s*```',
            response,
            re.DOTALL
        )
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to extract JSON array from response
            json_str = response.strip()
            
            # Remove any leading/trailing text
            array_start = json_str.find('[')
            array_end = json_str.rfind(']')
            
            if array_start != -1 and array_end != -1:
                json_str = json_str[array_start:array_end + 1]
        
        try:
            vqa_pairs = json.loads(json_str)
            
            # Validate structure
            if not isinstance(vqa_pairs, list):
                print(f"Warning: Response is not a list")
                return []
            
            # Ensure required fields
            valid_pairs = []
            for pair in vqa_pairs:
                if isinstance(pair, dict) and "question" in pair and "answer" in pair:
                    valid_pairs.append(pair)
            
            return valid_pairs
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response: {json_str[:200]}...")
            return []
    
    def generate_vqa_pairs(
        self,
        image_path: str,
        ground_truth: Dict,
        num_pairs: int = 3,
        retry: int = 3,
    ) -> List[Dict]:
        """
        Generate VQA pairs for a single image.
        
        Args:
            image_path: Path to X-ray image
            ground_truth: Ground truth annotations
            num_pairs: Number of VQA pairs to generate
            retry: Number of retries on failure
        
        Returns:
            List of VQA pair dictionaries
        """
        # Create prompt
        prompt = self._create_prompt(ground_truth, num_pairs)
        
        # Call LLM with retry
        for attempt in range(retry):
            try:
                if self.provider == "local_qwen":
                    # Local Qwen uses image path directly
                    response = self._call_local_qwen(image_path, prompt)
                else:
                    # API models need base64 encoding
                    try:
                        with open(image_path, 'rb') as f:
                            image_data = base64.b64encode(f.read()).decode('utf-8')
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        return []
                    
                    if self.provider == "anthropic":
                        response = self._call_claude(image_data, prompt)
                    elif self.provider == "gemini":
                        response = self._call_gemini(image_data, prompt)
                    else:
                        response = self._call_openai(image_data, prompt)
                
                # Parse response
                vqa_pairs = self._parse_response(response)
                
                if vqa_pairs:
                    return vqa_pairs
                
                print(f"Retry {attempt + 1}/{retry}: No valid pairs generated")
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{retry}: {e}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        print(f"Failed to generate VQA pairs for {image_path} after {retry} attempts")
        return []


def process_dataset(
    annotations_file: str,
    images_dir: str,
    output_file: str,
    model: str = "claude-3-5-sonnet-20241022",
    samples_per_image: int = 3,
    max_images: Optional[int] = None,
    rate_limit_delay: float = 1.0,
    batch_save: int = 100,
    use_local: bool = False,
    api_base: Optional[str] = None,
):
    """
    Process entire dataset with LLM to generate VQA pairs.
    
    Args:
        annotations_file: Path to annotations JSON
        images_dir: Directory containing images
        output_file: Output JSONL file
        model: LLM model to use
        samples_per_image: Number of VQA pairs per image
        max_images: Maximum images to process (for testing)
        rate_limit_delay: Delay between API calls (seconds)
        batch_save: Save checkpoint every N images
        use_local: Use local inference (load model weights)
        api_base: API base URL for OpenAI-compatible APIs
    """
    print("=" * 60)
    print("LLM-based VQA Dataset Generation")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Mode: {'Local' if use_local else 'API'}")
    if api_base:
        print(f"API Base: {api_base}")
    print(f"Annotations: {annotations_file}")
    print(f"Images directory: {images_dir}")
    print(f"Output: {output_file}")
    print(f"Samples per image: {samples_per_image}")
    
    # Load annotations
    with open(annotations_file) as f:
        annotations = json.load(f)
    
    print(f"\nLoaded {len(annotations)} annotations")
    
    if max_images:
        annotations = annotations[:max_images]
        print(f"Limited to {max_images} images for testing")
    
    # Initialize generator
    generator = LLMVQAGenerator(model=model, use_local=use_local, api_base=api_base)
    
    # Process images
    vqa_dataset = []
    failed_images = []
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint file
    checkpoint_file = output_path.parent / f"{output_path.stem}_checkpoint.jsonl"
    
    # Resume from checkpoint if exists
    start_idx = 0
    if checkpoint_file.exists():
        print(f"\nFound checkpoint file: {checkpoint_file}")
        with open(checkpoint_file) as f:
            for line in f:
                vqa_dataset.append(json.loads(line.strip()))
        
        # Calculate which image to resume from
        if vqa_dataset:
            last_image_id = vqa_dataset[-1]["metadata"]["image_id"]
            start_idx = last_image_id + 1
            print(f"Resuming from image {start_idx}")
    
    # Process each image
    for idx, ann in enumerate(tqdm(
        annotations[start_idx:],
        desc="Generating VQA pairs",
        initial=start_idx,
        total=len(annotations)
    )):
        image_path = ann['image_path']
        
        # Handle absolute/relative paths - try multiple strategies
        image_path_obj = Path(image_path)
        
        # Strategy 1: If path is relative, resolve from annotations file directory
        if not image_path_obj.is_absolute():
            # Relative to dataset root
            dataset_root = Path(annotations_file).parent.parent
            image_path_obj = dataset_root / image_path
        
        # Strategy 2: If doesn't exist, try absolute path from annotation
        if not image_path_obj.exists() and 'image_path_absolute' in ann:
            image_path_obj = Path(ann['image_path_absolute'])
        
        # Strategy 3: Try using images_dir + filename
        if not image_path_obj.exists():
            image_path_obj = Path(images_dir) / image_path_obj.name
        
        # Strategy 4: If image_filename exists in annotation, use that
        if not image_path_obj.exists() and 'image_filename' in ann:
            image_path_obj = Path(images_dir) / ann['image_filename']
        
        # Final check
        if not image_path_obj.exists():
            print(f"  ⚠ Image not found: {image_path_obj}")
            failed_images.append({"image_id": ann['image_id'], "error": "Image file not found"})
            continue
        
        image_path = str(image_path_obj)
        
        try:
            # Generate VQA pairs
            vqa_pairs = generator.generate_vqa_pairs(
                str(image_path),
                ground_truth=ann,
                num_pairs=samples_per_image
            )
            
            # Add to dataset
            for pair in vqa_pairs:
                vqa_dataset.append({
                    "image_path": str(image_path),
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "metadata": {
                        "image_id": ann['image_id'],
                        "question_type": pair.get("question_type", "general"),
                        "categories": ann.get('categories', []),
                        "num_categories": len(ann.get('categories', [])),
                        "generated_by": "llm",
                        "model": model,
                    }
                })
            
            # Save checkpoint periodically
            if (idx + 1) % batch_save == 0:
                with open(checkpoint_file, 'w') as f:
                    for item in vqa_dataset:
                        f.write(json.dumps(item) + '\n')
                print(f"\nCheckpoint saved: {len(vqa_dataset)} pairs")
            
            # Rate limiting (only for API calls)
            if not use_local:
                time.sleep(rate_limit_delay)
            
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            failed_images.append({"image_id": ann['image_id'], "error": str(e)})
            continue
    
    # Save final dataset
    print(f"\n\nSaving final dataset to {output_file}...")
    with open(output_file, 'w') as f:
        for item in vqa_dataset:
            f.write(json.dumps(item) + '\n')
    
    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    # Statistics
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total VQA pairs: {len(vqa_dataset)}")
    print(f"Failed images: {len(failed_images)}")
    print(f"Success rate: {(1 - len(failed_images) / len(annotations)) * 100:.1f}%")
    
    # Question type distribution
    question_types = {}
    for item in vqa_dataset:
        qtype = item["metadata"].get("question_type", "unknown")
        question_types[qtype] = question_types.get(qtype, 0) + 1
    
    print(f"\nQuestion type distribution:")
    for qtype, count in sorted(question_types.items()):
        print(f"  {qtype}: {count} ({count/len(vqa_dataset)*100:.1f}%)")
    
    # Category distribution
    category_counts = {}
    for item in vqa_dataset:
        for cat in item["metadata"].get("categories", []):
            category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nTop 10 categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cat}: {count}")
    
    # Save statistics
    stats = {
        "total_vqa_pairs": len(vqa_dataset),
        "total_images": len(annotations),
        "samples_per_image": samples_per_image,
        "failed_images": len(failed_images),
        "question_types": question_types,
        "category_counts": category_counts,
        "model": model,
    }
    
    stats_file = output_path.parent / f"{output_path.stem}_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Statistics saved to: {stats_file}")
    
    # Save failed images log
    if failed_images:
        failed_file = output_path.parent / f"{output_path.stem}_failed.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_images, f, indent=2)
        print(f"✓ Failed images log: {failed_file}")
    
    print(f"\n✓ Dataset saved to: {output_file}")
    print("\nNext step: Split dataset or start training")


def validate_vqa_quality(vqa_file: str, sample_size: int = 10):
    """
    Validate quality of generated VQA pairs.
    
    Args:
        vqa_file: Path to VQA JSONL file
        sample_size: Number of samples to display
    """
    print("=" * 60)
    print("VQA Quality Validation")
    print("=" * 60)
    
    # Load samples
    samples = []
    with open(vqa_file) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    print(f"\nTotal samples: {len(samples)}")
    
    # Display random samples
    import random
    random_samples = random.sample(samples, min(sample_size, len(samples)))
    
    print(f"\nRandom {len(random_samples)} samples:")
    print("-" * 60)
    
    for i, sample in enumerate(random_samples, 1):
        print(f"\nSample {i}:")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Type: {sample['metadata'].get('question_type', 'unknown')}")
        print(f"Categories: {sample['metadata'].get('categories', [])}")
        print("-" * 60)
    
    # Quality checks
    print("\n" + "=" * 60)
    print("Quality Metrics")
    print("=" * 60)
    
    # Average answer length
    avg_answer_len = sum(len(s['answer']) for s in samples) / len(samples)
    print(f"Average answer length: {avg_answer_len:.1f} characters")
    
    # Question diversity (unique questions)
    unique_questions = len(set(s['question'] for s in samples))
    print(f"Unique questions: {unique_questions} / {len(samples)} ({unique_questions/len(samples)*100:.1f}%)")
    
    # Answer diversity
    unique_answers = len(set(s['answer'] for s in samples))
    print(f"Unique answers: {unique_answers} / {len(samples)} ({unique_answers/len(samples)*100:.1f}%)")
    
    # Check for common issues
    issues = []
    for sample in samples:
        answer = sample['answer'].lower()
        
        # Check for risk assessment (should not be present)
        if any(word in answer for word in ['risk level', 'inspection required', 'recommend']):
            issues.append("Contains risk assessment language")
        
        # Check for very short answers
        if len(sample['answer']) < 15:
            issues.append(f"Very short answer: {sample['answer']}")
    
    if issues:
        print(f"\n⚠ Quality issues found ({len(issues)} samples):")
        for issue in issues[:10]:
            print(f"  - {issue}")
    else:
        print("\n✓ No quality issues detected")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate VQA dataset using LLM (GPT-4V/Claude)"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        help="Path to annotations JSON file (required for generation)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Directory containing images (required for generation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file (for generation) or input file (for validation)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="LLM model to use (claude-3-5-sonnet-20241022, gpt-4o, qwen2.5-vl-2b, qwen2.5-vl-7b)",
    )
    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=3,
        help="Number of VQA pairs per image",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max images to process (for testing)",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=1.0,
        help="Delay between API calls (seconds)",
    )
    parser.add_argument(
        "--batch-save",
        type=int,
        default=100,
        help="Save checkpoint every N images",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing VQA file instead of generating",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test data loading without making API calls",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use local inference (load model weights directly, not via API)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL for OpenAI-compatible APIs (e.g., http://localhost:8000/v1 for vLLM)",
    )
    
    args = parser.parse_args()
    
    if args.validate:
        # Validation mode
        validate_vqa_quality(args.output)
    elif args.dry_run:
        # Dry-run mode - verify data loading
        if not args.annotations or not args.images_dir:
            parser.error("--annotations and --images-dir are required for dry-run")
        
        print("=" * 60)
        print("Dry Run Mode - Verifying Data Loading")
        print("=" * 60)
        
        # Load annotations
        with open(args.annotations) as f:
            annotations = json.load(f)
        
        print(f"✓ Loaded {len(annotations)} annotations")
        
        if args.max_images:
            annotations = annotations[:args.max_images]
        
        # Verify first 5 images
        print(f"\nVerifying first {min(5, len(annotations))} images...")
        for i, ann in enumerate(annotations[:5]):
            image_path = ann['image_path']
            if not Path(image_path).is_absolute():
                image_path = Path(args.images_dir) / Path(image_path).name
            
            if Path(image_path).exists():
                print(f"  ✓ Image {i}: {Path(image_path).name}")
                print(f"    Categories: {ann.get('categories', [])}")
            else:
                print(f"  ✗ Image {i} NOT FOUND: {image_path}")
        
        print("\n✓ Dry run complete. Data looks good!")
        print(f"\nTo generate, run without --dry-run")
    else:
        # Generation mode
        if not args.annotations or not args.images_dir:
            parser.error("--annotations and --images-dir are required for generation")
        
        process_dataset(
            annotations_file=args.annotations,
            images_dir=args.images_dir,
            output_file=args.output,
            model=args.model,
            samples_per_image=args.samples_per_image,
            max_images=args.max_images,
            rate_limit_delay=args.rate_limit_delay,
            batch_save=args.batch_save,
            use_local=args.use_local,
            api_base=args.api_base,
        )


if __name__ == "__main__":
    main()
