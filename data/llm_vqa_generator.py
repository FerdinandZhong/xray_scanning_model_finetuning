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
        else:
            # OpenAI-compatible API (includes GPT, Gemini via AI Gateway, vLLM, etc.)
            self.provider = "openai"
            import openai
            
            # Use custom API base if provided (for Gemini AI Gateway, vLLM, local servers, etc.)
            openai_api_base = api_base or os.getenv("OPENAI_API_BASE")
            openai_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "EMPTY")
            
            self.client = openai.OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            
            if openai_api_base:
                print(f"Initialized OpenAI-compatible API with model: {model}")
                print(f"API Base: {openai_api_base}")
                if "gemini" in model.lower():
                    print(f"Using Gemini via OpenAI-compatible AI Gateway")
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
    
    def _calculate_locations(self, bboxes: List[List[int]], image_width: int = 640, image_height: int = 640) -> List[str]:
        """
        Convert bounding boxes to location descriptions.
        
        Args:
            bboxes: List of [x, y, width, height] bounding boxes
            image_width: Image width (default: 640)
            image_height: Image height (default: 640)
        
        Returns:
            List of location strings (e.g., "center-left", "upper-right")
        """
        locations = []
        for bbox in bboxes:
            if len(bbox) < 4:
                locations.append("center")
                continue
                
            x, y, w, h = bbox[:4]
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Determine horizontal position
            if center_x < image_width / 3:
                h_loc = "left"
            elif center_x < 2 * image_width / 3:
                h_loc = "center"
            else:
                h_loc = "right"
            
            # Determine vertical position
            if center_y < image_height / 3:
                v_loc = "upper"
            elif center_y < 2 * image_height / 3:
                v_loc = ""
            else:
                v_loc = "lower"
            
            # Combine location
            if v_loc and h_loc == "center":
                location = v_loc
            elif v_loc:
                location = f"{v_loc}-{h_loc}"
            else:
                location = h_loc
            
            locations.append(location)
        
        return locations
    
    def _create_prompt(self, ground_truth: Dict, num_pairs: int, structured_ratio: float = 0.0) -> str:
        """
        Create prompt for LLM to generate VQA pairs.
        
        Args:
            ground_truth: Ground truth annotations
            num_pairs: Total number of VQA pairs to generate
            structured_ratio: Ratio of structured JSON questions (0.0-1.0)
        """
        categories = ground_truth.get('categories', [])
        bboxes = ground_truth.get('bboxes', [])
        caption = ground_truth.get('caption', '')
        
        # Calculate locations from bboxes
        locations = self._calculate_locations(bboxes) if bboxes else []
        
        # Format categories nicely
        if categories:
            categories_str = ', '.join(categories)
        else:
            categories_str = 'None (clean scan)'
        
        # Calculate number of structured vs natural questions
        # Use probabilistic selection for small num_pairs to ensure we get some structured questions
        import random
        
        if structured_ratio > 0 and num_pairs > 0:
            # For each question, probabilistically decide if it's structured
            structured_mask = [random.random() < structured_ratio for _ in range(num_pairs)]
            num_structured = sum(structured_mask)
            num_natural = num_pairs - num_structured
            
            # Ensure at least 1 structured if ratio > 0.5 and num_pairs >= 2
            if num_structured == 0 and structured_ratio >= 0.5 and num_pairs >= 2:
                num_structured = 1
                num_natural = num_pairs - 1
        else:
            num_structured = 0
            num_natural = num_pairs
        
        # Build structured instructions
        if structured_ratio > 0 and num_structured > 0:
            # Create example JSON for this specific image
            example_items = []
            for i, (cat, loc) in enumerate(zip(categories[:len(locations)], locations)):
                example_items.append({
                    "name": cat.lower().replace('_', ' '),
                    "confidence": 0.90,
                    "location": loc
                })
            
            # Fill remaining if we have more categories than locations
            for i, cat in enumerate(categories[len(locations):]):
                example_items.append({
                    "name": cat.lower().replace('_', ' '),
                    "confidence": 0.85,
                    "location": "center"
                })
            
            example_json = {
                "items": example_items if example_items else [],
                "total_count": len(categories),
                "has_concealed_items": bool(caption and 'conceal' in caption.lower())
            }
            
            structured_instructions = f"""

7. STRUCTURED JSON questions - GENERATE EXACTLY {num_structured} questions with this format:
   
   Question (ALWAYS use this exact phrasing):
   "List all items detected in this X-ray scan in JSON format."
   
   Answer (MUST be VALID JSON string, no extra text):
   '{json.dumps(example_json)}'
   
   question_type: "structured_list"
   
   CRITICAL for structured questions:
   - Answer MUST be pure JSON (no "Here is...", no explanations)
   - Use exact item names from ground truth: {categories_str}
   - Use exact locations: {', '.join(locations) if locations else 'center'}
   - Confidence: 0.85-0.95 (clear items), 0.70-0.84 (occluded)
   - total_count MUST equal items array length: {len(categories)}
   - has_concealed_items: {str(bool(caption and 'conceal' in caption.lower())).lower()}
"""
        else:
            structured_instructions = ""
        
        prompt = f"""You are an expert X-ray security analyst. Analyze this X-ray baggage scan image.

Ground Truth Information:
- Threat items present: {categories_str}
- Number of threat items: {len(categories)}
- Item locations: {', '.join(locations) if locations else 'various'}
{f"- Caption: {caption}" if caption else ""}

Task: Generate {num_pairs} diverse Question-Answer pairs for training a Visual Question Answering (VQA) model.

Requirements:
1. Focus on ITEM RECOGNITION ONLY
   - DO mention: item names, locations, descriptions
   - DO NOT mention: risk levels, actions, recommendations, "inspection required"

2. Question types (vary them, generate {num_natural} natural + {num_structured} structured):
   - General: "What items are visible in this X-ray scan?"
   - Specific: "Is there a gun/knife/explosive in this scan?"
   - Location: "Where are the threat items located?"
   - Count: "How many prohibited items are in this scan?"
   - Detailed: "Describe all items visible with their locations"
   - Occlusion: "Are any items concealed or overlapping?"
   - Structured: "List all items detected in JSON format." (if structured_ratio > 0)

3. Answer guidelines for NATURAL language questions:
   - Use natural, conversational language
   - Be specific about item names (use ground truth categories)
   - Include spatial information (upper-left, center, lower-right, etc.)
   - Mention if items are partially visible, concealed, or overlapping
   - For clean scans, say "No prohibited items detected"
   - Keep answers focused and concise (2-4 sentences)

4. Answer guidelines for STRUCTURED JSON questions:
   - Output VALID JSON only (no markdown, no extra text)
   - Use exact schema format shown above
   - Include all detected items with name, confidence, location
   - Set has_concealed_items based on visibility
   - Ensure confidence scores are reasonable (0.7-0.95)

5. Accuracy requirements:
   - ONLY mention items present in ground truth
   - Do NOT hallucinate items not in the list
   - Be precise about item names
   - Use location descriptions that match ground truth positions

6. Diversity:
   - Vary question phrasing
   - Mix question types appropriately
   - Include both positive and negative examples (for specific questions){structured_instructions}

Output Format (JSON array only, no other text):
[
  {{
    "question": "What threat items can you identify in this X-ray scan?",
    "answer": "I can see a handgun in the upper-left area and a knife in the center portion of the baggage.",
    "question_type": "general"
  }},
  {{
    "question": "List all items detected in this X-ray scan in JSON format.",
    "answer": '{{"items": [{{"name": "gun", "confidence": 0.92, "location": "upper-left"}}, {{"name": "knife", "confidence": 0.88, "location": "center"}}], "total_count": 2, "has_concealed_items": false}}',
    "question_type": "structured_list"
  }},
  ...
]

CRITICAL REQUIREMENTS:
- Output ONLY a valid JSON array, nothing else
- No markdown code blocks (no ```)
- No additional text or explanations before/after
- Ensure valid JSON syntax with proper escaping
- Generate EXACTLY {num_natural} natural language questions
- Generate EXACTLY {num_structured} structured_list questions
- For structured_list: answer MUST be a JSON string (with escaped quotes)
- For structured_list: question MUST be "List all items detected in this X-ray scan in JSON format."
- Mixed question types for variety"""
        
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
        structured_ratio: float = 0.0
    ) -> List[Dict]:
        """
        Generate VQA pairs for a single image.
        
        Args:
            image_path: Path to X-ray image
            ground_truth: Ground truth annotations
            num_pairs: Number of VQA pairs to generate
            retry: Number of retries on failure
            structured_ratio: Ratio of structured JSON questions (0.0-1.0)
        
        Returns:
            List of VQA pair dictionaries
        """
        # Create prompt
        prompt = self._create_prompt(ground_truth, num_pairs, structured_ratio)
        
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
                    else:
                        # OpenAI-compatible API (includes GPT, Gemini via AI Gateway, vLLM, etc.)
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
    num_samples: Optional[int] = None,
    random_seed: int = 42,
    structured_ratio: float = 0.0,
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
        num_samples: Randomly sample N images from dataset (None = use all)
        random_seed: Random seed for reproducible sampling
        structured_ratio: Ratio of structured JSON questions (0.0-1.0, default: 0.0)
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
    if structured_ratio > 0:
        print(f"Structured JSON ratio: {structured_ratio*100:.0f}%")
    if num_samples:
        print(f"Random sampling: {num_samples} images (seed={random_seed})")
    
    # Load annotations
    with open(annotations_file) as f:
        annotations = json.load(f)
    
    print(f"\nLoaded {len(annotations)} annotations")
    
    # Random sampling if specified
    if num_samples and num_samples < len(annotations):
        import random
        print(f"Randomly sampling {num_samples} images from {len(annotations)}...")
        random.seed(random_seed)
        annotations = random.sample(annotations, num_samples)
        print(f"✓ Sampled {len(annotations)} images (seed={random_seed})")
    
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
                num_pairs=samples_per_image,
                structured_ratio=structured_ratio
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
        help="LLM model to use (gemini-2.0-flash, claude-3-5-sonnet-20241022, gpt-4o, qwen2.5-vl-2b)",
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
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Randomly sample N images from dataset (default: use all). Useful for testing or small datasets.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--structured-ratio",
        type=float,
        default=0.0,
        help="Ratio of structured JSON questions (0.0-1.0, default: 0.0 = all natural language)",
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
            num_samples=args.num_samples,
            random_seed=args.random_seed,
            structured_ratio=args.structured_ratio,
        )


if __name__ == "__main__":
    main()
