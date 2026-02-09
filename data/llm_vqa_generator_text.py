#!/usr/bin/env python3
"""
Text-based VQA generator using small LLMs (no vision required).
Uses ground truth annotations to generate natural language VQA pairs.
Cost-effective alternative to vision-capable LLMs like GPT-4V/Claude.

Supported models:
- Qwen2.5-3B-Instruct (local or API)
- Qwen2.5-7B-Instruct (local or API)
- Llama-3-8B-Instruct (local)
- Any OpenAI-compatible API endpoint
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm


class TextLLMVQAGenerator:
    """Generate VQA pairs using text-only LLMs from ground truth annotations."""
    
    def __init__(
        self,
        model: str = "qwen2.5-3b-instruct",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        use_local: bool = True,
    ):
        """
        Initialize text-based VQA generator.
        
        Args:
            model: Model name
            api_base: API base URL for OpenAI-compatible APIs
            api_key: API key (if using API)
            use_local: Use local inference via transformers
        """
        self.model = model
        self.use_local = use_local
        
        if use_local:
            print("Initializing local model...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Map model names to HuggingFace model IDs
            model_mapping = {
                "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
                "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
                "llama-3-8b-instruct": "meta-llama/Llama-3-8B-Instruct",
            }
            
            model_id = model_mapping.get(model, model)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model_obj = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.device = self.model_obj.device
            print(f"✓ Loaded {model_id} on {self.device}")
        else:
            print("Using API endpoint...")
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=api_base,
            )
            print(f"✓ Connected to {api_base or 'OpenAI API'}")
    
    def _create_prompt(self, ground_truth: Dict, num_pairs: int) -> str:
        """Create prompt for text-based VQA generation."""
        categories = ground_truth.get('categories', [])
        bboxes = ground_truth.get('bboxes', [])
        caption = ground_truth.get('caption', '')
        
        # Format categories and locations
        if categories and bboxes:
            items_list = []
            for cat, bbox in zip(categories, bboxes):
                x, y, w, h = bbox
                # Simple location mapping
                center_x = x + w / 2
                center_y = y + h / 2
                
                if center_y < 200:
                    v_pos = "upper"
                elif center_y > 400:
                    v_pos = "lower"
                else:
                    v_pos = "center"
                
                if center_x < 250:
                    h_pos = "left"
                elif center_x > 450:
                    h_pos = "right"
                else:
                    h_pos = "center"
                
                location = f"{v_pos}-{h_pos}" if v_pos != "center" or h_pos != "center" else "center"
                items_list.append(f"- {cat} at {location}")
            
            items_str = "\n".join(items_list)
        else:
            items_str = "None (clean scan)"
        
        prompt = f"""You are an X-ray security expert. Generate {num_pairs} diverse Question-Answer pairs for training a Visual Question Answering model.

Image Information:
{items_str}
{f"Description: {caption}" if caption else ""}

Requirements:
1. Focus on ITEM RECOGNITION ONLY (no risk assessment, no actions)
2. Question types: general, specific, location, count, detailed, occlusion
3. Use natural, conversational language
4. For clean scans: "No prohibited items detected"
5. Include locations when relevant
6. Vary question phrasing
7. ONLY mention items from the list above

Output Format (JSON array only, no markdown, no extra text):
[
  {{"question": "What items are visible in this X-ray scan?", "answer": "...", "question_type": "general"}},
  {{"question": "Is there a gun in this scan?", "answer": "...", "question_type": "specific"}},
  ...
]

Generate exactly {num_pairs} diverse VQA pairs as a JSON array:"""
        
        return prompt
    
    def _call_local(self, prompt: str) -> str:
        """Call local model."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates VQA data."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        outputs = self.model_obj.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|im_start|>assistant" in generated_text:
            response = generated_text.split("<|im_start|>assistant")[-1].strip()
        else:
            response = generated_text[len(text):].strip()
        
        return response
    
    def _call_api(self, prompt: str) -> str:
        """Call API endpoint."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates VQA data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
    
    def _parse_response(self, response: str) -> List[Dict]:
        """Parse LLM response into VQA pairs."""
        import re
        
        # Remove markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Extract JSON array
            json_str = response.strip()
            array_start = json_str.find('[')
            array_end = json_str.rfind(']')
            if array_start != -1 and array_end != -1:
                json_str = json_str[array_start:array_end + 1]
        
        try:
            vqa_pairs = json.loads(json_str)
            
            if not isinstance(vqa_pairs, list):
                return []
            
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
        ground_truth: Dict,
        num_pairs: int = 3,
        retry: int = 3,
    ) -> List[Dict]:
        """
        Generate VQA pairs from ground truth annotations.
        
        Args:
            ground_truth: Ground truth annotations
            num_pairs: Number of VQA pairs to generate
            retry: Number of retries on failure
        
        Returns:
            List of VQA pair dictionaries
        """
        prompt = self._create_prompt(ground_truth, num_pairs)
        
        for attempt in range(retry):
            try:
                if self.use_local:
                    response = self._call_local(prompt)
                else:
                    response = self._call_api(prompt)
                
                vqa_pairs = self._parse_response(response)
                
                if vqa_pairs:
                    return vqa_pairs
                
                print(f"Retry {attempt + 1}/{retry}: No valid pairs generated")
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{retry}: {e}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)
        
        return []


def process_dataset(
    annotations_file: str,
    output_file: str,
    model: str = "qwen2.5-3b-instruct",
    samples_per_image: int = 3,
    max_images: Optional[int] = None,
    use_local: bool = True,
    api_base: Optional[str] = None,
    batch_save: int = 100,
):
    """Process dataset with text-based LLM."""
    print("=" * 60)
    print("Text-Based VQA Dataset Generation")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Mode: {'Local' if use_local else 'API'}")
    print(f"Annotations: {annotations_file}")
    print(f"Output: {output_file}")
    print(f"Samples per image: {samples_per_image}")
    
    # Load annotations
    with open(annotations_file) as f:
        annotations = json.load(f)
    
    print(f"\nLoaded {len(annotations)} annotations")
    
    if max_images:
        annotations = annotations[:max_images]
        print(f"Limited to {max_images} images")
    
    # Initialize generator
    generator = TextLLMVQAGenerator(
        model=model,
        api_base=api_base,
        use_local=use_local,
    )
    
    # Process images
    vqa_dataset = []
    failed_images = []
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint
    checkpoint_file = output_path.parent / f"{output_path.stem}_checkpoint.jsonl"
    
    start_idx = 0
    if checkpoint_file.exists():
        print(f"\nResuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file) as f:
            for line in f:
                vqa_dataset.append(json.loads(line.strip()))
        if vqa_dataset:
            start_idx = vqa_dataset[-1]["metadata"]["image_id"] + 1
            print(f"Resuming from image {start_idx}")
    
    # Process each annotation
    for idx, ann in enumerate(tqdm(
        annotations[start_idx:],
        desc="Generating VQA pairs",
        initial=start_idx,
        total=len(annotations)
    )):
        try:
            vqa_pairs = generator.generate_vqa_pairs(
                ground_truth=ann,
                num_pairs=samples_per_image
            )
            
            for pair in vqa_pairs:
                vqa_dataset.append({
                    "image_path": ann.get('image_path', f"images/{ann['image_id']:06d}.jpg"),
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "metadata": {
                        "image_id": ann['image_id'],
                        "question_type": pair.get("question_type", "general"),
                        "categories": ann.get('categories', []),
                        "num_categories": len(ann.get('categories', [])),
                        "generated_by": "text_llm",
                        "model": model,
                    }
                })
            
            # Checkpoint save
            if (idx + 1) % batch_save == 0:
                with open(checkpoint_file, 'w') as f:
                    for item in vqa_dataset:
                        f.write(json.dumps(item) + '\n')
                print(f"\nCheckpoint saved: {len(vqa_dataset)} pairs")
            
        except Exception as e:
            print(f"\nError processing annotation {ann['image_id']}: {e}")
            failed_images.append({"image_id": ann['image_id'], "error": str(e)})
    
    # Save final dataset
    print(f"\n\nSaving final dataset to {output_file}...")
    with open(output_file, 'w') as f:
        for item in vqa_dataset:
            f.write(json.dumps(item) + '\n')
    
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    # Statistics
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total VQA pairs: {len(vqa_dataset)}")
    print(f"Failed images: {len(failed_images)}")
    print(f"Success rate: {(1 - len(failed_images) / len(annotations)) * 100:.1f}%")
    
    print(f"\n✓ Dataset saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VQA dataset using text-only LLM"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to annotations JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-3b-instruct",
        choices=[
            "qwen2.5-3b-instruct",
            "qwen2.5-7b-instruct",
            "llama-3-8b-instruct",
        ],
        help="Model to use",
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
        help="Max images to process",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use API instead of local inference",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL (OpenAI-compatible)",
    )
    parser.add_argument(
        "--batch-save",
        type=int,
        default=100,
        help="Save checkpoint every N images",
    )
    
    args = parser.parse_args()
    
    process_dataset(
        annotations_file=args.annotations,
        output_file=args.output,
        model=args.model,
        samples_per_image=args.samples_per_image,
        max_images=args.max_images,
        use_local=not args.use_api,
        api_base=args.api_base,
        batch_save=args.batch_save,
    )


if __name__ == "__main__":
    main()
