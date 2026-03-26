"""
PyTorch Dataset class for VQA (Visual Question Answering) on X-ray images.
Handles Qwen3-VL / Qwen2.5-VL image preprocessing and text tokenization.

Supports multi-object detection with structured JSON outputs.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


class XrayVQADataset(Dataset):
    """Dataset for X-ray VQA training with Qwen3-VL / Qwen2.5-VL."""
    
    def __init__(
        self,
        jsonl_file: str,
        processor: AutoProcessor,
        image_resolution: int = 448,
        max_seq_length: int = 2048,
        image_root: Optional[str] = None,
        use_chat_template: bool = True,
    ):
        """
        Args:
            jsonl_file: Path to JSONL file with VQA pairs
            processor: Qwen3-VL or Qwen2.5-VL processor (AutoProcessor)
            image_resolution: Image resolution (default: 448)
            max_seq_length: Maximum sequence length for text
            image_root: Root directory for images (if paths in JSONL are relative)
            use_chat_template: Use chat template formatting (recommended for Qwen3-VL)
        """
        self.jsonl_file = Path(jsonl_file)
        self.processor = processor
        self.image_resolution = image_resolution
        self.max_seq_length = max_seq_length
        self.image_root = Path(image_root) if image_root else None
        self.use_chat_template = use_chat_template
        
        # Load dataset
        self.data = []
        with open(self.jsonl_file, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    self.data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(self.data)} VQA pairs from {self.jsonl_file}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single VQA sample."""
        item = self.data[idx]
        
        # Load image
        image_path = item["image_path"]
        if self.image_root:
            # Handle relative paths
            image_path = self.image_root / Path(image_path).name
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new("RGB", (self.image_resolution, self.image_resolution), color=(128, 128, 128))
        
        # Get question and answer
        question = item["question"]
        answer = item["answer"]
        question_type = item.get("metadata", {}).get("question_type", "general")
        
        # Format prompt for Qwen3-VL / Qwen2.5-VL
        # The model expects: image + question, and learns to generate answer
        # For structured JSON questions, add explicit instruction
        if question_type == "structured_list":
            prompt = f"{question}\nProvide your response in valid JSON format only."
        else:
            prompt = question
        
        return {
            "image": image,
            "question": question,
            "prompt": prompt,
            "answer": answer,
            "metadata": item.get("metadata", {}),
            "question_type": question_type,
        }


def collate_fn(batch, processor, max_seq_length=2048, use_chat_template=True):
    """
    Custom collate function for batching VQA samples.
    Handles vision-language inputs for Qwen3-VL / Qwen2.5-VL.
    
    Args:
        batch: List of dataset items
        processor: Model processor
        max_seq_length: Maximum sequence length
        use_chat_template: Use chat template formatting (for Qwen3-VL)
    """
    images = [item["image"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    answers = [item["answer"] for item in batch]
    
    # For Qwen3-VL, use chat template format
    # Process images with Qwen3-VL/Qwen2.5-VL processor
    # Note: Models use special image tokens in text
    # Format: <|vision_start|><|image_pad|><|vision_end|>Question: ...
    
    if use_chat_template and hasattr(processor, 'apply_chat_template'):
        # Use chat template for Qwen3-VL
        # Build messages for each sample
        conversations = []
        for prompt, answer in zip(prompts, answers):
            messages = [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
            conversations.append(messages)
        
        # Apply chat template
        texts = []
        for messages in conversations:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        full_texts = texts
    else:
        # Fallback: simple concatenation
        full_texts = [f"{prompt}\nAnswer: {answer}" for prompt, answer in zip(prompts, answers)]
    
    # Tokenize text
    text_inputs = processor.tokenizer(
        full_texts,
        padding="longest",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    # Process images
    # Qwen3-VL/Qwen2.5-VL expects images to be processed separately
    try:
        image_inputs = processor.image_processor(
            images,
            return_tensors="pt",
        )
    except Exception as e:
        # Fallback for different processor interfaces
        print(f"Warning: image_processor failed: {e}, trying direct processing")
        image_inputs = {"pixel_values": None}
    
    # Create labels for causal LM training
    # We want the model to predict the answer, not the question
    # So we mask out the prompt part in labels
    labels = text_inputs["input_ids"].clone()
    
    # Find where the answer starts (after "Answer:")
    for i, (prompt, full_text) in enumerate(zip(prompts, full_texts)):
        # Tokenize just the prompt to find its length
        prompt_tokens = processor.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        
        # Mask prompt tokens with -100 (ignored in loss)
        labels[i, :len(prompt_tokens)] = -100
    
    # Mask padding tokens
    labels[text_inputs["attention_mask"] == 0] = -100
    
    return {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "pixel_values": image_inputs.get("pixel_values"),  # May be None for some processors
        "labels": labels,
    }


def create_dataloader(
    jsonl_file: str,
    processor: AutoProcessor,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
    image_resolution: int = 448,
    max_seq_length: int = 2048,
    image_root: Optional[str] = None,
    use_chat_template: bool = True,
):
    """
    Create a DataLoader for VQA training.
    
    Args:
        jsonl_file: Path to JSONL file
        processor: Qwen3-VL or Qwen2.5-VL processor
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        image_resolution: Image resolution
        max_seq_length: Max sequence length
        image_root: Root directory for images
        use_chat_template: Use chat template formatting (for Qwen3-VL)
    
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    dataset = XrayVQADataset(
        jsonl_file=jsonl_file,
        processor=processor,
        image_resolution=image_resolution,
        max_seq_length=max_seq_length,
        image_root=image_root,
        use_chat_template=use_chat_template,
    )
    
    # Create custom collate function with processor
    def collate_wrapper(batch):
        return collate_fn(batch, processor, max_seq_length, use_chat_template)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_wrapper,
        pin_memory=True,
    )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    from transformers import AutoProcessor
    
    # Load processor (Qwen3-VL or Qwen2.5-VL)
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",  # or "Qwen/Qwen2.5-VL-7B-Instruct"
        trust_remote_code=True,
    )
    
    # Create dataset
    dataset = XrayVQADataset(
        jsonl_file="data/stcray_vlm/stcray_vlm_train.jsonl",
        processor=processor,
        image_resolution=448,
        max_seq_length=2048,
        use_chat_template=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Prompt: {sample['prompt']}")
    print(f"Answer: {sample['answer'][:100]}...")
    
    # Test dataloader
    dataloader = create_dataloader(
        jsonl_file="data/stcray_vlm/stcray_vlm_train.jsonl",
        processor=processor,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # 0 for testing
        use_chat_template=True,
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    if batch.get("pixel_values") is not None:
        print(f"Pixel values shape: {batch['pixel_values'].shape}")
