"""
PyTorch Dataset class for VQA (Visual Question Answering) on X-ray images.
Handles Qwen2.5-VL image preprocessing and text tokenization.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


class XrayVQADataset(Dataset):
    """Dataset for X-ray VQA training with Qwen2.5-VL."""
    
    def __init__(
        self,
        jsonl_file: str,
        processor: AutoProcessor,
        image_resolution: int = 448,
        max_seq_length: int = 2048,
        image_root: Optional[str] = None,
    ):
        """
        Args:
            jsonl_file: Path to JSONL file with VQA pairs
            processor: Qwen2.5-VL processor (AutoProcessor)
            image_resolution: Image resolution (default: 448)
            max_seq_length: Maximum sequence length for text
            image_root: Root directory for images (if paths in JSONL are relative)
        """
        self.jsonl_file = Path(jsonl_file)
        self.processor = processor
        self.image_resolution = image_resolution
        self.max_seq_length = max_seq_length
        self.image_root = Path(image_root) if image_root else None
        
        # Load dataset
        self.data = []
        with open(self.jsonl_file, "r") as f:
            for line in f:
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
        
        # Format prompt for Qwen2.5-VL
        # The model expects: image + question, and learns to generate answer
        prompt = f"Question: {question}\nAnswer:"
        
        return {
            "image": image,
            "prompt": prompt,
            "answer": answer,
            "metadata": item.get("metadata", {}),
        }


def collate_fn(batch, processor, max_seq_length=2048):
    """
    Custom collate function for batching VQA samples.
    Handles vision-language inputs for Qwen2.5-VL.
    """
    images = [item["image"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    answers = [item["answer"] for item in batch]
    
    # Process images with Qwen2.5-VL processor
    # Note: Qwen2.5-VL uses special image tokens in text
    # Format: <|vision_start|><|image_pad|><|vision_end|>Question: ...
    
    # Prepare full texts (prompt + answer for training)
    full_texts = [f"{prompt} {answer}" for prompt, answer in zip(prompts, answers)]
    
    # Tokenize text
    text_inputs = processor.tokenizer(
        full_texts,
        padding="longest",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    # Process images
    # Qwen2.5-VL expects images to be processed separately
    image_inputs = processor.image_processor(
        images,
        return_tensors="pt",
    )
    
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
):
    """
    Create a DataLoader for VQA training.
    
    Args:
        jsonl_file: Path to JSONL file
        processor: Qwen2.5-VL processor
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        image_resolution: Image resolution
        max_seq_length: Max sequence length
        image_root: Root directory for images
    
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
    )
    
    # Create custom collate function with processor
    def collate_wrapper(batch):
        return collate_fn(batch, processor, max_seq_length)
    
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
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,
    )
    
    # Create dataset
    dataset = XrayVQADataset(
        jsonl_file="data/opixray_vqa_train.jsonl",
        processor=processor,
        image_resolution=448,
        max_seq_length=2048,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Prompt: {sample['prompt']}")
    print(f"Answer: {sample['answer'][:100]}...")
    
    # Test dataloader
    dataloader = create_dataloader(
        jsonl_file="data/opixray_vqa_train.jsonl",
        processor=processor,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # 0 for testing
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    if batch.get("pixel_values") is not None:
        print(f"Pixel values shape: {batch['pixel_values'].shape}")
