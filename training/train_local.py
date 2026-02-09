#!/usr/bin/env python3
"""
Phase 1: Single-node fine-tuning of Qwen2.5-VL-7B-Instruct with LoRA.
Supports automatic multi-GPU training via Hugging Face Trainer.
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from training.vqa_dataset import create_dataloader, collate_fn


@dataclass
class Config:
    """Training configuration."""
    model_name: str
    train_file: str
    eval_file: str
    output_dir: str
    image_resolution: int = 448
    max_seq_length: int = 2048
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    bf16: bool = True
    
    # LoRA config
    lora: dict = field(default_factory=dict)
    
    # Logging and checkpointing
    logging_steps: int = 20
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)


def setup_model_and_processor(config: Config):
    """Load Qwen2.5-VL model and apply LoRA."""
    print(f"Loading model: {config.model_name}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Model loaded. Total parameters: {model.num_parameters():,}")
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora.get("r", 64),
        lora_alpha=config.lora.get("alpha", 128),
        target_modules=config.lora.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_dropout=config.lora.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor


def setup_datasets(config: Config, processor):
    """Setup training and evaluation datasets."""
    from torch.utils.data import Dataset
    from training.vqa_dataset import XrayVQADataset
    
    print(f"Loading training data from {config.train_file}")
    train_dataset = XrayVQADataset(
        jsonl_file=config.train_file,
        processor=processor,
        image_resolution=config.image_resolution,
        max_seq_length=config.max_seq_length,
    )
    
    print(f"Loading validation data from {config.eval_file}")
    eval_dataset = XrayVQADataset(
        jsonl_file=config.eval_file,
        processor=processor,
        image_resolution=config.image_resolution,
        max_seq_length=config.max_seq_length,
    )
    
    return train_dataset, eval_dataset


def create_collate_fn(processor, max_seq_length):
    """Create collate function for DataLoader."""
    def collate(batch):
        return collate_fn(batch, processor, max_seq_length)
    return collate


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL on X-ray VQA dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("=" * 60)
    print("Qwen2.5-VL X-ray VQA Fine-tuning (Phase 1)")
    print("=" * 60)
    
    config = load_config(args.config)
    print(f"\nConfiguration loaded from: {args.config}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Setup model and processor
    model, processor = setup_model_and_processor(config)
    
    # Setup datasets
    train_dataset, eval_dataset = setup_datasets(config, processor)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        
        # Logging
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=config.logging_steps,
        logging_first_step=True,
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        
        # Performance
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        
        # Misc
        remove_unused_columns=False,
        report_to="tensorboard",
        seed=config.seed,
        
        # Multi-GPU (automatic via DDP)
        ddp_find_unused_parameters=False,
    )
    
    print(f"\nTraining arguments:")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Num epochs: {config.num_train_epochs}")
    print(f"  Batch size per device: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps * torch.cuda.device_count()}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    
    # Create data collator
    data_collator = create_collate_fn(processor, config.max_seq_length)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,  # For saving
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save final model
    print("\n" + "=" * 60)
    print("Training complete! Saving model...")
    print("=" * 60)
    
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    
    print(f"\n✓ Model saved to: {config.output_dir}")
    
    # Save training metrics
    metrics = trainer.evaluate()
    print(f"\nFinal evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Instructions for next steps
    print("\n" + "=" * 60)
    print("Next steps:")
    print(f"1. Evaluate model: python evaluation/eval_vqa.py --model {config.output_dir}")
    print(f"2. Test inference: python inference/vllm_server.py --model {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
