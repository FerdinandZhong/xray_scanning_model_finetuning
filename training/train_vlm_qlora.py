#!/usr/bin/env python3
"""
Train Qwen3-VL-2B-Instruct with QLoRA for multi-object X-ray threat detection.

This script fine-tunes the model on STCray dataset (46,642 images, 21 categories)
using QLoRA (4-bit quantization + LoRA adapters) to fit on T4 GPU (16GB VRAM).

Features:
- 4-bit NF4 quantization with double quantization
- LoRA adapters for memory-efficient training
- Multi-object detection with structured JSON output
- Gradient checkpointing for memory optimization
- Mixed precision training (BF16)
- Automatic checkpoint saving and resumption
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Import custom dataset
import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.vqa_dataset import XrayVQADataset, create_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Qwen3-VL-2B with QLoRA on STCray X-ray dataset"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Pretrained model name or path"
    )
    
    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/stcray_vlm/stcray_vlm_train.jsonl",
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="data/stcray_vlm/stcray_vlm_test.jsonl",
        help="Path to evaluation JSONL file"
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Root directory for images (if paths in JSONL are relative)"
    )
    
    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/qwen3vl-2b-xray-qlora",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Batch size per GPU for training"
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=2,
        help="Batch size per GPU for evaluation"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps"
    )
    
    # QLoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    
    # Other
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def setup_model_and_tokenizer(args):
    """
    Setup Qwen3-VL model with QLoRA configuration.
    
    Returns:
        model: PEFT model with LoRA adapters
        processor: Model processor
    """
    print("=" * 60)
    print("Setting up model with QLoRA...")
    print("=" * 60)
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Double quantization for extra memory savings
    )
    
    print(f"Loading base model: {args.model_name}")
    print("  - 4-bit NF4 quantization enabled")
    print("  - Double quantization enabled")
    print("  - Compute dtype: bfloat16")
    
    # Load model with quantization
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    
    print(f"✓ Model loaded: {args.model_name}")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    print("✓ Model prepared for k-bit training")
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # Uncomment to train more modules (requires more VRAM):
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print(f"\nLoRA Configuration:")
    print(f"  - Rank (r): {args.lora_r}")
    print(f"  - Alpha: {args.lora_alpha}")
    print(f"  - Dropout: {args.lora_dropout}")
    print(f"  - Target modules: {lora_config.target_modules}")
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters:")
    print(f"  - Trainable: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
    print(f"  - Total: {total_params:,}")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled")
    
    print("=" * 60)
    print()
    
    return model, processor


def setup_datasets(args, processor):
    """Setup training and evaluation datasets."""
    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    
    train_dataset = XrayVQADataset(
        jsonl_file=args.train_data,
        processor=processor,
        max_seq_length=args.max_seq_length,
        image_root=args.image_root,
        use_chat_template=True,
    )
    
    eval_dataset = None
    if args.eval_data and Path(args.eval_data).exists():
        eval_dataset = XrayVQADataset(
            jsonl_file=args.eval_data,
            processor=processor,
            max_seq_length=args.max_seq_length,
            image_root=args.image_root,
            use_chat_template=True,
        )
        print(f"✓ Evaluation dataset: {len(eval_dataset)} samples")
    
    print("=" * 60)
    print()
    
    return train_dataset, eval_dataset


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training arguments
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n" + "=" * 60)
    print("Qwen3-VL-2B QLoRA Training for X-ray Object Detection")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Train data: {args.train_data}")
    print(f"Eval data: {args.eval_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps} (effective)")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    print()
    
    # Setup model and processor
    model, processor = setup_model_and_tokenizer(args)
    
    # Setup datasets
    train_dataset, eval_dataset = setup_datasets(args, processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        save_total_limit=3,  # Keep only last 3 checkpoints
        evaluation_strategy="steps" if eval_dataset else "no",
        logging_dir=str(output_dir / "logs"),
        fp16=False,
        bf16=True,  # Use BF16 for better stability with quantized models
        optim="paged_adamw_8bit",  # 8-bit Adam optimizer
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # Important for vision-language models
        report_to="tensorboard",
        seed=args.seed,
        gradient_checkpointing=True,
        # Push to hub (optional)
        # push_to_hub=False,
    )
    
    print("=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"  Optimizer: {training_args.optim}")
    print(f"  Mixed precision: BF16")
    print(f"  Gradient checkpointing: Enabled")
    print(f"  Max gradient norm: {training_args.max_grad_norm}")
    print(f"  Logging: {training_args.logging_dir}")
    print("=" * 60)
    print()
    
    # Custom data collator for VQA
    from training.vqa_dataset import collate_fn
    
    def data_collator(batch):
        return collate_fn(batch, processor, args.max_seq_length, use_chat_template=True)
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Resume from checkpoint if specified
    checkpoint = args.resume_from_checkpoint
    if checkpoint and not Path(checkpoint).exists():
        print(f"⚠️  Checkpoint not found: {checkpoint}")
        print("    Starting training from scratch...")
        checkpoint = None
    
    # Train
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    print()
    
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_model(str(output_dir / "interrupted"))
        print("✓ Checkpoint saved")
        return
    
    # Save final model
    print("\n" + "=" * 60)
    print("Training complete! Saving final model...")
    print("=" * 60)
    
    # Save LoRA adapters
    model.save_pretrained(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    
    print(f"✓ Model saved to: {output_dir / 'final'}")
    print()
    print("To use the fine-tuned model:")
    print("  1. Load base model with quantization")
    print("  2. Load LoRA adapters from final/")
    print("  3. Run inference")
    print()
    print("=" * 60)
    print("✅ Training finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
