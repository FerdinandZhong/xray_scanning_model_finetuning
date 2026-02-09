#!/usr/bin/env python3
"""
Phase 2: Ray Train distributed fine-tuning of Qwen2.5-VL-7B-Instruct with LoRA.
Scales training across multiple nodes and GPUs using Ray cluster.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig
from ray.train.torch import TorchTrainer

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from training.vqa_dataset import XrayVQADataset, collate_fn


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_func(config: Dict):
    """
    Training function for Ray Train.
    This runs on each worker in the Ray cluster.
    
    Args:
        config: Training configuration dictionary
    """
    # Get Ray Train context
    train_context = train.get_context()
    world_rank = train_context.get_world_rank()
    local_rank = train_context.get_local_rank()
    world_size = train_context.get_world_size()
    
    print(f"Worker {world_rank}/{world_size} (local rank {local_rank}) starting...")
    
    # Set seed
    set_seed(config.get("seed", 42))
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16 if config.get("bf16", True) else torch.float16,
        trust_remote_code=True,
    )
    
    if world_rank == 0:
        print(f"Model loaded. Total parameters: {model.num_parameters():,}")
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=config["lora"].get("r", 64),
        lora_alpha=config["lora"].get("alpha", 128),
        target_modules=config["lora"].get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_dropout=config["lora"].get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    if world_rank == 0:
        model.print_trainable_parameters()
    
    # Prepare model for distributed training
    model = train.torch.prepare_model(model)
    
    # Load datasets
    if world_rank == 0:
        print(f"Loading training data from {config['train_file']}")
    
    train_dataset = XrayVQADataset(
        jsonl_file=config["train_file"],
        processor=processor,
        image_resolution=config.get("image_resolution", 448),
        max_seq_length=config.get("max_seq_length", 2048),
    )
    
    eval_dataset = XrayVQADataset(
        jsonl_file=config["eval_file"],
        processor=processor,
        image_resolution=config.get("image_resolution", 448),
        max_seq_length=config.get("max_seq_length", 2048),
    )
    
    # Create collate function
    def collate_wrapper(batch):
        return collate_fn(batch, processor, config.get("max_seq_length", 2048))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 2e-4),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        weight_decay=config.get("weight_decay", 0.01),
        bf16=config.get("bf16", True),
        
        # Logging
        logging_dir=os.path.join(config["output_dir"], "logs"),
        logging_steps=config.get("logging_steps", 20),
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=config.get("eval_steps", 500),
        
        # Checkpointing (handled by Ray)
        save_strategy="steps",
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        
        # Performance
        dataloader_num_workers=0,  # Ray handles data loading
        remove_unused_columns=False,
        
        # Distributed training
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_wrapper,
        tokenizer=processor.tokenizer,
    )
    
    # Train
    if world_rank == 0:
        print("\n" + "=" * 60)
        print("Starting Ray distributed training...")
        print(f"World size: {world_size}")
        print(f"Effective batch size: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps'] * world_size}")
        print("=" * 60 + "\n")
    
    results = trainer.train()
    
    # Save final model (only on rank 0)
    if world_rank == 0:
        trainer.save_model(config["output_dir"])
        processor.save_pretrained(config["output_dir"])
        print(f"\n✓ Model saved to: {config['output_dir']}")
    
    # Report metrics to Ray
    metrics = trainer.evaluate()
    train.report(metrics)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ray Train distributed fine-tuning for Qwen2.5-VL"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default="auto",
        help="Ray cluster address (auto, or ray://host:port)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers (overrides config)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("=" * 60)
    print("Qwen2.5-VL X-ray VQA Fine-tuning (Phase 2 - Ray Train)")
    print("=" * 60)
    
    config = load_config(args.config)
    print(f"\nConfiguration loaded from: {args.config}")
    
    # Initialize Ray
    print(f"\nConnecting to Ray cluster: {args.ray_address}")
    ray.init(address=args.ray_address)
    
    print(f"Ray cluster resources:")
    print(f"  CPUs: {ray.cluster_resources().get('CPU', 0)}")
    print(f"  GPUs: {ray.cluster_resources().get('GPU', 0)}")
    print(f"  Nodes: {len(ray.nodes())}")
    
    # Ray scaling configuration
    ray_config = config.get("ray", {})
    scaling_config_dict = ray_config.get("scaling_config", {})
    
    num_workers = args.num_workers or scaling_config_dict.get("num_workers", 4)
    use_gpu = scaling_config_dict.get("use_gpu", True)
    resources_per_worker = scaling_config_dict.get("resources_per_worker", {"CPU": 8, "GPU": 1})
    
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker,
    )
    
    # Ray run configuration
    run_config_dict = ray_config.get("run_config", {})
    checkpoint_config_dict = run_config_dict.get("checkpoint_config", {})
    failure_config_dict = run_config_dict.get("failure_config", {})
    
    run_config = RunConfig(
        name="qwen_vl_xray_training",
        storage_path=config["output_dir"],
        checkpoint_config=CheckpointConfig(
            num_to_keep=checkpoint_config_dict.get("num_to_keep", 3),
            checkpoint_frequency=checkpoint_config_dict.get("checkpoint_frequency", 500),
        ),
        failure_config=FailureConfig(
            max_failures=failure_config_dict.get("max_failures", 3),
        ),
    )
    
    print(f"\nRay Train configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  GPUs per worker: {resources_per_worker.get('GPU', 1)}")
    print(f"  Total GPUs: {num_workers * resources_per_worker.get('GPU', 1)}")
    
    # Create TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting Ray Train job...")
    print("=" * 60 + "\n")
    
    result = trainer.fit()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Final metrics: {result.metrics}")
    print(f"Checkpoint: {result.checkpoint}")
    
    # Shutdown Ray
    ray.shutdown()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print(f"1. Evaluate model: python evaluation/eval_vqa.py --model {config['output_dir']}")
    print(f"2. Deploy inference: python inference/vllm_server.py --model {config['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
