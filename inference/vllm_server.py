#!/usr/bin/env python3
"""
vLLM inference server for Qwen2.5-VL X-ray VQA.
Provides fast inference with PagedAttention and continuous batching.
"""

import argparse
import base64
import io
import json
from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image
from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser


class XrayVQAServer:
    """vLLM-based inference server for X-ray VQA."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 32,
        max_model_len: int = 2048,
        dtype: str = "bfloat16",
    ):
        """
        Initialize vLLM server.
        
        Args:
            model_path: Path to fine-tuned model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0-1)
            max_num_seqs: Maximum number of sequences in batch
            max_model_len: Maximum model sequence length
            dtype: Data type (bfloat16, float16)
        """
        print("Initializing vLLM engine...")
        print(f"  Model: {model_path}")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization}")
        print(f"  Max sequences: {max_num_seqs}")
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype=dtype,
        )
        
        print("✓ vLLM engine initialized successfully")
    
    def prepare_prompt(self, question: str) -> str:
        """Prepare VQA prompt."""
        return f"Question: {question}\nAnswer:"
    
    def generate(
        self,
        prompts: List[str],
        images: Optional[List[Image.Image]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[str]:
        """
        Generate answers for batch of prompts.
        
        Args:
            prompts: List of question prompts
            images: List of PIL Images (if using vision model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter
        
        Returns:
            List of generated answers
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            skip_special_tokens=True,
        )
        
        # Generate
        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        
        # Extract answers
        answers = []
        for output in outputs:
            generated_text = output.outputs[0].text
            
            # Extract answer (text after "Answer:")
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text.strip()
            
            answers.append(answer)
        
        return answers
    
    def inspect_xray(
        self,
        image_path: str = None,
        image_base64: str = None,
        question: str = None,
        declared_items: List[str] = None,
    ) -> Dict:
        """
        Inspect X-ray image with VQA.
        
        Args:
            image_path: Path to X-ray image
            image_base64: Base64-encoded image
            question: Custom question (optional)
            declared_items: List of declared items for comparison
        
        Returns:
            Inspection result dictionary
        """
        # Load image
        if image_path:
            image = Image.open(image_path).convert("RGB")
        elif image_base64:
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            raise ValueError("Either image_path or image_base64 must be provided")
        
        # Prepare questions
        questions = []
        
        if question:
            # Custom question
            questions.append(question)
        else:
            # Default questions
            questions.append("What prohibited items are visible in this X-ray scan?")
            
            if declared_items:
                declaration_str = ", ".join(declared_items)
                questions.append(
                    f"Is the declared content '{declaration_str}' consistent with the X-ray image?"
                )
        
        # Prepare prompts
        prompts = [self.prepare_prompt(q) for q in questions]
        
        # Generate answers
        answers = self.generate(prompts, images=[image] * len(prompts))
        
        # Parse results
        result = {
            "questions": questions,
            "answers": answers,
            "declared_items": declared_items,
        }
        
        # Risk assessment (simple heuristic based on keywords)
        combined_text = " ".join(answers).lower()
        
        prohibited_keywords = ["knife", "blade", "scissor", "weapon", "threat"]
        risk_keywords = ["detected", "visible", "found", "inconsistent", "mismatch"]
        
        has_prohibited = any(kw in combined_text for kw in prohibited_keywords)
        has_risk = any(kw in combined_text for kw in risk_keywords)
        
        if has_prohibited and has_risk:
            result["risk_level"] = "high"
            result["recommended_action"] = "PHYSICAL_INSPECTION"
        elif has_prohibited:
            result["risk_level"] = "medium"
            result["recommended_action"] = "REVIEW"
        else:
            result["risk_level"] = "low"
            result["recommended_action"] = "CLEAR"
        
        return result


def main():
    parser = argparse.ArgumentParser(description="vLLM inference server for X-ray VQA")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization (0-1)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=32,
        help="Maximum number of sequences to process in batch",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model sequence length",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Model data type",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with sample inference",
    )
    
    args = parser.parse_args()
    
    # Initialize server
    server = XrayVQAServer(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )
    
    if args.test_mode:
        # Test mode: run sample inference
        print("\n" + "=" * 60)
        print("Test Mode: Running sample inference")
        print("=" * 60)
        
        # Sample prompt
        test_prompt = server.prepare_prompt(
            "What prohibited items are visible in this X-ray scan?"
        )
        
        print(f"Test prompt: {test_prompt}")
        
        answers = server.generate([test_prompt])
        
        print(f"\nGenerated answer: {answers[0]}")
        print("\n✓ Test inference successful!")
        
    else:
        # Production mode: start HTTP server
        print("\n" + "=" * 60)
        print(f"Starting HTTP server on port {args.port}...")
        print("=" * 60)
        print("\nNote: This is a standalone vLLM engine.")
        print("For full API server with REST endpoints, use:")
        print(f"  python inference/api_server.py --vllm-url http://localhost:{args.port}")
        print("\nFor programmatic usage:")
        print("  from inference.vllm_server import XrayVQAServer")
        print(f"  server = XrayVQAServer('{args.model}')")
        print("  results = server.inspect_xray(image_path='scan.jpg')")
        
        # Keep alive
        print("\nServer ready. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")


if __name__ == "__main__":
    main()
