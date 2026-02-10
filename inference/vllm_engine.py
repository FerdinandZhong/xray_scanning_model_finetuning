"""
vLLM Inference Engine with XGrammar Guided Generation support.
Provides both structured JSON output and natural language generation.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from PIL import Image


class VLLMInferenceEngine:
    """
    Inference engine using vLLM with guided generation for structured output.
    Supports both structured JSON (with XGrammar) and natural language generation.
    """
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        schema_path: Optional[str] = None,
    ):
        """
        Initialize vLLM inference engine.
        
        Args:
            model_path: Path to fine-tuned model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_model_len: Maximum model context length
            schema_path: Path to JSON schema for guided generation (default: inference/output_schema.json)
        """
        try:
            from vllm import LLM, SamplingParams
            self.vllm_available = True
        except ImportError:
            print("Warning: vLLM not installed. Install with: pip install vllm")
            self.vllm_available = False
            return
        
        self.model_path = model_path
        
        print(f"Loading vLLM model from {model_path}...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        print(f"✓ Model loaded successfully")
        
        # Load output schema for guided generation
        if schema_path is None:
            schema_path = Path(__file__).parent / "output_schema.json"
        
        with open(schema_path) as f:
            self.output_schema = json.load(f)
        
        print(f"✓ Loaded output schema from {schema_path}")
    
    def generate_structured(
        self,
        image_path: str,
        prompt: str = "List all items detected in this X-ray scan in JSON format.",
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using XGrammar guided generation.
        
        Args:
            image_path: Path to X-ray image
            prompt: Question/instruction for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling parameter
        
        Returns:
            Structured output dict with items, total_count, has_concealed_items
        """
        if not self.vllm_available:
            raise RuntimeError("vLLM is not available")
        
        from vllm import SamplingParams
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Format input for Qwen2.5-VL
        # vLLM expects a specific format for vision-language models
        full_prompt = f"Question: {prompt}\nProvide your response in valid JSON format only.\nAnswer:"
        
        # Create sampling params with guided JSON generation
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            guided_json=self.output_schema,  # XGrammar guided generation
        )
        
        # Generate
        # Note: For Qwen2.5-VL with vLLM, you need to pass both text and image
        # This may require vLLM's multi-modal API
        inputs = {
            "prompt": full_prompt,
            "multi_modal_data": {"image": image},
        }
        
        outputs = self.llm.generate(inputs, sampling_params)
        
        # Parse JSON output
        output_text = outputs[0].outputs[0].text
        try:
            result = json.loads(output_text)
            return result
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON output: {e}")
            print(f"Raw output: {output_text}")
            # Return empty result
            return {
                "items": [],
                "total_count": 0,
                "has_concealed_items": False,
            }
    
    def generate_natural(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate natural language response (no guided generation).
        
        Args:
            image_path: Path to X-ray image
            prompt: Question/instruction for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Natural language response string
        """
        if not self.vllm_available:
            raise RuntimeError("vLLM is not available")
        
        from vllm import SamplingParams
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Format input
        full_prompt = f"Question: {prompt}\nAnswer:"
        
        # Create sampling params (no guided generation)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        # Generate
        inputs = {
            "prompt": full_prompt,
            "multi_modal_data": {"image": image},
        }
        
        outputs = self.llm.generate(inputs, sampling_params)
        
        return outputs[0].outputs[0].text
    
    def batch_generate_structured(
        self,
        image_paths: List[str],
        prompts: Optional[List[str]] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Batch generate structured JSON outputs.
        
        Args:
            image_paths: List of image paths
            prompts: List of prompts (default: use same prompt for all)
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            List of structured output dicts
        """
        if prompts is None:
            prompts = ["List all items detected in this X-ray scan in JSON format."] * len(image_paths)
        
        assert len(image_paths) == len(prompts), "Number of images must match prompts"
        
        # Generate for each image (vLLM will batch automatically)
        results = []
        for image_path, prompt in zip(image_paths, prompts):
            result = self.generate_structured(
                image_path=image_path,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            results.append(result)
        
        return results


class TransformersInferenceEngine:
    """
    Fallback inference engine using Transformers (without vLLM).
    Does not support XGrammar guided generation, but provides basic inference.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str = "float16",
    ):
        """
        Initialize Transformers inference engine.
        
        Args:
            model_path: Path to fine-tuned model
            device: Device to use (cuda/cpu)
            torch_dtype: Torch dtype (float16/float32)
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        self.model_path = model_path
        self.device = device
        
        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(torch_dtype, torch.float16)
        
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"✓ Model loaded successfully on {device}")
    
    def generate_structured(
        self,
        image_path: str,
        prompt: str = "List all items detected in this X-ray scan in JSON format.",
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generate structured output (attempts JSON but not guaranteed without XGrammar).
        
        Args:
            image_path: Path to X-ray image
            prompt: Question/instruction
            max_tokens: Maximum tokens
            temperature: Sampling temperature
        
        Returns:
            Parsed JSON dict (or empty dict if parsing fails)
        """
        import torch
        from PIL import Image
        from inference.json_parser import parse_json_response
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Format prompt
        full_prompt = f"Question: {prompt}\nProvide your response in valid JSON format only.\nAnswer:"
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=full_prompt,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        
        # Decode
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (after "Answer:")
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text
        
        # Parse JSON
        result = parse_json_response(answer)
        if result is None:
            print(f"Warning: Failed to parse JSON. Raw output: {answer}")
            result = {
                "items": [],
                "total_count": 0,
                "has_concealed_items": False,
            }
        
        return result
    
    def generate_natural(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate natural language response.
        
        Args:
            image_path: Path to X-ray image
            prompt: Question/instruction
            max_tokens: Maximum tokens
            temperature: Sampling temperature
        
        Returns:
            Natural language response
        """
        import torch
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Format prompt
        full_prompt = f"Question: {prompt}\nAnswer:"
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=full_prompt,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        
        # Decode
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text
        
        return answer


# Factory function
def create_inference_engine(
    model_path: str,
    use_vllm: bool = True,
    **kwargs
) -> Union[VLLMInferenceEngine, TransformersInferenceEngine]:
    """
    Create an inference engine (vLLM or Transformers).
    
    Args:
        model_path: Path to model
        use_vllm: Whether to use vLLM (recommended for production)
        **kwargs: Additional arguments for the engine
    
    Returns:
        Inference engine instance
    """
    if use_vllm:
        try:
            return VLLMInferenceEngine(model_path, **kwargs)
        except (ImportError, RuntimeError) as e:
            print(f"Warning: Failed to create vLLM engine: {e}")
            print("Falling back to Transformers engine...")
            use_vllm = False
    
    if not use_vllm:
        return TransformersInferenceEngine(model_path, **kwargs)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python vllm_engine.py <model_path> <image_path> [--use-transformers]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    use_vllm = "--use-transformers" not in sys.argv
    
    # Create engine
    engine = create_inference_engine(model_path, use_vllm=use_vllm)
    
    # Test structured generation
    print("\n=== Structured JSON Generation ===")
    result = engine.generate_structured(image_path)
    print(json.dumps(result, indent=2))
    
    # Test natural language generation
    print("\n=== Natural Language Generation ===")
    response = engine.generate_natural(
        image_path,
        prompt="What items can you see in this X-ray scan?"
    )
    print(response)
