#!/usr/bin/env python3
"""
Evaluate fine-tuned Qwen2.5-VL on VQA metrics.
Computes accuracy, F1 score, BLEU, ROUGE for X-ray inspection task.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge


def load_model_and_processor(model_path: str, base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """Load fine-tuned model and processor."""
    print(f"Loading model from {model_path}...")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Try loading as LoRA model first
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights
        print("Loaded as LoRA model and merged weights")
    except Exception as e:
        # Load as full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Loaded as full model")
    
    model.eval()
    return model, processor


def generate_answer(model, processor, image_path: str, question: str, max_new_tokens: int = 256):
    """Generate answer for a question given an image."""
    from PIL import Image
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Format prompt
    prompt = f"Question: {question}\nAnswer:"
    
    # Prepare inputs
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for evaluation
            temperature=1.0,
        )
    
    # Decode
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer (text after "Answer:")
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        answer = generated_text.strip()
    
    return answer


def extract_prohibited_items(text: str) -> List[str]:
    """Extract prohibited items mentioned in text."""
    prohibited_items = [
        "folding knife", "straight knife", "scissors", "scissor",
        "utility knife", "multi-tool knife", "knife", "blade"
    ]
    
    text_lower = text.lower()
    found = []
    
    for item in prohibited_items:
        if item in text_lower:
            found.append(item)
    
    return list(set(found))


def compute_exact_match(pred: str, target: str) -> float:
    """Compute exact match (case-insensitive)."""
    return float(pred.strip().lower() == target.strip().lower())


def compute_item_detection_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Compute precision, recall, F1 for item detection."""
    y_true = []
    y_pred = []
    
    for pred, gt in zip(predictions, ground_truth):
        # Extract items from both
        pred_items = set(extract_prohibited_items(pred["predicted_answer"]))
        gt_items = set(extract_prohibited_items(gt["answer"]))
        
        # Binary classification: has prohibited items or not
        y_true.append(1 if gt_items else 0)
        y_pred.append(1 if pred_items else 0)
    
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute average BLEU score."""
    smoothing = SmoothingFunction().method1
    scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0


def compute_rouge(predictions: List[str], references: List[str]) -> Dict:
    """Compute ROUGE scores."""
    rouge = Rouge()
    
    try:
        scores = rouge.get_scores(predictions, references, avg=True)
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }
    except Exception as e:
        print(f"Warning: ROUGE computation failed: {e}")
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}


def evaluate_vqa(
    model,
    processor,
    test_file: str,
    output_file: str = None,
    max_samples: int = None,
):
    """Evaluate VQA model on test set."""
    # Load test data
    test_data = []
    with open(test_file, "r") as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"Evaluating on {len(test_data)} samples...")
    
    # Generate predictions
    predictions = []
    
    for item in tqdm(test_data, desc="Generating predictions"):
        try:
            predicted_answer = generate_answer(
                model,
                processor,
                item["image_path"],
                item["question"],
            )
        except Exception as e:
            print(f"Error generating answer: {e}")
            predicted_answer = ""
        
        predictions.append({
            "image_path": item["image_path"],
            "question": item["question"],
            "ground_truth": item["answer"],
            "predicted_answer": predicted_answer,
            "metadata": item.get("metadata", {}),
        })
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # Exact match
    exact_matches = [
        compute_exact_match(p["predicted_answer"], p["ground_truth"])
        for p in predictions
    ]
    exact_match_accuracy = sum(exact_matches) / len(exact_matches)
    
    # Item detection metrics
    item_metrics = compute_item_detection_metrics(predictions, test_data)
    
    # BLEU score
    pred_texts = [p["predicted_answer"] for p in predictions]
    ref_texts = [p["ground_truth"] for p in predictions]
    bleu_score = compute_bleu(pred_texts, ref_texts)
    
    # ROUGE scores
    rouge_scores = compute_rouge(pred_texts, ref_texts)
    
    # Per-question-type metrics
    question_type_metrics = defaultdict(list)
    for p in predictions:
        qtype = p["metadata"].get("question_type", "unknown")
        question_type_metrics[qtype].append(
            compute_exact_match(p["predicted_answer"], p["ground_truth"])
        )
    
    per_qtype = {
        qtype: sum(scores) / len(scores)
        for qtype, scores in question_type_metrics.items()
    }
    
    # Compile results
    results = {
        "num_samples": len(predictions),
        "exact_match_accuracy": exact_match_accuracy,
        "item_detection": item_metrics,
        "bleu_score": bleu_score,
        "rouge_scores": rouge_scores,
        "per_question_type": per_qtype,
        "predictions": predictions,
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Num samples: {results['num_samples']}")
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"\nItem Detection:")
    print(f"  Precision: {item_metrics['precision']:.4f}")
    print(f"  Recall: {item_metrics['recall']:.4f}")
    print(f"  F1 Score: {item_metrics['f1_score']:.4f}")
    print(f"\nText Generation Quality:")
    print(f"  BLEU: {bleu_score:.4f}")
    print(f"  ROUGE-1: {rouge_scores['rouge-1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge-2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rouge-l']:.4f}")
    print(f"\nPer Question Type:")
    for qtype, acc in sorted(per_qtype.items()):
        print(f"  {qtype}: {acc:.4f}")
    print("=" * 60)
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/opixray_vqa_test.jsonl",
        help="Path to test data JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_vqa_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Base model name (for LoRA models)",
    )
    
    args = parser.parse_args()
    
    # Load model
    model, processor = load_model_and_processor(args.model, args.base_model)
    
    # Evaluate
    results = evaluate_vqa(
        model,
        processor,
        args.test_file,
        args.output,
        args.max_samples,
    )
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    # Install required packages if missing
    try:
        import nltk
        nltk.download("punkt", quiet=True)
    except:
        pass
    
    main()
