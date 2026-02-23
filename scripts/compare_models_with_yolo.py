#!/usr/bin/env python3
"""
Compare YOLO API results with GPT-4 and RolmOCR on the same test images.

This script:
1. Loads existing GPT-4 and RolmOCR test results
2. Tests the same images with the deployed YOLO API
3. Compares predictions and metrics
4. Generates a comprehensive comparison report

Usage:
    python scripts/compare_models_with_yolo.py
    python scripts/compare_models_with_yolo.py --num-samples 50
    python scripts/compare_models_with_yolo.py --api-url https://xray-yolo-api.[domain]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from collections import Counter, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ModelComparator:
    """Compare YOLO API with GPT-4 and RolmOCR results."""
    
    def __init__(self, api_url: str, num_samples: Optional[int] = None):
        self.api_url = api_url.rstrip('/')
        self.num_samples = num_samples
        
        # Load existing results
        self.gpt4_results = self._load_results("test_results/gpt4_luggage/gpt4_luggage_results.json")
        self.rolm_results = self._load_results("test_results/rolmocr_luggage/rolmocr_luggage_results.json")
        
        # YOLO results will be populated
        self.yolo_results = {
            "model": "YOLO (deployed API)",
            "dataset": "Luggage X-ray (yolov5xray)",
            "num_samples": 0,
            "api_url": api_url,
            "samples": []
        }
        
        print(f"✓ Loaded GPT-4 results: {self.gpt4_results['num_samples']} samples")
        print(f"✓ Loaded RolmOCR results: {self.rolm_results['num_samples']} samples")
    
    def _load_results(self, path: str) -> Dict[str, Any]:
        """Load existing test results."""
        full_path = project_root / path
        if not full_path.exists():
            print(f"Warning: Results not found: {path}")
            return {"samples": [], "num_samples": 0}
        
        with open(full_path) as f:
            return json.load(f)
    
    def _test_health(self) -> bool:
        """Test API health."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            response.raise_for_status()
            health = response.json()
            return health.get("status") == "healthy" and health.get("model_loaded")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def _call_yolo_api(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Call YOLO API for detection."""
        url = f"{self.api_url}/v1/detect"
        
        try:
            with open(image_path, "rb") as f:
                files = {"file": (Path(image_path).name, f, "image/jpeg")}
                response = requests.post(url, files=files, timeout=30)
                response.raise_for_status()
            
            result = response.json()
            
            # Convert API response format to internal format
            # API returns: {"items": [...], "total_count": N, "has_concealed_items": bool}
            # Convert to: {"detections": [...], "num_detections": N}
            return {
                "detections": result.get("items", []),
                "num_detections": result.get("total_count", 0),
                "threats_detected": [item['name'] for item in result.get("items", []) if self._is_threat(item['name'])],
                "is_threat": any(self._is_threat(item['name']) for item in result.get("items", []))
            }
        
        except Exception as e:
            print(f"  ⚠ API call failed for {Path(image_path).name}: {e}")
            return None
    
    def _is_threat(self, item_name: str) -> bool:
        """Check if an item is a threat."""
        threats = ['knife', 'gun', 'scissors', 'blade', 'weapon', 'explosive', 'dagger']
        return any(threat in item_name.lower() for threat in threats)
    
    def _map_yolo_to_category(self, detections: List[Dict[str, Any]]) -> str:
        """Map YOLO detections to a single category for comparison."""
        if not detections:
            return "unknown"
        
        # Use highest confidence detection
        # API returns 'name' not 'label'
        best_detection = max(detections, key=lambda d: d.get('confidence', 0))
        return best_detection.get('name', best_detection.get('label', 'unknown'))
    
    def test_yolo_on_samples(self):
        """Test YOLO API on the same images used for GPT-4/RolmOCR."""
        print()
        print("=" * 60)
        print("Testing YOLO API on Sample Images")
        print("=" * 60)
        print()
        
        # Check API health
        print("Checking API health...")
        if not self._test_health():
            print("❌ API is not healthy. Aborting.")
            return False
        print("✓ API is healthy")
        print()
        
        # Get samples to test (use GPT-4 samples as reference)
        samples = self.gpt4_results.get('samples', [])
        if not samples:
            print("❌ No GPT-4 samples found")
            return False
        
        # Limit samples if requested
        if self.num_samples:
            samples = samples[:self.num_samples]
        
        print(f"Testing {len(samples)} images...")
        print()
        
        # Test each image
        for i, sample in enumerate(samples):
            image_path = project_root / sample['image_path']
            image_name = sample['image_name']
            ground_truth = sample['ground_truth']
            
            print(f"[{i+1}/{len(samples)}] {image_name} (GT: {ground_truth})...", end=" ", flush=True)
            
            if not image_path.exists():
                print(f"⚠ Image not found, skipping")
                continue
            
            # Call YOLO API
            result = self._call_yolo_api(str(image_path))
            
            if result is None:
                print("✗ API call failed")
                continue
            
            # Map YOLO detections to category
            predicted = self._map_yolo_to_category(result.get('detections', []))
            correct = predicted.lower() == ground_truth.lower()
            
            # Store result
            yolo_sample = {
                "image_name": image_name,
                "image_path": sample['image_path'],
                "index": i,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": correct,
                "detections": result.get('detections', []),
                "num_detections": result.get('num_detections', 0),
                "threats_detected": result.get('threats_detected', []),
                "is_threat": result.get('is_threat', False)
            }
            
            self.yolo_results['samples'].append(yolo_sample)
            
            status = "✓" if correct else "✗"
            print(f"{status} Predicted: {predicted} ({result.get('num_detections', 0)} detections)")
            
            # Small delay to avoid overwhelming API
            time.sleep(0.5)
        
        self.yolo_results['num_samples'] = len(self.yolo_results['samples'])
        print()
        print(f"✓ Completed {self.yolo_results['num_samples']} samples")
        
        return True
    
    def calculate_metrics(self):
        """Calculate accuracy and metrics for all models."""
        print()
        print("=" * 60)
        print("Calculating Metrics")
        print("=" * 60)
        print()
        
        results = {
            "GPT-4o": self.gpt4_results,
            "RolmOCR": self.rolm_results,
            "YOLO API": self.yolo_results
        }
        
        metrics = {}
        
        for model_name, data in results.items():
            samples = data.get('samples', [])
            if not samples:
                continue
            
            correct = sum(1 for s in samples if s.get('correct', False))
            total = len(samples)
            accuracy = correct / total if total > 0 else 0
            
            # Count predictions
            predictions = [s.get('predicted', 'unknown') for s in samples]
            pred_counter = Counter(predictions)
            
            # Threat detection metrics (for YOLO)
            if model_name == "YOLO API":
                threats = sum(1 for s in samples if s.get('is_threat', False))
                avg_detections = sum(s.get('num_detections', 0) for s in samples) / total if total > 0 else 0
                
                metrics[model_name] = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "predictions": pred_counter,
                    "threats_detected": threats,
                    "avg_detections": avg_detections
                }
            else:
                metrics[model_name] = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "predictions": pred_counter
                }
        
        return metrics
    
    def generate_report(self, metrics: Dict[str, Any], output_path: str):
        """Generate comparison report in markdown."""
        report = []
        
        report.append("# YOLO API vs GPT-4 vs RolmOCR - Comparison Report")
        report.append("")
        report.append(f"**Test Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Dataset**: Luggage X-ray (yolov5xray)")
        report.append(f"**API URL**: {self.api_url}")
        report.append("")
        
        # Overall metrics comparison
        report.append("## Overall Performance")
        report.append("")
        report.append("| Model | Accuracy | Correct | Total | Avg Detections | Threats Found |")
        report.append("|-------|----------|---------|-------|----------------|---------------|")
        
        for model_name, data in metrics.items():
            accuracy = f"{data['accuracy']:.1%}"
            correct = data['correct']
            total = data['total']
            avg_det = f"{data.get('avg_detections', 0):.1f}" if 'avg_detections' in data else "N/A"
            threats = data.get('threats_detected', 'N/A')
            
            report.append(f"| {model_name} | {accuracy} | {correct}/{total} | {total} | {avg_det} | {threats} |")
        
        report.append("")
        
        # Per-category breakdown
        report.append("## Category-wise Comparison")
        report.append("")
        report.append("Top predicted categories by each model:")
        report.append("")
        
        for model_name, data in metrics.items():
            report.append(f"### {model_name}")
            report.append("")
            
            predictions = data['predictions'].most_common(10)
            if predictions:
                report.append("| Category | Count |")
                report.append("|----------|-------|")
                for cat, count in predictions:
                    report.append(f"| {cat} | {count} |")
            else:
                report.append("No predictions available")
            
            report.append("")
        
        # Sample-by-sample comparison
        report.append("## Sample-by-Sample Results")
        report.append("")
        report.append("First 20 samples compared:")
        report.append("")
        report.append("| # | Image | Ground Truth | GPT-4o | RolmOCR | YOLO API |")
        report.append("|---|-------|--------------|--------|---------|----------|")
        
        num_compare = min(20, len(self.yolo_results.get('samples', [])))
        
        for i in range(num_compare):
            yolo_s = self.yolo_results['samples'][i]
            gpt4_s = self.gpt4_results['samples'][i]
            rolm_s = self.rolm_results['samples'][i]
            
            gt = yolo_s['ground_truth']
            yolo_pred = yolo_s['predicted']
            gpt4_pred = gpt4_s['predicted']
            rolm_pred = rolm_s['predicted']
            
            # Add checkmarks for correct predictions
            def mark(pred, gt):
                return f"**{pred}** ✓" if pred.lower() == gt.lower() else pred
            
            report.append(f"| {i+1} | {yolo_s['image_name'][:20]} | {gt} | {mark(gpt4_pred, gt)} | {mark(rolm_pred, gt)} | {mark(yolo_pred, gt)} |")
        
        report.append("")
        
        # Key findings
        report.append("## Key Findings")
        report.append("")
        
        yolo_acc = metrics.get("YOLO API", {}).get("accuracy", 0)
        gpt4_acc = metrics.get("GPT-4o", {}).get("accuracy", 0)
        rolm_acc = metrics.get("RolmOCR", {}).get("accuracy", 0)
        
        best_model = max(
            [("YOLO API", yolo_acc), ("GPT-4o", gpt4_acc), ("RolmOCR", rolm_acc)],
            key=lambda x: x[1]
        )[0]
        
        report.append(f"1. **Best Accuracy**: {best_model} ({max(yolo_acc, gpt4_acc, rolm_acc):.1%})")
        report.append("")
        
        if "YOLO API" in metrics:
            avg_det = metrics["YOLO API"].get("avg_detections", 0)
            threats = metrics["YOLO API"].get("threats_detected", 0)
            report.append(f"2. **YOLO Detections**: Average {avg_det:.1f} objects per image, {threats} threats found")
        
        report.append("")
        report.append(f"3. **Comparison**:")
        report.append(f"   - GPT-4o: {gpt4_acc:.1%} accuracy (VLM, detailed descriptions)")
        report.append(f"   - RolmOCR: {rolm_acc:.1%} accuracy (OCR-based VLM)")
        report.append(f"   - YOLO API: {yolo_acc:.1%} accuracy (Object detection, fast inference)")
        report.append("")
        
        # Performance notes
        report.append("## Notes")
        report.append("")
        report.append("- **YOLO Model**: Pre-trained yolov8x.pt (COCO dataset, not X-ray specific)")
        report.append("- **Expected**: YOLO accuracy may be lower as it's not trained on X-ray images")
        report.append("- **Recommendation**: Train YOLO on luggage X-ray dataset for better accuracy")
        report.append("- **Performance**: YOLO inference is much faster (<100ms vs 5-10s for VLMs)")
        report.append("")
        
        # Threat detection comparison
        if "YOLO API" in metrics:
            report.append("## Threat Detection")
            report.append("")
            report.append("YOLO API detected the following threats:")
            report.append("")
            
            threat_samples = [s for s in self.yolo_results['samples'] if s.get('is_threat', False)]
            if threat_samples:
                report.append("| Image | Ground Truth | Threats Detected |")
                report.append("|-------|--------------|------------------|")
                for s in threat_samples[:10]:  # First 10
                    threats_str = ", ".join(s['threats_detected'])
                    report.append(f"| {s['image_name']} | {s['ground_truth']} | {threats_str} |")
            else:
                report.append("No threats detected in the test set.")
            
            report.append("")
        
        # Write report
        report_text = "\n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def run(self, output_path: str):
        """Run full comparison."""
        print()
        print("=" * 60)
        print("Model Comparison: YOLO API vs GPT-4 vs RolmOCR")
        print("=" * 60)
        print()
        
        # Test YOLO on same images
        if not self.test_yolo_on_samples():
            print("❌ YOLO testing failed")
            return False
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Print summary
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print()
        
        for model_name, data in metrics.items():
            print(f"{model_name}:")
            print(f"  Accuracy: {data['accuracy']:.1%}")
            print(f"  Correct:  {data['correct']}/{data['total']}")
            if 'avg_detections' in data:
                print(f"  Avg Detections: {data['avg_detections']:.1f}")
                print(f"  Threats Found: {data['threats_detected']}")
            print()
        
        # Generate report
        print("Generating comparison report...")
        report = self.generate_report(metrics, output_path)
        
        # Save YOLO results
        yolo_results_path = output_path.replace('.md', '_yolo_results.json')
        with open(yolo_results_path, 'w') as f:
            json.dump(self.yolo_results, f, indent=2)
        
        print(f"✓ Report saved: {output_path}")
        print(f"✓ YOLO results saved: {yolo_results_path}")
        print()
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLO API with GPT-4 and RolmOCR"
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site',
        help='YOLO API base URL'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to test (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_results/model_comparison_with_yolo.md',
        help='Output report path'
    )
    
    args, _ = parser.parse_known_args()
    
    print()
    print("🔬 YOLO API Model Comparison")
    print(f"   API: {args.api_url}")
    print(f"   Samples: {args.num_samples or 'All available'}")
    print()
    
    # Run comparison
    comparator = ModelComparator(args.api_url, args.num_samples)
    success = comparator.run(args.output)
    
    if success:
        print("=" * 60)
        print("✅ Comparison Complete!")
        print("=" * 60)
        print()
        print(f"View report: {args.output}")
        print()
        return 0
    else:
        print("❌ Comparison failed")
        return 1


if __name__ == '__main__':
    exit_code = main()
    if exit_code != 0:
        sys.exit(exit_code)
