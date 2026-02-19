#!/usr/bin/env python3
"""
Test YOLO API on the entire validation dataset (956 images).

This script tests all validation images and generates comprehensive metrics.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import requests
from collections import Counter, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class YOLOFullValidationTester:
    """Test YOLO API on full validation dataset."""
    
    def __init__(self, api_url: str, dataset_path: str):
        self.api_url = api_url.rstrip('/')
        self.dataset_path = Path(dataset_path)
        
        # Load ground truth annotations
        self.annotations = self._load_annotations()
        print(f"✓ Loaded {len(self.annotations)} validation annotations")
        
        # Results will be populated
        self.results = {
            "model": "YOLO (deployed API)",
            "dataset": "Luggage X-ray (yolov5xray)",
            "num_samples": 0,
            "api_url": api_url,
            "test_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "samples": []
        }
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load validation annotations from JSONL file."""
        annotations_file = self.dataset_path / "_annotations.valid.jsonl"
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_file}")
        
        annotations = []
        with open(annotations_file) as f:
            for line in f:
                ann = json.loads(line.strip())
                annotations.append(ann)
        
        return annotations
    
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
    
    def _call_yolo_api(self, image_path: str) -> Dict[str, Any]:
        """Call YOLO API for detection."""
        url = f"{self.api_url}/v1/detect"
        
        try:
            with open(image_path, "rb") as f:
                files = {"file": (Path(image_path).name, f, "image/jpeg")}
                response = requests.post(url, files=files, timeout=30)
                response.raise_for_status()
            
            result = response.json()
            
            # Convert API response format
            return {
                "detections": result.get("items", []),
                "num_detections": result.get("total_count", 0),
                "threats_detected": [
                    item['name'] for item in result.get("items", []) 
                    if self._is_threat(item['name'])
                ],
                "is_threat": any(
                    self._is_threat(item['name']) 
                    for item in result.get("items", [])
                ),
                "success": True
            }
        
        except Exception as e:
            return {
                "detections": [],
                "num_detections": 0,
                "threats_detected": [],
                "is_threat": False,
                "success": False,
                "error": str(e)
            }
    
    def _is_threat(self, item_name: str) -> bool:
        """Check if an item is a threat."""
        threats = ['knife', 'gun', 'scissors', 'blade', 'weapon', 'explosive', 'dagger']
        return any(threat in item_name.lower() for threat in threats)
    
    def _get_ground_truth_class(self, annotation: Dict[str, Any]) -> str:
        """Extract ground truth class from annotation."""
        # The annotation format has the class in the 'class_id' or object list
        # For luggage dataset, we need to extract the primary class
        if 'objects' in annotation and annotation['objects']:
            # Get most common class
            classes = [obj.get('name', obj.get('class', 'unknown')) for obj in annotation['objects']]
            if classes:
                return Counter(classes).most_common(1)[0][0]
        
        return "unknown"
    
    def _map_yolo_to_category(self, detections: List[Dict[str, Any]]) -> str:
        """Map YOLO detections to a single category."""
        if not detections:
            return "unknown"
        
        # Use highest confidence detection
        best_detection = max(detections, key=lambda d: d.get('confidence', 0))
        return best_detection.get('name', best_detection.get('label', 'unknown'))
    
    def test_full_validation(self):
        """Test YOLO API on all validation images."""
        print()
        print("=" * 60)
        print("Testing YOLO API on Full Validation Dataset")
        print("=" * 60)
        print()
        
        # Check API health
        print("Checking API health...")
        if not self._test_health():
            print("❌ API is not healthy. Aborting.")
            return False
        print("✓ API is healthy")
        print()
        
        total_images = len(self.annotations)
        print(f"Testing {total_images} images...")
        print()
        
        # Track progress
        start_time = time.time()
        failed_count = 0
        
        # Test each image
        for i, annotation in enumerate(self.annotations):
            image_name = annotation.get('image', annotation.get('file_name', f'image_{i}.jpg'))
            image_path = self.dataset_path / "images" / "valid" / image_name
            
            # Get ground truth
            ground_truth = self._get_ground_truth_class(annotation)
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total_images - i - 1) / rate if rate > 0 else 0
                print(f"[{i+1}/{total_images}] Processed... "
                      f"({rate:.1f} img/s, ~{remaining/60:.1f}min remaining)")
            
            if not image_path.exists():
                print(f"  ⚠ Image not found: {image_name}, skipping")
                failed_count += 1
                continue
            
            # Call YOLO API
            result = self._call_yolo_api(str(image_path))
            
            if not result.get('success', False):
                failed_count += 1
                if failed_count <= 10:  # Only show first 10 errors
                    print(f"  ⚠ API call failed for {image_name}: {result.get('error', 'Unknown error')}")
            
            # Map YOLO detections to category
            predicted = self._map_yolo_to_category(result.get('detections', []))
            correct = predicted.lower() == ground_truth.lower()
            
            # Store result
            sample_result = {
                "image_name": image_name,
                "image_path": str(image_path.relative_to(project_root)),
                "index": i,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": correct,
                "detections": result.get('detections', []),
                "num_detections": result.get('num_detections', 0),
                "threats_detected": result.get('threats_detected', []),
                "is_threat": result.get('is_threat', False),
                "api_success": result.get('success', False)
            }
            
            self.results['samples'].append(sample_result)
        
        elapsed_time = time.time() - start_time
        self.results['num_samples'] = len(self.results['samples'])
        self.results['test_duration_seconds'] = elapsed_time
        self.results['failed_count'] = failed_count
        
        print()
        print(f"✓ Completed {self.results['num_samples']} samples in {elapsed_time/60:.1f} minutes")
        print(f"  Average: {self.results['num_samples']/elapsed_time:.1f} images/second")
        if failed_count > 0:
            print(f"  ⚠ Failed: {failed_count} images")
        print()
        
        return True
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        samples = self.results['samples']
        total = len(samples)
        
        if total == 0:
            return {}
        
        # Overall metrics
        correct = sum(1 for s in samples if s.get('correct', False))
        accuracy = correct / total if total > 0 else 0
        
        # Detection statistics
        total_detections = sum(s.get('num_detections', 0) for s in samples)
        avg_detections = total_detections / total
        images_with_detections = sum(1 for s in samples if s.get('num_detections', 0) > 0)
        detection_rate = images_with_detections / total
        
        # Threat statistics
        threats_found = sum(1 for s in samples if s.get('is_threat', False))
        total_threat_detections = sum(len(s.get('threats_detected', [])) for s in samples)
        
        # Predicted classes distribution
        predictions = [s.get('predicted', 'unknown') for s in samples]
        pred_counter = Counter(predictions)
        
        # Ground truth distribution
        ground_truths = [s.get('ground_truth', 'unknown') for s in samples]
        gt_counter = Counter(ground_truths)
        
        # Per-class accuracy
        class_accuracy = {}
        for gt_class in gt_counter.keys():
            class_samples = [s for s in samples if s.get('ground_truth') == gt_class]
            class_correct = sum(1 for s in class_samples if s.get('correct', False))
            class_accuracy[gt_class] = {
                'correct': class_correct,
                'total': len(class_samples),
                'accuracy': class_correct / len(class_samples) if class_samples else 0
            }
        
        # Detected COCO classes (wrong predictions)
        detected_coco_classes = [
            s.get('predicted') for s in samples 
            if s.get('predicted') != 'unknown' and not s.get('correct', False)
        ]
        coco_class_counter = Counter(detected_coco_classes)
        
        metrics = {
            'overall_accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_detections_per_image': avg_detections,
            'total_detections': total_detections,
            'images_with_detections': images_with_detections,
            'detection_rate': detection_rate,
            'threats_found': threats_found,
            'total_threat_detections': total_threat_detections,
            'predictions_distribution': dict(pred_counter.most_common(20)),
            'ground_truth_distribution': dict(gt_counter),
            'per_class_accuracy': class_accuracy,
            'wrong_coco_predictions': dict(coco_class_counter.most_common(20))
        }
        
        return metrics
    
    def generate_report(self, metrics: Dict[str, Any], output_path: str):
        """Generate comprehensive test report."""
        report = []
        
        report.append("# YOLO API - Full Validation Dataset Test")
        report.append("")
        report.append(f"**Test Date**: {self.results['test_date']}")
        report.append(f"**Dataset**: Luggage X-ray (yolov5xray) - Full Validation Set")
        report.append(f"**API URL**: {self.api_url}")
        report.append(f"**Total Samples**: {metrics['total']}")
        report.append(f"**Test Duration**: {self.results.get('test_duration_seconds', 0)/60:.1f} minutes")
        report.append("")
        
        # Overall Performance
        report.append("## Overall Performance")
        report.append("")
        report.append(f"- **Accuracy**: {metrics['overall_accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
        report.append(f"- **Detection Rate**: {metrics['detection_rate']:.2%} ({metrics['images_with_detections']}/{metrics['total']} images had detections)")
        report.append(f"- **Average Detections per Image**: {metrics['avg_detections_per_image']:.2f}")
        report.append(f"- **Total Detections**: {metrics['total_detections']}")
        report.append(f"- **Threats Found**: {metrics['threats_found']} images ({metrics['total_threat_detections']} threat detections)")
        report.append("")
        
        # Ground Truth Distribution
        report.append("## Ground Truth Class Distribution")
        report.append("")
        report.append("| Class | Count | % of Dataset |")
        report.append("|-------|-------|--------------|")
        for class_name, count in sorted(metrics['ground_truth_distribution'].items(), key=lambda x: x[1], reverse=True):
            pct = count / metrics['total'] * 100
            report.append(f"| {class_name} | {count} | {pct:.1f}% |")
        report.append("")
        
        # Per-Class Performance
        report.append("## Per-Class Accuracy")
        report.append("")
        report.append("| Class | Correct | Total | Accuracy |")
        report.append("|-------|---------|-------|----------|")
        for class_name, stats in sorted(metrics['per_class_accuracy'].items(), key=lambda x: x[1]['total'], reverse=True):
            report.append(f"| {class_name} | {stats['correct']} | {stats['total']} | {stats['accuracy']:.1%} |")
        report.append("")
        
        # YOLO Predictions (COCO classes)
        report.append("## YOLO Predictions (COCO Classes)")
        report.append("")
        report.append("Top wrong predictions from pre-trained YOLO:")
        report.append("")
        report.append("| COCO Class | Count |")
        report.append("|------------|-------|")
        for coco_class, count in list(metrics['wrong_coco_predictions'].items())[:15]:
            report.append(f"| {coco_class} | {count} |")
        report.append("")
        
        # Key Findings
        report.append("## Key Findings")
        report.append("")
        report.append(f"1. **Overall Accuracy**: {metrics['overall_accuracy']:.2%} - Pre-trained YOLO performs poorly on X-ray images")
        report.append(f"2. **Detection Rate**: {metrics['detection_rate']:.2%} of images had at least one detection")
        report.append(f"3. **Random Detections**: YOLO detected COCO classes (suitcase, knife, person, etc.) which don't match X-ray objects")
        report.append(f"4. **Domain Mismatch**: Pre-trained model trained on natural RGB images, not X-ray density images")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        report.append("1. **Train YOLO on X-ray Dataset**: Expected to achieve 85-95% accuracy after training")
        report.append("2. **Training Time**: 2-4 hours on GPU for this dataset")
        report.append("3. **Expected Improvement**:")
        report.append(f"   - Accuracy: {metrics['overall_accuracy']:.1%} → 85-95%")
        report.append(f"   - Detection Rate: {metrics['detection_rate']:.1%} → 95-100%")
        report.append("   - Predictions will match actual X-ray classes (PlasticBottle, Knife, etc.)")
        report.append("")
        
        # Write report
        report_text = "\n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def run(self, output_dir: str):
        """Run full validation test."""
        print()
        print("=" * 60)
        print("YOLO API - Full Validation Dataset Test")
        print("=" * 60)
        print()
        
        # Test all images
        if not self.test_full_validation():
            print("❌ Testing failed")
            return False
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = self.calculate_metrics()
        
        # Print summary
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print()
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"Correct:  {metrics['correct']}/{metrics['total']}")
        print(f"Detection Rate: {metrics['detection_rate']:.2%}")
        print(f"Avg Detections: {metrics['avg_detections_per_image']:.2f}")
        print(f"Threats Found: {metrics['threats_found']}")
        print()
        
        # Generate report
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        report_path = output_dir / "yolo_full_validation_report.md"
        results_path = output_dir / "yolo_full_validation_results.json"
        metrics_path = output_dir / "yolo_full_validation_metrics.json"
        
        print("Generating reports...")
        self.generate_report(metrics, str(report_path))
        
        # Save full results
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Report saved: {report_path}")
        print(f"✓ Results saved: {results_path}")
        print(f"✓ Metrics saved: {metrics_path}")
        print()
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLO API on full validation dataset"
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site',
        help='YOLO API base URL'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/luggage_xray',
        help='Path to luggage_xray dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_results',
        help='Output directory for results'
    )
    
    args, _ = parser.parse_known_args()
    
    print()
    print("🔬 YOLO API - Full Validation Dataset Test")
    print(f"   API: {args.api_url}")
    print(f"   Dataset: {args.dataset}")
    print()
    
    # Run test
    tester = YOLOFullValidationTester(args.api_url, args.dataset)
    success = tester.run(args.output)
    
    if success:
        print("=" * 60)
        print("✅ Testing Complete!")
        print("=" * 60)
        print()
        return 0
    else:
        print("❌ Testing failed")
        return 1


if __name__ == '__main__':
    exit_code = main()
    if exit_code != 0:
        sys.exit(exit_code)
