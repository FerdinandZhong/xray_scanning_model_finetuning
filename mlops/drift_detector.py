#!/usr/bin/env python3
"""
Drift detection for X-ray inspection model.
Monitors for input drift, prediction drift, and concept drift.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
from scipy import stats


class DriftDetector:
    """Detect various types of drift in model predictions."""
    
    def __init__(self, baseline_data: Dict = None):
        """
        Initialize drift detector.
        
        Args:
            baseline_data: Baseline statistics for comparison
        """
        self.baseline_data = baseline_data or {}
    
    def compute_image_statistics(self, images: List[np.ndarray]) -> Dict:
        """
        Compute statistical features of images.
        
        Args:
            images: List of image arrays (H, W, C)
        
        Returns:
            Dictionary of image statistics
        """
        stats_dict = {
            "mean_brightness": [],
            "std_brightness": [],
            "mean_contrast": [],
            "num_images": len(images),
        }
        
        for img in images:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            
            stats_dict["mean_brightness"].append(np.mean(gray))
            stats_dict["std_brightness"].append(np.std(gray))
            stats_dict["mean_contrast"].append(np.max(gray) - np.min(gray))
        
        # Aggregate
        return {
            "mean_brightness": float(np.mean(stats_dict["mean_brightness"])),
            "std_brightness": float(np.mean(stats_dict["std_brightness"])),
            "mean_contrast": float(np.mean(stats_dict["mean_contrast"])),
            "num_images": stats_dict["num_images"],
        }
    
    def detect_input_drift(
        self,
        current_stats: Dict,
        threshold: float = 0.05,
    ) -> Tuple[bool, Dict]:
        """
        Detect input drift using statistical tests.
        
        Args:
            current_stats: Current image statistics
            threshold: P-value threshold for drift detection
        
        Returns:
            (has_drift, drift_details)
        """
        if not self.baseline_data:
            return False, {"message": "No baseline data for comparison"}
        
        baseline_stats = self.baseline_data.get("image_stats", {})
        
        drift_detected = False
        drift_details = {}
        
        # Compare brightness
        baseline_brightness = baseline_stats.get("mean_brightness", 0)
        current_brightness = current_stats.get("mean_brightness", 0)
        brightness_diff = abs(current_brightness - baseline_brightness)
        brightness_pct_change = (
            brightness_diff / baseline_brightness * 100
            if baseline_brightness > 0 else 0
        )
        
        if brightness_pct_change > 10:  # >10% change
            drift_detected = True
            drift_details["brightness_drift"] = {
                "baseline": baseline_brightness,
                "current": current_brightness,
                "pct_change": brightness_pct_change,
            }
        
        # Compare contrast
        baseline_contrast = baseline_stats.get("mean_contrast", 0)
        current_contrast = current_stats.get("mean_contrast", 0)
        contrast_diff = abs(current_contrast - baseline_contrast)
        contrast_pct_change = (
            contrast_diff / baseline_contrast * 100
            if baseline_contrast > 0 else 0
        )
        
        if contrast_pct_change > 15:  # >15% change
            drift_detected = True
            drift_details["contrast_drift"] = {
                "baseline": baseline_contrast,
                "current": current_contrast,
                "pct_change": contrast_pct_change,
            }
        
        return drift_detected, drift_details
    
    def detect_prediction_drift(
        self,
        current_predictions: List[Dict],
        threshold: float = 0.05,
    ) -> Tuple[bool, Dict]:
        """
        Detect drift in prediction distributions.
        
        Args:
            current_predictions: List of prediction dictionaries
            threshold: P-value threshold for KS test
        
        Returns:
            (has_drift, drift_details)
        """
        if not self.baseline_data:
            return False, {"message": "No baseline data for comparison"}
        
        baseline_preds = self.baseline_data.get("predictions", [])
        
        if not baseline_preds:
            return False, {"message": "No baseline predictions"}
        
        # Extract risk level distributions
        baseline_risks = [p.get("risk_level", "low") for p in baseline_preds]
        current_risks = [p.get("risk_level", "low") for p in current_predictions]
        
        # Convert to numeric for statistical test
        risk_mapping = {"low": 0, "medium": 1, "high": 2}
        baseline_numeric = [risk_mapping.get(r, 0) for r in baseline_risks]
        current_numeric = [risk_mapping.get(r, 0) for r in current_risks]
        
        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(baseline_numeric, current_numeric)
        
        drift_detected = p_value < threshold
        
        # Compute distribution changes
        baseline_dist = {
            level: baseline_risks.count(level) / len(baseline_risks)
            for level in ["low", "medium", "high"]
        }
        current_dist = {
            level: current_risks.count(level) / len(current_risks)
            for level in ["low", "medium", "high"]
        }
        
        drift_details = {
            "ks_statistic": float(ks_statistic),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "baseline_distribution": baseline_dist,
            "current_distribution": current_dist,
        }
        
        return drift_detected, drift_details
    
    def detect_concept_drift(
        self,
        feedback_data: List[Dict],
        window_size: int = 100,
        threshold: float = 0.1,
    ) -> Tuple[bool, Dict]:
        """
        Detect concept drift using feedback data.
        
        Args:
            feedback_data: List of {prediction, ground_truth} dictionaries
            window_size: Window size for drift detection
            threshold: Accuracy drop threshold
        
        Returns:
            (has_drift, drift_details)
        """
        if len(feedback_data) < window_size:
            return False, {"message": f"Insufficient feedback data (need {window_size})"}
        
        # Compute accuracy over sliding windows
        accuracies = []
        
        for i in range(len(feedback_data) - window_size + 1):
            window = feedback_data[i:i + window_size]
            correct = sum(
                1 for item in window
                if item.get("prediction") == item.get("ground_truth")
            )
            accuracy = correct / window_size
            accuracies.append(accuracy)
        
        # Detect significant drop
        baseline_accuracy = np.mean(accuracies[:len(accuracies)//2]) if len(accuracies) > 1 else 0
        current_accuracy = np.mean(accuracies[len(accuracies)//2:]) if len(accuracies) > 1 else 0
        
        accuracy_drop = baseline_accuracy - current_accuracy
        drift_detected = accuracy_drop > threshold
        
        drift_details = {
            "baseline_accuracy": float(baseline_accuracy),
            "current_accuracy": float(current_accuracy),
            "accuracy_drop": float(accuracy_drop),
            "drift_detected": drift_detected,
            "window_size": window_size,
        }
        
        return drift_detected, drift_details
    
    def psi_score(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            expected: Baseline distribution
            actual: Current distribution
            bins: Number of bins for discretization
        
        Returns:
            PSI score (>0.2 indicates significant drift)
        """
        # Bin the data
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            bins + 1
        )
        
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        # Normalize
        expected_percents = expected_counts / len(expected)
        actual_percents = actual_counts / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi = np.sum(
            (actual_percents - expected_percents) *
            np.log(actual_percents / expected_percents)
        )
        
        return float(psi)
    
    def generate_drift_report(
        self,
        current_data: Dict,
        output_file: str = None,
    ) -> Dict:
        """
        Generate comprehensive drift report.
        
        Args:
            current_data: Current monitoring data
            output_file: Optional output file path
        
        Returns:
            Drift report dictionary
        """
        report = {
            "timestamp": current_data.get("timestamp"),
            "drifts_detected": [],
        }
        
        # Input drift
        if "image_stats" in current_data:
            has_drift, details = self.detect_input_drift(current_data["image_stats"])
            report["input_drift"] = {
                "detected": has_drift,
                "details": details,
            }
            if has_drift:
                report["drifts_detected"].append("input")
        
        # Prediction drift
        if "predictions" in current_data:
            has_drift, details = self.detect_prediction_drift(current_data["predictions"])
            report["prediction_drift"] = {
                "detected": has_drift,
                "details": details,
            }
            if has_drift:
                report["drifts_detected"].append("prediction")
        
        # Concept drift
        if "feedback" in current_data:
            has_drift, details = self.detect_concept_drift(current_data["feedback"])
            report["concept_drift"] = {
                "detected": has_drift,
                "details": details,
            }
            if has_drift:
                report["drifts_detected"].append("concept")
        
        # Save report
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            
            print(f"Drift report saved to: {output_file}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Drift detection for X-ray inspection model")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline data JSON file",
    )
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Current monitoring data JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics/drift_report.json",
        help="Output drift report file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="P-value threshold for drift detection",
    )
    
    args = parser.parse_args()
    
    # Load baseline
    with open(args.baseline, "r") as f:
        baseline_data = json.load(f)
    
    # Load current data
    with open(args.current, "r") as f:
        current_data = json.load(f)
    
    # Initialize detector
    detector = DriftDetector(baseline_data)
    
    # Generate report
    print("Analyzing drift...")
    report = detector.generate_drift_report(current_data, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Drift Detection Summary")
    print("=" * 60)
    
    if report["drifts_detected"]:
        print(f"⚠ Drifts detected: {', '.join(report['drifts_detected'])}")
        
        for drift_type in report["drifts_detected"]:
            print(f"\n{drift_type.upper()} DRIFT:")
            drift_info = report.get(f"{drift_type}_drift", {})
            print(json.dumps(drift_info.get("details", {}), indent=2))
    else:
        print("✓ No significant drift detected")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
