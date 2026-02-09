#!/usr/bin/env python3
"""
Monitoring service for X-ray inspection model.
Tracks performance metrics, latencies, and prediction distributions.
Exports metrics in Prometheus format.
"""

import argparse
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np


@dataclass
class MetricsCollector:
    """Collects and aggregates model performance metrics."""
    
    # Latency tracking (sliding window)
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Prediction counts
    total_predictions: int = 0
    risk_level_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    action_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Error tracking
    total_errors: int = 0
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Feedback tracking (from human inspectors)
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # GPU metrics (if available)
    gpu_utilization: deque = field(default_factory=lambda: deque(maxlen=100))
    gpu_memory_used: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_prediction(
        self,
        latency_ms: float,
        risk_level: str,
        recommended_action: str,
        error: Optional[str] = None,
    ):
        """Record a prediction."""
        self.total_predictions += 1
        self.latencies.append(latency_ms)
        self.risk_level_counts[risk_level] += 1
        self.action_counts[recommended_action] += 1
        
        if error:
            self.total_errors += 1
            self.error_types[error] += 1
    
    def record_feedback(self, prediction: str, ground_truth: str):
        """Record human feedback on prediction."""
        # Simplified binary classification: threat vs no-threat
        pred_threat = prediction.lower() in ["high", "medium"]
        gt_threat = ground_truth.lower() in ["high", "medium"]
        
        if pred_threat and gt_threat:
            self.true_positives += 1
        elif pred_threat and not gt_threat:
            self.false_positives += 1
        elif not pred_threat and gt_threat:
            self.false_negatives += 1
        else:
            self.true_negatives += 1
    
    def record_gpu_metrics(self, utilization: float, memory_used_gb: float):
        """Record GPU metrics."""
        self.gpu_utilization.append(utilization)
        self.gpu_memory_used.append(memory_used_gb)
    
    def get_latency_stats(self) -> Dict:
        """Get latency statistics."""
        if not self.latencies:
            return {}
        
        latencies_array = np.array(self.latencies)
        return {
            "mean_ms": float(np.mean(latencies_array)),
            "median_ms": float(np.median(latencies_array)),
            "p50_ms": float(np.percentile(latencies_array, 50)),
            "p95_ms": float(np.percentile(latencies_array, 95)),
            "p99_ms": float(np.percentile(latencies_array, 99)),
            "min_ms": float(np.min(latencies_array)),
            "max_ms": float(np.max(latencies_array)),
        }
    
    def get_accuracy_metrics(self) -> Dict:
        """Get accuracy metrics from feedback."""
        total_feedback = (
            self.true_positives + self.false_positives +
            self.true_negatives + self.false_negatives
        )
        
        if total_feedback == 0:
            return {}
        
        accuracy = (self.true_positives + self.true_negatives) / total_feedback
        
        # Precision and recall
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0 else 0
        )
        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0 else 0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )
        
        # False positive rate
        fpr = (
            self.false_positives / (self.false_positives + self.true_negatives)
            if (self.false_positives + self.true_negatives) > 0 else 0
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }
    
    def get_gpu_stats(self) -> Dict:
        """Get GPU statistics."""
        if not self.gpu_utilization:
            return {}
        
        return {
            "utilization_mean": float(np.mean(self.gpu_utilization)),
            "utilization_max": float(np.max(self.gpu_utilization)),
            "memory_used_mean_gb": float(np.mean(self.gpu_memory_used)),
            "memory_used_max_gb": float(np.max(self.gpu_memory_used)),
        }
    
    def get_summary(self) -> Dict:
        """Get complete metrics summary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / self.total_predictions if self.total_predictions > 0 else 0,
            "latency": self.get_latency_stats(),
            "accuracy": self.get_accuracy_metrics(),
            "risk_level_distribution": dict(self.risk_level_counts),
            "action_distribution": dict(self.action_counts),
            "error_types": dict(self.error_types),
            "gpu": self.get_gpu_stats(),
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Total predictions
        lines.append(f"# HELP xray_predictions_total Total number of predictions")
        lines.append(f"# TYPE xray_predictions_total counter")
        lines.append(f"xray_predictions_total {self.total_predictions}")
        
        # Error rate
        error_rate = self.total_errors / self.total_predictions if self.total_predictions > 0 else 0
        lines.append(f"# HELP xray_error_rate Prediction error rate")
        lines.append(f"# TYPE xray_error_rate gauge")
        lines.append(f"xray_error_rate {error_rate}")
        
        # Latency
        latency_stats = self.get_latency_stats()
        if latency_stats:
            lines.append(f"# HELP xray_latency_ms Prediction latency in milliseconds")
            lines.append(f"# TYPE xray_latency_ms summary")
            lines.append(f'xray_latency_ms{{quantile="0.5"}} {latency_stats["p50_ms"]}')
            lines.append(f'xray_latency_ms{{quantile="0.95"}} {latency_stats["p95_ms"]}')
            lines.append(f'xray_latency_ms{{quantile="0.99"}} {latency_stats["p99_ms"]}')
        
        # Risk levels
        lines.append(f"# HELP xray_risk_level_count Count by risk level")
        lines.append(f"# TYPE xray_risk_level_count gauge")
        for level, count in self.risk_level_counts.items():
            lines.append(f'xray_risk_level_count{{level="{level}"}} {count}')
        
        # Accuracy metrics
        accuracy_metrics = self.get_accuracy_metrics()
        if accuracy_metrics:
            lines.append(f"# HELP xray_accuracy Model accuracy from feedback")
            lines.append(f"# TYPE xray_accuracy gauge")
            lines.append(f"xray_accuracy {accuracy_metrics['accuracy']}")
            
            lines.append(f"# HELP xray_precision Model precision")
            lines.append(f"# TYPE xray_precision gauge")
            lines.append(f"xray_precision {accuracy_metrics['precision']}")
            
            lines.append(f"# HELP xray_recall Model recall")
            lines.append(f"# TYPE xray_recall gauge")
            lines.append(f"xray_recall {accuracy_metrics['recall']}")
        
        return "\n".join(lines)
    
    def save_to_file(self, filepath: str):
        """Save metrics to JSON file."""
        summary = self.get_summary()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Metrics saved to: {filepath}")


def simulate_monitoring(duration_seconds: int = 60):
    """Simulate monitoring with sample data."""
    collector = MetricsCollector()
    
    print("Simulating monitoring...")
    print(f"Duration: {duration_seconds} seconds")
    
    import random
    
    for i in range(duration_seconds):
        # Simulate predictions
        num_predictions = random.randint(1, 5)
        
        for _ in range(num_predictions):
            latency = random.gauss(350, 50)  # ~350ms mean
            risk_level = random.choice(["low", "low", "low", "medium", "high"])
            action = "PHYSICAL_INSPECTION" if risk_level == "high" else "CLEAR"
            
            collector.record_prediction(latency, risk_level, action)
        
        # Simulate feedback
        if random.random() < 0.1:  # 10% feedback rate
            pred = random.choice(["high", "low"])
            gt = pred if random.random() < 0.85 else ("low" if pred == "high" else "high")
            collector.record_feedback(pred, gt)
        
        # Simulate GPU metrics
        collector.record_gpu_metrics(
            utilization=random.uniform(70, 95),
            memory_used_gb=random.uniform(18, 22),
        )
        
        time.sleep(0.1)  # Simulate time passing
    
    return collector


def main():
    parser = argparse.ArgumentParser(description="Monitoring service for X-ray inspection")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulation mode",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Simulation duration in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics/monitoring_summary.json",
        help="Output file for metrics",
    )
    parser.add_argument(
        "--prometheus-output",
        type=str,
        default="metrics/prometheus_metrics.txt",
        help="Prometheus metrics output file",
    )
    
    args = parser.parse_args()
    
    if args.simulate:
        # Simulation mode
        collector = simulate_monitoring(args.duration)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Monitoring Summary")
        print("=" * 60)
        
        summary = collector.get_summary()
        print(json.dumps(summary, indent=2))
        
        # Save metrics
        collector.save_to_file(args.output)
        
        # Export Prometheus format
        prom_metrics = collector.export_prometheus()
        Path(args.prometheus_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.prometheus_output, "w") as f:
            f.write(prom_metrics)
        print(f"Prometheus metrics saved to: {args.prometheus_output}")
        
    else:
        # Production mode - would integrate with actual API server
        print("Production monitoring mode")
        print("In production, this would:")
        print("  1. Integrate with API server to collect real metrics")
        print("  2. Export to Prometheus/Grafana")
        print("  3. Send alerts on anomalies")
        print("\nFor simulation, run: python mlops/monitoring_service.py --simulate")


if __name__ == "__main__":
    main()
