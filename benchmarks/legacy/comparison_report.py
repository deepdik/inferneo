#!/usr/bin/env python3
"""
Comprehensive Benchmark Comparison Report

Analyzes and compares benchmark results from different inference systems.
"""

import json
import time
import os
from typing import Dict, List, Any, Optional
import numpy as np

class BenchmarkComparison:
    """Generate comprehensive comparison reports"""
    
    def __init__(self):
        self.results = {}
    
    def load_results(self, file_path: str, system_name: str):
        """Load benchmark results from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.results[system_name] = data
                print(f"Loaded {system_name} results from {file_path}")
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
    
    def generate_comparison_report(self, output_file: str = None):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("No results loaded")
            return
        
        report = {
            "timestamp": time.time(),
            "summary": self._generate_summary(),
            "detailed_comparison": self._generate_detailed_comparison(),
            "recommendations": self._generate_recommendations()
        }
        
        if output_file is None:
            output_file = f"comprehensive_comparison_report_{int(time.time())}.json"
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Comprehensive report saved to {output_file}")
        
        # Print summary
        self._print_summary(report["summary"])
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary"""
        summary = {
            "systems_tested": list(self.results.keys()),
            "models_tested": [],
            "overall_winner": None,
            "key_metrics": {}
        }
        
        # Collect models and key metrics
        for system, data in self.results.items():
            if "config" in data:
                summary["models_tested"].append(data["config"].get("model_name", "unknown"))
            
            # Extract key metrics
            if "results" in data and "single_requests" in data["results"]:
                single_results = data["results"]["single_requests"]
                for sys_name, metrics in single_results.items():
                    if "mean" in metrics:
                        summary["key_metrics"][f"{system}_{sys_name}"] = {
                            "mean_latency_ms": metrics["mean"],
                            "p95_latency_ms": metrics.get("p95", 0),
                            "throughput_req_s": 1000 / metrics["mean"] if metrics["mean"] > 0 else 0
                        }
        
        # Determine overall winner
        if summary["key_metrics"]:
            best_latency = float('inf')
            best_system = None
            
            for metric_key, metrics in summary["key_metrics"].items():
                if metrics["mean_latency_ms"] < best_latency:
                    best_latency = metrics["mean_latency_ms"]
                    best_system = metric_key
            
            summary["overall_winner"] = best_system
            summary["best_latency_ms"] = best_latency
        
        return summary
    
    def _generate_detailed_comparison(self) -> Dict[str, Any]:
        """Generate detailed comparison analysis"""
        comparison = {
            "latency_analysis": {},
            "throughput_analysis": {},
            "batch_processing": {},
            "concurrent_processing": {},
            "resource_utilization": {}
        }
        
        for system, data in self.results.items():
            if "results" not in data:
                continue
            
            results = data["results"]
            
            # Latency analysis
            if "single_requests" in results:
                for sys_name, metrics in results["single_requests"].items():
                    key = f"{system}_{sys_name}"
                    comparison["latency_analysis"][key] = {
                        "mean_ms": metrics.get("mean", 0),
                        "median_ms": metrics.get("median", 0),
                        "p95_ms": metrics.get("p95", 0),
                        "p99_ms": metrics.get("p99", 0),
                        "std_dev_ms": metrics.get("std", 0),
                        "errors": metrics.get("errors", 0)
                    }
            
            # Throughput analysis
            if "throughput" in results:
                for rate_key, rate_data in results["throughput"].items():
                    for sys_name, metrics in rate_data.items():
                        key = f"{system}_{sys_name}_{rate_key}"
                        comparison["throughput_analysis"][key] = {
                            "target_rate": metrics.get("target_rate", 0),
                            "actual_rate": metrics.get("actual_rate", 0),
                            "mean_latency_ms": metrics.get("latency_stats", {}).get("mean", 0)
                        }
            
            # Batch processing
            if "batch_processing" in results:
                for batch_key, batch_data in results["batch_processing"].items():
                    for sys_name, metrics in batch_data.items():
                        key = f"{system}_{sys_name}_{batch_key}"
                        comparison["batch_processing"][key] = {
                            "mean_ms": metrics.get("mean", 0),
                            "throughput_tokens_s": 1000 / metrics.get("mean", 1) if metrics.get("mean", 0) > 0 else 0
                        }
            
            # Concurrent processing
            if "concurrent_requests" in results:
                for client_key, client_data in results["concurrent_requests"].items():
                    for sys_name, metrics in client_data.items():
                        key = f"{system}_{sys_name}_{client_key}"
                        comparison["concurrent_processing"][key] = {
                            "mean_ms": metrics.get("mean", 0),
                            "throughput_req_s": 1000 / metrics.get("mean", 1) if metrics.get("mean", 0) > 0 else 0
                        }
        
        return comparison
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        if not self.results:
            return ["No results available for recommendations"]
        
        # Analyze latency
        best_latency_system = None
        best_latency = float('inf')
        
        for system, data in self.results.items():
            if "results" in data and "single_requests" in data["results"]:
                for sys_name, metrics in data["results"]["single_requests"].items():
                    if "mean" in metrics and metrics["mean"] < best_latency:
                        best_latency = metrics["mean"]
                        best_latency_system = f"{system}_{sys_name}"
        
        if best_latency_system:
            recommendations.append(f"{best_latency_system} shows the best single-request latency ({best_latency:.2f} ms)")
        
        # Analyze throughput
        best_throughput_system = None
        best_throughput = 0
        
        for system, data in self.results.items():
            if "results" in data and "throughput" in data["results"]:
                for rate_key, rate_data in data["results"]["throughput"].items():
                    for sys_name, metrics in rate_data.items():
                        actual_rate = metrics.get("actual_rate", 0)
                        if actual_rate > best_throughput:
                            best_throughput = actual_rate
                            best_throughput_system = f"{system}_{sys_name}_{rate_key}"
        
        if best_throughput_system:
            recommendations.append(f"{best_throughput_system} achieved the highest throughput ({best_throughput:.1f} req/s)")
        
        # General recommendations
        if len(self.results) > 1:
            recommendations.append("Consider running both systems in production with load balancing for optimal performance")
            recommendations.append("Monitor resource utilization and scale based on actual production load")
        
        recommendations.append("Run benchmarks with production-like workloads for more accurate comparisons")
        recommendations.append("Consider model-specific optimizations for better performance")
        
        return recommendations
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK COMPARISON SUMMARY")
        print("="*80)
        
        print(f"Systems Tested: {', '.join(summary['systems_tested'])}")
        print(f"Models Tested: {', '.join(summary['models_tested'])}")
        
        if summary["overall_winner"]:
            print(f"Overall Winner: {summary['overall_winner']}")
            print(f"Best Latency: {summary['best_latency_ms']:.2f} ms")
        
        print("\nKey Metrics:")
        for metric_key, metrics in summary["key_metrics"].items():
            print(f"  {metric_key}:")
            print(f"    Mean Latency: {metrics['mean_latency_ms']:.2f} ms")
            print(f"    P95 Latency: {metrics['p95_latency_ms']:.2f} ms")
            print(f"    Throughput: {metrics['throughput_req_s']:.1f} req/s")
        
        print("\nRecommendations:")
        for rec in summary.get("recommendations", []):
            print(f"  - {rec}")

def main():
    """Main function to generate comparison report"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive benchmark comparison")
    parser.add_argument("--inferneo-report", help="Path to Inferneo benchmark report")
    parser.add_argument("--triton-report", help="Path to Triton benchmark report")
    parser.add_argument("--output", help="Output file for comparison report")
    
    args = parser.parse_args()
    
    comparison = BenchmarkComparison()
    
    # Load available reports
    if args.inferneo_report and os.path.exists(args.inferneo_report):
        comparison.load_results(args.inferneo_report, "inferneo")
    
    if args.triton_report and os.path.exists(args.triton_report):
        comparison.load_results(args.triton_report, "triton")
    
    # Auto-detect report files
    if not comparison.results:
        # Look for recent benchmark reports
        for file in os.listdir("."):
            if file.startswith("production_benchmark_report_") and file.endswith(".json"):
                comparison.load_results(file, "inferneo")
                break
            elif file.startswith("triton_benchmark_report_") and file.endswith(".json"):
                comparison.load_results(file, "triton")
                break
    
    # Generate comparison report
    comparison.generate_comparison_report(args.output)

if __name__ == "__main__":
    main() 