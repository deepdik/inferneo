#!/usr/bin/env python3
"""
Test runner for Inferneo

Runs all tests and provides detailed reporting.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Exit code: {result.returncode}")
    print(f"Duration: {end_time - start_time:.2f} seconds")
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
        
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
        
    return result.returncode == 0


def run_unit_tests():
    """Run unit tests"""
    return run_command(
        "python3 -m pytest tests/unit/ -v --tb=short",
        "Unit Tests"
    )


def run_integration_tests():
    """Run integration tests"""
    return run_command(
        "python3 -m pytest tests/integration/ -v --tb=short",
        "Integration Tests"
    )


def run_performance_tests():
    """Run performance tests"""
    return run_command(
        "python3 -m pytest tests/performance/ -v --tb=short",
        "Performance Tests"
    )


def run_all_tests():
    """Run all tests"""
    return run_command(
        "python3 -m pytest tests/ -v --tb=short",
        "All Tests"
    )


def run_coverage():
    """Run tests with coverage"""
    return run_command(
        "python3 -m pytest tests/ --cov=inferneo --cov-report=html --cov-report=term",
        "Tests with Coverage"
    )


def run_linting():
    """Run code linting"""
    return run_command(
        "flake8 inferneo/ tests/ examples/ --max-line-length=120 --ignore=E203,W503",
        "Code Linting"
    )


def run_type_checking():
    """Run type checking"""
    return run_command(
        "mypy inferneo/ --ignore-missing-imports",
        "Type Checking"
    )


def run_benchmarks():
    """Run benchmarks"""
    return run_command(
        "python3 benchmarks/latency/benchmark_latency.py --num-runs=10",
        "Latency Benchmarks"
    )


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Inferneo Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--lint", action="store_true", help="Run code linting")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--benchmarks", action="store_true", help="Run benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all tests and checks")
    
    args = parser.parse_args()
    
    print("🚀 Inferneo - Test Runner")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    results = {}
    
    # Run requested tests
    if args.unit:
        results["unit_tests"] = run_unit_tests()
        
    if args.integration:
        results["integration_tests"] = run_integration_tests()
        
    if args.performance:
        results["performance_tests"] = run_performance_tests()
        
    if args.coverage:
        results["coverage"] = run_coverage()
        
    if args.lint:
        results["linting"] = run_linting()
        
    if args.type_check:
        results["type_checking"] = run_type_checking()
        
    if args.benchmarks:
        results["benchmarks"] = run_benchmarks()
        
    if args.all:
        results["unit_tests"] = run_unit_tests()
        results["integration_tests"] = run_integration_tests()
        results["performance_tests"] = run_performance_tests()
        results["coverage"] = run_coverage()
        results["linting"] = run_linting()
        results["type_checking"] = run_type_checking()
        results["benchmarks"] = run_benchmarks()
        
    # If no specific tests requested, run all tests
    if not any([args.unit, args.integration, args.performance, args.coverage, 
                args.lint, args.type_check, args.benchmarks, args.all]):
        results["all_tests"] = run_all_tests()
        
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20}: {status}")
        if not passed:
            all_passed = False
            
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main()) 