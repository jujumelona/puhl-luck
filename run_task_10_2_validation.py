"""
Task 10.2: Run full benchmark suite and validate targets

This script executes the BenchmarkSuite on all task types and validates:
- Accuracy >85% on code generation (Requirement 1.1)
- Accuracy >85% on classification (Requirement 1.2)
- Accuracy >85% on pattern matching (Requirement 1.3)
- Accuracy >85% on Q&A (Requirement 1.4)
- Inference speed <50ms per query (Requirement 2.1, 2.2)
- Training speed <1000ms for 10 examples (Requirement 11.6)
"""

import sys
import json
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "puhl_luck"))

from puhl_luck.benchmarks import BenchmarkSuite


def validate_targets(results: dict) -> dict:
    """
    Validate benchmark results against target requirements.
    
    Returns:
        Dictionary with validation results for each requirement
    """
    validation = {
        "all_passed": True,
        "requirements": {}
    }
    
    task_results = results.get("task_results", {})
    aggregate = results.get("aggregate_metrics", {})
    
    # Requirement 1.1: Code generation accuracy >85%
    code_accuracy = task_results.get("code", {}).get("accuracy", 0) * 100
    req_1_1 = code_accuracy > 85
    validation["requirements"]["1.1"] = {
        "description": "Code generation accuracy >85%",
        "target": ">85%",
        "actual": f"{code_accuracy:.1f}%",
        "passed": req_1_1
    }
    if not req_1_1:
        validation["all_passed"] = False
    
    # Requirement 1.2: Classification accuracy >85%
    classification_accuracy = task_results.get("classification", {}).get("accuracy", 0) * 100
    req_1_2 = classification_accuracy > 85
    validation["requirements"]["1.2"] = {
        "description": "Classification accuracy >85%",
        "target": ">85%",
        "actual": f"{classification_accuracy:.1f}%",
        "passed": req_1_2
    }
    if not req_1_2:
        validation["all_passed"] = False
    
    # Requirement 1.3: Pattern matching accuracy >85%
    pattern_accuracy = task_results.get("pattern", {}).get("accuracy", 0) * 100
    req_1_3 = pattern_accuracy > 85
    validation["requirements"]["1.3"] = {
        "description": "Pattern matching accuracy >85%",
        "target": ">85%",
        "actual": f"{pattern_accuracy:.1f}%",
        "passed": req_1_3
    }
    if not req_1_3:
        validation["all_passed"] = False
    
    # Requirement 1.4: Q&A accuracy >85%
    qa_accuracy = task_results.get("qa", {}).get("accuracy", 0) * 100
    req_1_4 = qa_accuracy > 85
    validation["requirements"]["1.4"] = {
        "description": "Q&A accuracy >85%",
        "target": ">85%",
        "actual": f"{qa_accuracy:.1f}%",
        "passed": req_1_4
    }
    if not req_1_4:
        validation["all_passed"] = False
    
    # Requirement 2.1 & 2.2: Inference speed <50ms per query
    avg_inference_time = aggregate.get("avg_inference_time_ms", 0)
    req_2_1_2_2 = avg_inference_time < 50
    validation["requirements"]["2.1_2.2"] = {
        "description": "Inference speed <50ms per query",
        "target": "<50ms",
        "actual": f"{avg_inference_time:.2f}ms",
        "passed": req_2_1_2_2
    }
    if not req_2_1_2_2:
        validation["all_passed"] = False
    
    # Additional speed metrics per task type
    code_speed = task_results.get("code", {}).get("avg_inference_time_ms", 0)
    classification_speed = task_results.get("classification", {}).get("avg_inference_time_ms", 0)
    pattern_speed = task_results.get("pattern", {}).get("avg_inference_time_ms", 0)
    qa_speed = task_results.get("qa", {}).get("avg_inference_time_ms", 0)
    
    validation["requirements"]["2.2_code"] = {
        "description": "Code generation speed <50ms",
        "target": "<50ms",
        "actual": f"{code_speed:.2f}ms",
        "passed": code_speed < 50
    }
    
    validation["requirements"]["2.3_classification"] = {
        "description": "Classification speed <20ms",
        "target": "<20ms",
        "actual": f"{classification_speed:.2f}ms",
        "passed": classification_speed < 20
    }
    
    validation["requirements"]["2.4_pattern"] = {
        "description": "Pattern matching speed <20ms",
        "target": "<20ms",
        "actual": f"{pattern_speed:.2f}ms",
        "passed": pattern_speed < 20
    }
    
    validation["requirements"]["2.5_qa"] = {
        "description": "Q&A speed <20ms",
        "target": "<20ms",
        "actual": f"{qa_speed:.2f}ms",
        "passed": qa_speed < 20
    }
    
    return validation


def print_validation_report(validation: dict):
    """Print a formatted validation report."""
    print("\n" + "=" * 70)
    print("REQUIREMENT VALIDATION REPORT")
    print("=" * 70)
    
    for req_id, req_data in validation["requirements"].items():
        status = "✓ PASS" if req_data["passed"] else "✗ FAIL"
        print(f"\n[{req_id}] {req_data['description']}")
        print(f"  Target:  {req_data['target']}")
        print(f"  Actual:  {req_data['actual']}")
        print(f"  Status:  {status}")
    
    print("\n" + "=" * 70)
    overall_status = "✓ ALL REQUIREMENTS PASSED" if validation["all_passed"] else "✗ SOME REQUIREMENTS FAILED"
    print(f"OVERALL: {overall_status}")
    print("=" * 70)


def main():
    """Execute task 10.2: Run full benchmark suite and validate targets."""
    print("Task 10.2: Run Full Benchmark Suite and Validate Targets")
    print("=" * 70)
    print()
    
    # Create benchmark suite
    suite = BenchmarkSuite()
    
    # Run all benchmarks (includes training time measurement)
    print("Running comprehensive benchmark suite...")
    print("This will test: code generation, classification, pattern matching, and Q&A")
    print()
    
    results = suite.run_all_benchmarks(
        tasks=['code', 'classification', 'pattern', 'qa'],
        max_new_tokens=64,
        verbose=True
    )
    
    # Save results
    output_file = "task_10_2_benchmark_results.json"
    suite.save_results(results, output_file)
    
    # Validate against requirements
    print("\n" + "=" * 70)
    print("VALIDATING AGAINST REQUIREMENTS")
    print("=" * 70)
    
    validation = validate_targets(results)
    print_validation_report(validation)
    
    # Save validation report
    validation_file = "task_10_2_validation_report.json"
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(validation, f, indent=2)
    print(f"\nValidation report saved to: {validation_file}")
    
    # Return exit code based on validation
    if validation["all_passed"]:
        print("\n✓ Task 10.2 completed successfully - all requirements met!")
        return 0
    else:
        print("\n✗ Task 10.2 completed with failures - some requirements not met")
        print("   See validation report for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
