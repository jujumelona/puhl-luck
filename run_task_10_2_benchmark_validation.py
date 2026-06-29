"""
Task 10.2: Run full benchmark suite and validate targets

This script:
1. Executes BenchmarkSuite on all task types (code, classification, pattern, Q&A)
2. Validates accuracy >85% on each task type (Requirements 1.1, 1.2, 1.3, 1.4)
3. Validates inference speed <50ms per query (Requirements 2.1, 2.2)
4. Validates training speed <1000ms for 10 examples (Requirement 11.6)
5. Reports which requirements are met/failed
6. Saves detailed results to JSON for analysis

Requirements Validated:
- Requirement 1.1: Accuracy >85% on code generation
- Requirement 1.2: Accuracy >85% on classification
- Requirement 1.3: Accuracy >85% on pattern matching
- Requirement 1.4: Accuracy >85% on Q&A
- Requirement 2.1: Inference speed <50ms per query (all tasks)
- Requirement 2.2: Inference speed <50ms for code generation
- Requirement 11.6: Training speed <1000ms for 10 examples
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# Add package to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "puhl_luck"))

from puhl_luck.benchmarks import BenchmarkSuite
from puhl_luck.brain_memory import BrainMemory


@dataclass
class RequirementValidation:
    """Validation result for a single requirement"""
    requirement_id: str
    description: str
    target: str
    actual: str
    passed: bool
    details: str


class BenchmarkValidator:
    """Validates benchmark results against spec requirements"""
    
    def __init__(self):
        self.validations: List[RequirementValidation] = []
    
    def validate_accuracy(
        self,
        task_name: str,
        requirement_id: str,
        accuracy: float,
        target: float = 0.85
    ) -> RequirementValidation:
        """Validate accuracy requirement"""
        passed = accuracy >= target
        
        validation = RequirementValidation(
            requirement_id=requirement_id,
            description=f"Accuracy on {task_name} tasks",
            target=f">={target * 100:.0f}%",
            actual=f"{accuracy * 100:.1f}%",
            passed=passed,
            details=f"{'✓ PASS' if passed else '✗ FAIL'}: {task_name} accuracy is {accuracy * 100:.1f}% (target: ≥{target * 100:.0f}%)"
        )
        
        self.validations.append(validation)
        return validation
    
    def validate_speed(
        self,
        task_name: str,
        requirement_id: str,
        speed_ms: float,
        target_ms: float = 50.0
    ) -> RequirementValidation:
        """Validate speed requirement"""
        passed = speed_ms < target_ms
        
        validation = RequirementValidation(
            requirement_id=requirement_id,
            description=f"Inference speed on {task_name}",
            target=f"<{target_ms:.0f}ms",
            actual=f"{speed_ms:.2f}ms",
            passed=passed,
            details=f"{'✓ PASS' if passed else '✗ FAIL'}: {task_name} inference is {speed_ms:.2f}ms (target: <{target_ms:.0f}ms)"
        )
        
        self.validations.append(validation)
        return validation
    
    def validate_training_speed(
        self,
        training_time_ms: float,
        num_examples: int,
        target_ms_per_10: float = 1000.0
    ) -> RequirementValidation:
        """Validate training speed requirement"""
        # Normalize to 10 examples
        normalized_time = (training_time_ms / num_examples) * 10
        passed = normalized_time < target_ms_per_10
        
        validation = RequirementValidation(
            requirement_id="11.6",
            description="Training speed for 10 examples",
            target=f"<{target_ms_per_10:.0f}ms",
            actual=f"{normalized_time:.2f}ms",
            passed=passed,
            details=f"{'✓ PASS' if passed else '✗ FAIL'}: Training 10 examples takes {normalized_time:.2f}ms (target: <{target_ms_per_10:.0f}ms)"
        )
        
        self.validations.append(validation)
        return validation
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total = len(self.validations)
        passed = sum(1 for v in self.validations if v.passed)
        failed = total - passed
        
        return {
            "summary": {
                "total_requirements": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total if total > 0 else 0.0
            },
            "validations": [
                {
                    "requirement_id": v.requirement_id,
                    "description": v.description,
                    "target": v.target,
                    "actual": v.actual,
                    "passed": v.passed,
                    "details": v.details
                }
                for v in self.validations
            ]
        }


def measure_training_speed(num_examples: int = 10) -> float:
    """Measure training speed for a fixed number of examples"""
    brain = BrainMemory()
    
    # Simple training data
    training_data = [
        (f"def func{i}(x):", f"def func{i}(x):\n    return x * {i}")
        for i in range(num_examples)
    ]
    
    start_time = time.time()
    for input_text, target_text in training_data:
        brain.expose_pair(
            partial=input_text,
            complete=target_text,
            domain='code',
            modality='code'
        )
    training_time_ms = (time.time() - start_time) * 1000
    
    return training_time_ms


def main():
    """Main benchmark validation execution"""
    
    print("=" * 80)
    print("TASK 10.2: FULL BENCHMARK SUITE AND VALIDATION")
    print("=" * 80)
    print()
    print("This script will:")
    print("  1. Run BenchmarkSuite on all task types")
    print("  2. Validate accuracy targets (>85% on all tasks)")
    print("  3. Validate speed targets (<50ms inference, <1000ms training)")
    print("  4. Generate comprehensive validation report")
    print("  5. Save results to JSON")
    print()
    print("-" * 80)
    print()
    
    # Initialize suite and validator
    suite = BenchmarkSuite()
    validator = BenchmarkValidator()
    
    # ========================================================================
    # STEP 1: RUN COMPREHENSIVE BENCHMARK SUITE
    # ========================================================================
    print("STEP 1: Running comprehensive benchmark suite...")
    print("=" * 80)
    print()
    
    benchmark_results = suite.run_all_benchmarks(
        tasks=['code', 'classification', 'pattern', 'qa'],
        max_new_tokens=64,
        verbose=True
    )
    
    print()
    print("✓ Benchmark suite completed")
    print()
    
    # ========================================================================
    # STEP 2: MEASURE TRAINING SPEED
    # ========================================================================
    print()
    print("STEP 2: Measuring training speed...")
    print("=" * 80)
    print()
    
    training_time_ms = measure_training_speed(num_examples=10)
    print(f"Training time for 10 examples: {training_time_ms:.2f}ms")
    print()
    
    # ========================================================================
    # STEP 3: VALIDATE REQUIREMENTS
    # ========================================================================
    print()
    print("STEP 3: Validating requirements...")
    print("=" * 80)
    print()
    
    task_results = benchmark_results['task_results']
    aggregate = benchmark_results['aggregate_metrics']
    
    # Validate accuracy requirements (1.1, 1.2, 1.3, 1.4)
    print("Accuracy Requirements:")
    print("-" * 80)
    
    if 'code' in task_results:
        v = validator.validate_accuracy(
            'code generation',
            '1.1',
            task_results['code']['accuracy'],
            target=0.85
        )
        print(f"  {v.details}")
    
    if 'classification' in task_results:
        v = validator.validate_accuracy(
            'classification',
            '1.2',
            task_results['classification']['accuracy'],
            target=0.85
        )
        print(f"  {v.details}")
    
    if 'pattern' in task_results:
        v = validator.validate_accuracy(
            'pattern matching',
            '1.3',
            task_results['pattern']['accuracy'],
            target=0.85
        )
        print(f"  {v.details}")
    
    if 'qa' in task_results:
        v = validator.validate_accuracy(
            'Q&A',
            '1.4',
            task_results['qa']['accuracy'],
            target=0.85
        )
        print(f"  {v.details}")
    
    print()
    
    # Validate speed requirements (2.1, 2.2)
    print("Speed Requirements:")
    print("-" * 80)
    
    # Overall inference speed (Requirement 2.1)
    v = validator.validate_speed(
        'all tasks',
        '2.1',
        aggregate['avg_inference_time_ms'],
        target_ms=50.0
    )
    print(f"  {v.details}")
    
    # Code generation inference speed (Requirement 2.2)
    if 'code' in task_results:
        v = validator.validate_speed(
            'code generation',
            '2.2',
            task_results['code']['avg_inference_time_ms'],
            target_ms=50.0
        )
        print(f"  {v.details}")
    
    print()
    
    # Validate training speed (Requirement 11.6)
    print("Training Speed Requirements:")
    print("-" * 80)
    
    v = validator.validate_training_speed(
        training_time_ms,
        num_examples=10,
        target_ms_per_10=1000.0
    )
    print(f"  {v.details}")
    
    print()
    
    # ========================================================================
    # STEP 4: GENERATE VALIDATION REPORT
    # ========================================================================
    print()
    print("STEP 4: Generating validation report...")
    print("=" * 80)
    print()
    
    validation_report = validator.generate_report()
    
    summary = validation_report['summary']
    print(f"Requirements Summary:")
    print(f"  Total requirements: {summary['total_requirements']}")
    print(f"  Passed: {summary['passed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Pass rate: {summary['pass_rate'] * 100:.1f}%")
    print()
    
    # ========================================================================
    # STEP 5: DETAILED ANALYSIS
    # ========================================================================
    print()
    print("STEP 5: Detailed analysis...")
    print("=" * 80)
    print()
    
    print("Requirements Status:")
    print()
    
    passed_reqs = [v for v in validation_report['validations'] if v['passed']]
    failed_reqs = [v for v in validation_report['validations'] if not v['passed']]
    
    if passed_reqs:
        print("✓ PASSED REQUIREMENTS:")
        for v in passed_reqs:
            print(f"  [{v['requirement_id']}] {v['description']}")
            print(f"      Target: {v['target']}, Actual: {v['actual']}")
        print()
    
    if failed_reqs:
        print("✗ FAILED REQUIREMENTS:")
        for v in failed_reqs:
            print(f"  [{v['requirement_id']}] {v['description']}")
            print(f"      Target: {v['target']}, Actual: {v['actual']}")
        print()
    
    # ========================================================================
    # STEP 6: OPTIMIZATION RECOMMENDATIONS
    # ========================================================================
    print()
    print("STEP 6: Optimization recommendations...")
    print("=" * 80)
    print()
    
    recommendations = []
    
    # Check for accuracy issues
    for task, metrics in task_results.items():
        if metrics['accuracy'] < 0.85:
            recommendations.append(
                f"• {task.upper()}: Accuracy {metrics['accuracy']*100:.1f}% is below target (85%). "
                f"Consider hyperparameter tuning or increasing training data."
            )
    
    # Check for speed issues
    if aggregate['avg_inference_time_ms'] >= 50.0:
        recommendations.append(
            f"• INFERENCE SPEED: Average {aggregate['avg_inference_time_ms']:.2f}ms exceeds target (50ms). "
            f"Consider enabling Rust acceleration or optimizing sparse table lookups."
        )
    
    # Check for training speed issues
    normalized_training = (training_time_ms / 10) * 10
    if normalized_training >= 1000.0:
        recommendations.append(
            f"• TRAINING SPEED: {normalized_training:.2f}ms for 10 examples exceeds target (1000ms). "
            f"Consider batch processing or reducing feature extraction overhead."
        )
    
    if recommendations:
        print("Recommendations for further optimization:")
        for rec in recommendations:
            print(rec)
        print()
    else:
        print("✓ All targets achieved! No further optimization needed.")
        print()
    
    # ========================================================================
    # STEP 7: SAVE RESULTS
    # ========================================================================
    print()
    print("STEP 7: Saving results...")
    print("=" * 80)
    print()
    
    # Combine all results
    final_results = {
        "timestamp": benchmark_results['timestamp'],
        "task_results": benchmark_results['task_results'],
        "aggregate_metrics": benchmark_results['aggregate_metrics'],
        "training_metrics": {
            "training_time_ms": training_time_ms,
            "num_examples": 10,
            "time_per_example_ms": training_time_ms / 10
        },
        "validation_report": validation_report,
        "recommendations": recommendations
    }
    
    # Save to JSON
    output_file = "task_10_2_validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Full results saved to: {output_file}")
    print()
    
    # Also save using BenchmarkSuite's save method
    benchmark_file = suite.save_results(benchmark_results, "task_10_2_benchmark_results.json")
    print(f"✓ Benchmark results saved to: {benchmark_file}")
    print()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print()
    print("=" * 80)
    print("TASK 10.2 COMPLETE: FINAL SUMMARY")
    print("=" * 80)
    print()
    
    print(f"📊 Overall Performance:")
    print(f"   • Tests run: {aggregate['total_tests']}")
    print(f"   • Overall accuracy: {aggregate['overall_accuracy'] * 100:.1f}%")
    print(f"   • Avg inference time: {aggregate['avg_inference_time_ms']:.2f}ms")
    print(f"   • Training time (10 examples): {training_time_ms:.2f}ms")
    print()
    
    print(f"📈 Per-Task Performance:")
    for task_name, metrics in task_results.items():
        status = "✓" if metrics['accuracy'] >= 0.85 else "✗"
        print(f"   {status} {task_name.upper()}: {metrics['accuracy']*100:.1f}% accuracy, {metrics['avg_inference_time_ms']:.2f}ms inference")
    print()
    
    print(f"✅ Requirements Validation:")
    print(f"   • Total: {summary['total_requirements']}")
    print(f"   • Passed: {summary['passed']} ({summary['pass_rate']*100:.1f}%)")
    print(f"   • Failed: {summary['failed']}")
    print()
    
    if summary['pass_rate'] == 1.0:
        print("🎉 ALL REQUIREMENTS MET! System is production-ready.")
    elif summary['pass_rate'] >= 0.7:
        print("⚠️  Most requirements met. Minor optimizations needed.")
    else:
        print("❌ Significant optimization needed to meet targets.")
    
    print()
    print("=" * 80)
    print()
    
    return final_results


if __name__ == '__main__':
    results = main()
