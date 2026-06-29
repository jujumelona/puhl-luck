"""
Demonstration of BenchmarkSuite functionality for Task 1.1

This script demonstrates all features implemented:
- BenchmarkSuite class with run_all_benchmarks() method
- Coverage of code generation, classification, pattern matching, and Q&A tasks
- save_results() method with JSON output and timestamps
- Tracking of backoff statistics, copy gate activations, and empty output rates
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "puhl_luck"))

from puhl_luck.benchmarks import BenchmarkSuite


def main():
    """Demonstrate BenchmarkSuite functionality"""
    
    print("=" * 80)
    print("BENCHMARK SUITE DEMONSTRATION (Task 1.1)")
    print("=" * 80)
    print()
    print("This demonstrates the BenchmarkSuite implementation covering:")
    print("  ✓ Code generation tasks")
    print("  ✓ Classification tasks")
    print("  ✓ Pattern matching tasks")
    print("  ✓ Q&A tasks")
    print()
    print("Tracked metrics:")
    print("  ✓ Accuracy per task")
    print("  ✓ Inference speed per task")
    print("  ✓ Backoff statistics")
    print("  ✓ Copy gate activations")
    print("  ✓ Empty output rates")
    print()
    print("-" * 80)
    print()
    
    # Initialize benchmark suite
    suite = BenchmarkSuite()
    
    # Run all benchmarks
    print("Running comprehensive benchmark suite...")
    print("(This may take 30-60 seconds)")
    print()
    
    results = suite.run_all_benchmarks(
        tasks=['code', 'classification', 'pattern', 'qa'],
        max_new_tokens=64,
        verbose=True
    )
    
    # Save results
    print("\nSaving results to JSON file...")
    output_file = "demo_benchmark_results.json"
    saved_path = suite.save_results(results, output_file)
    
    # Display key findings
    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    agg = results['aggregate_metrics']
    
    print(f"\n📊 Overall Performance:")
    print(f"   • Tests run: {agg['total_tests']}")
    print(f"   • Accuracy: {agg['overall_accuracy'] * 100:.1f}%")
    print(f"   • Avg inference time: {agg['avg_inference_time_ms']:.2f}ms")
    print(f"   • Total execution time: {agg['total_execution_time_ms'] / 1000:.1f}s")
    
    print(f"\n🔍 Diagnostic Metrics:")
    print(f"   • Empty output rate: {agg['overall_empty_rate'] * 100:.1f}%")
    print(f"   • Copy gate activations: {agg['total_copy_gate_activations']}")
    print(f"   • Avg backoff level: {agg['avg_backoff_level']:.2f}")
    print(f"   • Backoff level distribution: {agg['total_backoff_levels']}")
    
    print(f"\n📈 Per-Task Breakdown:")
    for task_name, task_metrics in results['task_results'].items():
        print(f"\n   {task_name.upper()}:")
        print(f"      Accuracy: {task_metrics['accuracy'] * 100:.1f}%")
        print(f"      Inference: {task_metrics['avg_inference_time_ms']:.2f}ms")
        print(f"      Empty: {task_metrics['empty_outputs']}/{task_metrics['total_tests']}")
    
    print()
    print("=" * 80)
    print("✅ TASK 1.1 COMPLETE")
    print("=" * 80)
    print()
    print("Implemented components:")
    print("  ✓ BenchmarkSuite class in packages/puhl_luck/puhl_luck/benchmarks/__init__.py")
    print("  ✓ run_all_benchmarks() method covering all 4 task types")
    print("  ✓ save_results() method with JSON output and timestamps")
    print("  ✓ Comprehensive metric tracking (backoff, copy gate, empty outputs)")
    print()
    print(f"Results saved to: {saved_path}")
    print()


if __name__ == '__main__':
    main()
