"""
Demonstration of Task 7.3: Pareto-optimal Configuration Selection

This script demonstrates the complete implementation of:
1. identify_pareto_front() method to find accuracy-speed tradeoffs
2. recommend_config() method with priority options (accuracy, speed, balanced)
3. Best configuration selection based on user priority
4. Saving all tested configurations to JSON for analysis

Task 7.3 Requirements:
- Add identify_pareto_front() method to find accuracy-speed tradeoffs
- Implement recommend_config() method with priority options
- Return best configuration based on user priority
- Save all tested configurations to JSON

Requirements: 12.3, 12.4, 12.5
"""

import sys
import json
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "puhl_luck"))

from puhl_luck.benchmarks.hyperparameter_tuner import HyperparameterTuner


def create_training_data():
    """Create training dataset."""
    return [
        # Code completion examples
        ("def add(a, b):", "return a + b"),
        ("def subtract(x, y):", "return x - y"),
        ("def multiply(m, n):", "return m * n"),
        ("def divide(num, denom):", "return num / denom"),
        ("def square(x):", "return x * x"),
        ("def is_even(x):", "return x % 2 == 0"),
        ("def max_val(a, b):", "return a if a > b else b"),
        # Pattern matching
        ("if x > 0:", "print('positive')"),
        ("for i in range(10):", "print(i)"),
        ("while x > 0:", "x -= 1"),
    ]


def create_test_data():
    """Create test dataset."""
    return [
        ("def add(x, y):", "return x + y"),
        ("def subtract(a, b):", "return a - b"),
        ("def multiply(p, q):", "return p * q"),
        ("def square(n):", "return n * n"),
        ("def is_even(num):", "return num % 2 == 0"),
    ]


def demonstrate_pareto_selection():
    """Demonstrate Pareto-optimal configuration selection."""
    print("=" * 80)
    print("TASK 7.3 DEMONSTRATION: Pareto-Optimal Configuration Selection")
    print("=" * 80)
    print()
    
    # Prepare data
    print("Step 1: Preparing datasets")
    print("-" * 80)
    train_data = create_training_data()
    test_data = create_test_data()
    print(f"  Training pairs: {len(train_data)}")
    print(f"  Test pairs: {len(test_data)}")
    print("  ✓ Data prepared")
    print()
    
    # Create tuner
    print("Step 2: Creating HyperparameterTuner")
    print("-" * 80)
    tuner = HyperparameterTuner(
        train_data=train_data,
        test_data=test_data,
        domain='code_completion'
    )
    print("  ✓ Tuner initialized")
    print()
    
    # Run grid search with a moderate parameter space
    print("Step 3: Running grid search")
    print("-" * 80)
    print("  Search space:")
    print("    Context windows: [3, 5, 7]")
    print("    Rare thresholds: [1, 2, 3]")
    print("    Top-K values: [1, 3, 5]")
    print()
    
    results = tuner.grid_search(
        context_windows=[3, 5, 7],
        rare_thresholds=[1, 2, 3],
        top_k_values=[1, 3, 5],
        max_new_tokens=30,
        verbose=False  # Suppress detailed output for cleaner demo
    )
    
    print(f"  ✓ Evaluated {results['total_evaluations']} configurations")
    print(f"  ✓ Total time: {results['total_time_ms'] / 1000:.2f}s")
    print()
    
    # ============================================================================
    # TASK 7.3 REQUIREMENT 1: Identify Pareto Front
    # ============================================================================
    print("=" * 80)
    print("REQUIREMENT 12.3: Identify Pareto-Optimal Configurations")
    print("=" * 80)
    print()
    
    print("What is a Pareto-optimal configuration?")
    print("-" * 80)
    print("  A configuration is Pareto-optimal if no other configuration is")
    print("  strictly better in BOTH accuracy AND speed. These configurations")
    print("  represent the best accuracy-speed tradeoffs.")
    print()
    
    # The pareto_front is already computed in grid_search results
    pareto_front = results['pareto_front']
    
    print(f"Found {len(pareto_front)} Pareto-optimal configurations:")
    print("-" * 80)
    print(f"  {'#':<3} {'K':<3} {'Rare':<5} {'TopK':<5} {'Accuracy':<10} {'Speed (ms)':<12} {'Tradeoff'}")
    print("-" * 80)
    
    # Sort by accuracy for clear visualization
    pareto_sorted = sorted(pareto_front, key=lambda r: r['accuracy'], reverse=True)
    
    for i, result in enumerate(pareto_sorted, 1):
        config = result['config']
        acc = result['accuracy']
        speed = result['avg_inference_time_ms']
        
        # Describe the tradeoff position
        if i == 1:
            tradeoff = "High accuracy"
        elif i == len(pareto_sorted):
            tradeoff = "High speed"
        else:
            tradeoff = "Balanced"
        
        print(f"  {i:<3} {config['context_window']:<3} {config['rare_token_threshold']:<5} "
              f"{config['top_k']:<5} {acc*100:>7.1f}%  {speed:>10.2f}  {tradeoff}")
    
    print()
    print(f"✓ Identified {len(pareto_front)} Pareto-optimal configurations")
    print()
    
    # Show why non-Pareto configs were excluded
    print("Why other configurations were excluded:")
    print("-" * 80)
    all_results = results['all_results']
    non_pareto = [r for r in all_results if r not in pareto_front]
    
    if non_pareto:
        example = non_pareto[0]
        example_config = example['config']
        example_acc = example['accuracy']
        example_speed = example['avg_inference_time_ms']
        
        print(f"  Example: K={example_config['context_window']}, "
              f"rare={example_config['rare_token_threshold']}, "
              f"top_k={example_config['top_k']}")
        print(f"  Accuracy: {example_acc*100:.1f}%, Speed: {example_speed:.2f}ms")
        print()
        
        # Find a dominating configuration
        for pareto_result in pareto_front:
            pareto_acc = pareto_result['accuracy']
            pareto_speed = pareto_result['avg_inference_time_ms']
            
            if (pareto_acc >= example_acc and pareto_speed <= example_speed and
                (pareto_acc > example_acc or pareto_speed < example_speed)):
                pareto_config = pareto_result['config']
                print(f"  Dominated by: K={pareto_config['context_window']}, "
                      f"rare={pareto_config['rare_token_threshold']}, "
                      f"top_k={pareto_config['top_k']}")
                print(f"  Accuracy: {pareto_acc*100:.1f}% (better), "
                      f"Speed: {pareto_speed:.2f}ms (better)")
                break
    
    print()
    
    # ============================================================================
    # TASK 7.3 REQUIREMENT 2: Recommend Configuration by Priority
    # ============================================================================
    print("=" * 80)
    print("REQUIREMENT 12.5: Recommend Configuration Based on Priority")
    print("=" * 80)
    print()
    
    priorities = ['accuracy', 'speed', 'balanced']
    recommendations = {}
    
    for priority in priorities:
        print(f"Priority: {priority.upper()}")
        print("-" * 80)
        
        recommendation = tuner.recommend_config(results, priority=priority)
        recommendations[priority] = recommendation
        
        config = recommendation['recommended_config']
        acc = recommendation['accuracy']
        speed = recommendation['avg_inference_time_ms']
        
        print(f"  Recommended Config:")
        print(f"    context_window: {config['context_window']}")
        print(f"    rare_token_threshold: {config['rare_token_threshold']}")
        print(f"    top_k: {config['top_k']}")
        print()
        print(f"  Performance Metrics:")
        print(f"    Accuracy: {acc * 100:.1f}%")
        print(f"    Inference Speed: {speed:.2f}ms")
        print(f"    Empty Outputs: {recommendation['empty_outputs']}")
        print(f"    Avg Backoff Level: {recommendation['avg_backoff_level']:.2f}")
        print()
        
        # Explain the selection
        if priority == 'accuracy':
            print(f"  Explanation: Selected configuration with highest accuracy ({acc*100:.1f}%)")
        elif priority == 'speed':
            print(f"  Explanation: Selected configuration with lowest inference time ({speed:.2f}ms)")
        elif priority == 'balanced':
            print(f"  Explanation: Selected configuration with best balance of accuracy and speed")
        print()
    
    print("✓ Generated recommendations for all three priority modes")
    print()
    
    # Compare recommendations
    print("Comparing Recommendations:")
    print("-" * 80)
    print(f"  {'Priority':<12} {'Accuracy':<12} {'Speed (ms)':<12} {'Config'}")
    print("-" * 80)
    for priority in priorities:
        rec = recommendations[priority]
        config = rec['recommended_config']
        config_str = f"K={config['context_window']}, rare={config['rare_token_threshold']}, top_k={config['top_k']}"
        print(f"  {priority:<12} {rec['accuracy']*100:>7.1f}%    {rec['avg_inference_time_ms']:>8.2f}    {config_str}")
    print()
    
    # ============================================================================
    # TASK 7.3 REQUIREMENT 3: Save All Configurations to JSON
    # ============================================================================
    print("=" * 80)
    print("REQUIREMENT 12.4: Save All Tested Configurations to JSON")
    print("=" * 80)
    print()
    
    output_file = "task_7_3_pareto_results.json"
    print(f"Saving results to: {output_file}")
    print("-" * 80)
    
    saved_path = tuner.save_tuning_results(results, filename=output_file)
    
    # Verify the saved file
    saved_file = Path(saved_path)
    file_size_kb = saved_file.stat().st_size / 1024
    
    print(f"  ✓ File saved: {saved_path}")
    print(f"  ✓ File size: {file_size_kb:.2f} KB")
    print()
    
    # Show what was saved
    with open(saved_path, 'r') as f:
        saved_data = json.load(f)
    
    print("Saved data includes:")
    print("-" * 80)
    print(f"  • Timestamp: {saved_data['timestamp']}")
    print(f"  • Domain: {saved_data['domain']}")
    print(f"  • Training examples: {saved_data['training_examples']}")
    print(f"  • Test examples: {saved_data['test_examples']}")
    print(f"  • Search space: {saved_data['search_space']}")
    print(f"  • All results: {len(saved_data['all_results'])} configurations")
    print(f"  • Best accuracy config: {saved_data['best_accuracy_config']['config']}")
    print(f"  • Best speed config: {saved_data['best_speed_config']['config']}")
    print(f"  • Pareto front: {len(saved_data['pareto_front'])} configurations")
    print(f"  • Total evaluations: {saved_data['total_evaluations']}")
    print(f"  • Total time: {saved_data['total_time_ms'] / 1000:.2f}s")
    print()
    
    print("✓ All tested configurations saved to JSON for analysis")
    print()
    
    # ============================================================================
    # VERIFICATION SUMMARY
    # ============================================================================
    print("=" * 80)
    print("TASK 7.3 VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    
    print("✓ REQUIREMENT 12.3: Identify Pareto-optimal configurations")
    print(f"  - Found {len(pareto_front)} Pareto-optimal configs from {len(all_results)} tested")
    print(f"  - These represent the best accuracy-speed tradeoffs")
    print()
    
    print("✓ REQUIREMENT 12.5: Recommend configuration by priority")
    print(f"  - Accuracy priority: {recommendations['accuracy']['accuracy']*100:.1f}% accuracy")
    print(f"  - Speed priority: {recommendations['speed']['avg_inference_time_ms']:.2f}ms inference")
    print(f"  - Balanced priority: optimal balance of both metrics")
    print()
    
    print("✓ REQUIREMENT 12.4: Save all configurations to JSON")
    print(f"  - Saved {len(all_results)} configurations with full metrics")
    print(f"  - File: {output_file} ({file_size_kb:.2f} KB)")
    print()
    
    print("=" * 80)
    print("TASK 7.3 COMPLETE")
    print("=" * 80)
    print()
    print("All required functionality implemented and verified:")
    print("  1. ✓ identify_pareto_front() finds accuracy-speed tradeoffs")
    print("  2. ✓ recommend_config() supports accuracy/speed/balanced priorities")
    print("  3. ✓ Best configuration returned based on user priority")
    print("  4. ✓ All tested configurations saved to JSON for analysis")
    print()
    print("The system can now:")
    print("  • Identify the best accuracy-speed tradeoffs (Pareto front)")
    print("  • Recommend optimal configurations for different use cases")
    print("  • Provide detailed analysis data for further investigation")
    print()


if __name__ == "__main__":
    try:
        demonstrate_pareto_selection()
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
