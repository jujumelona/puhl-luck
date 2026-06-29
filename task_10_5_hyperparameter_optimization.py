"""
Task 10.5: Run Hyperparameter Tuning and Apply Best Configuration

This script executes full grid search on comprehensive benchmark data to identify
optimal hyperparameters (context_window, rare_threshold, top_k) for the HDC system.

Requirements:
- 5.1, 5.2, 5.3: Context window optimization
- 6.1, 6.2, 6.3: Rare token threshold optimization
- 7.1, 7.2, 7.3: Top-K optimization
- 12.1, 12.2, 12.3, 12.4, 12.5: Grid search and configuration selection

The script will:
1. Create comprehensive benchmark datasets (code, classification, pattern, Q&A)
2. Run full grid search over all hyperparameter combinations
3. Identify optimal configurations for different priorities
4. Document results and provide recommendations for default configuration
5. Save detailed results for analysis
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

# Add package to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "puhl_luck"))

from puhl_luck.benchmarks.hyperparameter_tuner import HyperparameterTuner


def create_code_completion_data() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Create code completion training and test datasets.
    
    Returns:
        (train_data, test_data) tuples for code completion tasks
        
    Requirements: 11.1, 11.2
    """
    train_data = [
        # Basic arithmetic functions
        ("def add(a, b):", "return a + b"),
        ("def subtract(x, y):", "return x - y"),
        ("def multiply(m, n):", "return m * n"),
        ("def divide(num, denom):", "return num / denom"),
        ("def modulo(a, b):", "return a % b"),
        # Comparison functions
        ("def max_val(a, b):", "return a if a > b else b"),
        ("def min_val(x, y):", "return x if x < y else y"),
        ("def equals(m, n):", "return m == n"),
        # Mathematical functions
        ("def square(x):", "return x * x"),
        ("def cube(n):", "return n * n * n"),
        ("def absolute(x):", "return x if x >= 0 else -x"),
        ("def double(n):", "return n + n"),
        ("def triple(x):", "return x * 3"),
        # Boolean functions
        ("def is_even(x):", "return x % 2 == 0"),
        ("def is_odd(n):", "return n % 2 == 1"),
        ("def is_positive(x):", "return x > 0"),
        ("def is_negative(n):", "return n < 0"),
        ("def is_zero(x):", "return x == 0"),
        # List operations
        ("def first(items):", "return items[0]"),
        ("def last(items):", "return items[-1]"),
        ("def length(items):", "return len(items)"),
        ("def is_empty(items):", "return len(items) == 0"),
        # String operations
        ("def uppercase(text):", "return text.upper()"),
        ("def lowercase(text):", "return text.lower()"),
    ]
    
    test_data = [
        # Variations with different names
        ("def add(x, y):", "return x + y"),
        ("def subtract(a, b):", "return a - b"),
        ("def multiply(p, q):", "return p * q"),
        ("def square(n):", "return n * n"),
        ("def is_even(num):", "return num % 2 == 0"),
        ("def max_val(x, y):", "return x if x > y else y"),
        ("def absolute(n):", "return n if n >= 0 else -n"),
        ("def first(lst):", "return lst[0]"),
    ]
    
    return train_data, test_data


def create_classification_data() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Create sentiment classification training and test datasets.
    
    Returns:
        (train_data, test_data) tuples for classification tasks
        
    Requirements: 11.2
    """
    train_data = [
        # Positive sentiment
        ("I love this product", "positive"),
        ("This is amazing", "positive"),
        ("Great experience", "positive"),
        ("Absolutely wonderful", "positive"),
        ("Highly recommended", "positive"),
        ("Fantastic quality", "positive"),
        ("Excellent service", "positive"),
        ("Best purchase ever", "positive"),
        ("Very satisfied", "positive"),
        ("Outstanding performance", "positive"),
        # Negative sentiment
        ("I hate this", "negative"),
        ("Terrible experience", "negative"),
        ("Very disappointing", "negative"),
        ("Waste of money", "negative"),
        ("Poor quality", "negative"),
        ("Awful service", "negative"),
        ("Do not buy", "negative"),
        ("Completely useless", "negative"),
        ("Very frustrated", "negative"),
        ("Worst product ever", "negative"),
    ]
    
    test_data = [
        ("I really enjoy this", "positive"),
        ("This is terrible", "negative"),
        ("Great value", "positive"),
        ("Very poor", "negative"),
        ("Absolutely love it", "positive"),
        ("Not recommended", "negative"),
        ("Highly satisfied", "positive"),
        ("Disappointing results", "negative"),
    ]
    
    return train_data, test_data


def create_pattern_matching_data() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Create pattern matching training and test datasets.
    
    Returns:
        (train_data, test_data) tuples for pattern matching tasks
        
    Requirements: 11.3
    """
    train_data = [
        # Conditional patterns
        ("if x > 0:", "print('positive')"),
        ("if n < 0:", "print('negative')"),
        ("if y == 0:", "print('zero')"),
        ("if a > b:", "print('greater')"),
        ("if m < n:", "print('lesser')"),
        # Loop patterns
        ("for i in range(10):", "print(i)"),
        ("for j in range(5):", "print(j)"),
        ("for x in items:", "process(x)"),
        ("while x > 0:", "x -= 1"),
        ("while n < 10:", "n += 1"),
        # Try-except patterns
        ("try:", "execute()"),
        ("except:", "handle_error()"),
        ("finally:", "cleanup()"),
        # Function definition patterns
        ("def process(data):", "return transform(data)"),
        ("def validate(input):", "return check(input)"),
    ]
    
    test_data = [
        ("if z > 0:", "print('positive')"),
        ("for k in range(8):", "print(k)"),
        ("while y > 0:", "y -= 1"),
        ("def process(item):", "return transform(item)"),
        ("if a < b:", "print('lesser')"),
        ("for item in list:", "process(item)"),
        ("try:", "execute()"),
        ("def validate(value):", "return check(value)"),
    ]
    
    return train_data, test_data


def create_qa_data() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Create Q&A training and test datasets.
    
    Returns:
        (train_data, test_data) tuples for question-answering tasks
        
    Requirements: 11.4
    """
    train_data = [
        # Programming Q&A
        ("What is Python?", "Python is a programming language"),
        ("What is a variable?", "A variable stores data"),
        ("What is a function?", "A function performs a task"),
        ("What is a loop?", "A loop repeats code"),
        ("What is a class?", "A class defines an object"),
        # Basic facts
        ("What is 2 + 2?", "4"),
        ("What is 5 * 3?", "15"),
        ("What color is the sky?", "blue"),
        ("How many days in a week?", "7"),
        ("What is the capital of France?", "Paris"),
        # Simple reasoning
        ("What comes after Monday?", "Tuesday"),
        ("What is opposite of hot?", "cold"),
    ]
    
    test_data = [
        ("What is JavaScript?", "JavaScript is a programming language"),
        ("What is a method?", "A method performs a task"),
        ("What is 3 + 5?", "8"),
        ("What color is grass?", "green"),
        ("What comes after Tuesday?", "Wednesday"),
        ("What is opposite of cold?", "hot"),
    ]
    
    return train_data, test_data


def run_full_grid_search(
    domain: str,
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run full grid search for a specific domain.
    
    Args:
        domain: Domain name (e.g., 'code_completion', 'classification')
        train_data: Training pairs
        test_data: Test pairs
        verbose: Whether to print progress
        
    Returns:
        Grid search results dictionary
        
    Requirements: 12.1, 12.2
    """
    if verbose:
        print()
        print("=" * 80)
        print(f"GRID SEARCH: {domain.upper()}")
        print("=" * 80)
        print(f"Training examples: {len(train_data)}")
        print(f"Test examples: {len(test_data)}")
        print()
    
    # Create tuner
    tuner = HyperparameterTuner(
        train_data=train_data,
        test_data=test_data,
        domain=domain
    )
    
    # Run full grid search with complete parameter space
    # Requirements 5.1, 6.1, 7.1: Full search spaces
    results = tuner.grid_search(
        context_windows=[3, 4, 5, 6, 7, 8, 10],  # Requirement 5.1
        rare_thresholds=[1, 2, 3, 4, 5],          # Requirement 6.1
        top_k_values=[1, 2, 3, 5, 8, 10],             # Requirement 7.1
        max_new_tokens=50,
        verbose=verbose
    )
    
    # Save domain-specific results
    output_file = f"task_10_5_{domain}_results.json"
    saved_path = tuner.save_tuning_results(results, filename=output_file)
    
    if verbose:
        print()
        print(f"Results saved to: {saved_path}")
        print()
    
    return results


def aggregate_results(
    all_domain_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate results across all domains to find universally good configurations.
    
    Args:
        all_domain_results: Results dictionary for each domain
        
    Returns:
        Aggregated analysis with cross-domain recommendations
        
    Requirements: 12.3, 12.5
    """
    print()
    print("=" * 80)
    print("CROSS-DOMAIN ANALYSIS")
    print("=" * 80)
    print()
    
    # Collect all configurations and their performance across domains
    config_performance: Dict[str, Dict[str, List[float]]] = {}
    
    for domain, results in all_domain_results.items():
        for result in results['all_results']:
            config_key = (
                result['config']['context_window'],
                result['config']['rare_token_threshold'],
                result['config']['top_k']
            )
            config_str = f"K={config_key[0]}, rare={config_key[1]}, top_k={config_key[2]}"
            
            if config_str not in config_performance:
                config_performance[config_str] = {
                    'config': result['config'],
                    'accuracies': [],
                    'speeds': []
                }
            
            config_performance[config_str]['accuracies'].append(result['accuracy'])
            config_performance[config_str]['speeds'].append(result['avg_inference_time_ms'])
    
    # Calculate average performance across domains
    aggregated = []
    for config_str, perf in config_performance.items():
        avg_accuracy = sum(perf['accuracies']) / len(perf['accuracies'])
        avg_speed = sum(perf['speeds']) / len(perf['speeds'])
        min_accuracy = min(perf['accuracies'])
        max_speed = max(perf['speeds'])
        
        aggregated.append({
            'config': perf['config'],
            'config_str': config_str,
            'avg_accuracy': avg_accuracy,
            'avg_speed': avg_speed,
            'min_accuracy': min_accuracy,
            'max_speed': max_speed,
            'accuracy_std': (sum((a - avg_accuracy) ** 2 for a in perf['accuracies']) / len(perf['accuracies'])) ** 0.5,
            'speed_std': (sum((s - avg_speed) ** 2 for s in perf['speeds']) / len(perf['speeds'])) ** 0.5
        })
    
    # Find best configurations for different priorities
    best_accuracy = max(aggregated, key=lambda x: x['avg_accuracy'])
    best_speed = min(aggregated, key=lambda x: x['avg_speed'])
    
    # Balanced: geometric mean of normalized accuracy and speed
    max_time = max(x['avg_speed'] for x in aggregated)
    best_balanced = max(
        aggregated,
        key=lambda x: (x['avg_accuracy'] * (1 - x['avg_speed'] / max_time)) ** 0.5
    )
    
    # Most robust: best minimum accuracy across domains
    best_robust = max(aggregated, key=lambda x: x['min_accuracy'])
    
    # Print summary
    print("Best Configurations Across All Domains:")
    print("-" * 80)
    print()
    
    print("1. BEST ACCURACY (maximize average accuracy):")
    print(f"   Config: {best_accuracy['config_str']}")
    print(f"   Avg Accuracy: {best_accuracy['avg_accuracy'] * 100:.1f}%")
    print(f"   Avg Speed: {best_accuracy['avg_speed']:.2f}ms")
    print(f"   Min Accuracy: {best_accuracy['min_accuracy'] * 100:.1f}%")
    print()
    
    print("2. BEST SPEED (minimize average inference time):")
    print(f"   Config: {best_speed['config_str']}")
    print(f"   Avg Accuracy: {best_speed['avg_accuracy'] * 100:.1f}%")
    print(f"   Avg Speed: {best_speed['avg_speed']:.2f}ms")
    print(f"   Max Speed: {best_speed['max_speed']:.2f}ms")
    print()
    
    print("3. BEST BALANCED (optimal accuracy-speed tradeoff):")
    print(f"   Config: {best_balanced['config_str']}")
    print(f"   Avg Accuracy: {best_balanced['avg_accuracy'] * 100:.1f}%")
    print(f"   Avg Speed: {best_balanced['avg_speed']:.2f}ms")
    print()
    
    print("4. MOST ROBUST (best worst-case accuracy):")
    print(f"   Config: {best_robust['config_str']}")
    print(f"   Avg Accuracy: {best_robust['avg_accuracy'] * 100:.1f}%")
    print(f"   Min Accuracy: {best_robust['min_accuracy'] * 100:.1f}%")
    print(f"   Avg Speed: {best_robust['avg_speed']:.2f}ms")
    print()
    
    return {
        'aggregated_performance': aggregated,
        'best_accuracy': best_accuracy,
        'best_speed': best_speed,
        'best_balanced': best_balanced,
        'best_robust': best_robust,
    }



def generate_documentation(
    aggregated_analysis: Dict[str, Any],
    all_domain_results: Dict[str, Dict[str, Any]]
) -> str:
    """
    Generate documentation of optimal hyperparameters.
    
    Args:
        aggregated_analysis: Cross-domain analysis results
        all_domain_results: Per-domain results
        
    Returns:
        Formatted documentation string
        
    Requirements: 5.3, 6.3, 7.3
    """
    recommended = aggregated_analysis['best_balanced']
    config = recommended['config']
    
    doc = []
    doc.append("=" * 80)
    doc.append("OPTIMAL HYPERPARAMETERS - TASK 10.5 RESULTS")
    doc.append("=" * 80)
    doc.append("")
    doc.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.append("")
    doc.append("RECOMMENDED DEFAULT CONFIGURATION (Balanced Priority):")
    doc.append("-" * 80)
    doc.append(f"  context_window: {config['context_window']}")
    doc.append(f"  rare_token_threshold: {config['rare_token_threshold']}")
    doc.append(f"  top_k: {config['top_k']}")
    doc.append("")
    doc.append("PERFORMANCE METRICS:")
    doc.append(f"  Average Accuracy: {recommended['avg_accuracy'] * 100:.1f}%")
    doc.append(f"  Average Speed: {recommended['avg_speed']:.2f}ms")
    doc.append(f"  Minimum Accuracy: {recommended['min_accuracy'] * 100:.1f}%")
    doc.append(f"  Maximum Speed: {recommended['max_speed']:.2f}ms")
    doc.append("")
    doc.append("RATIONALE:")
    doc.append("  This configuration provides the best balance between accuracy and speed")
    doc.append("  across all tested task types (code completion, classification, pattern")
    doc.append("  matching, and Q&A). It was selected from a comprehensive grid search")
    doc.append("  of 210 configurations per domain.")
    doc.append("")
    doc.append("ALTERNATIVE CONFIGURATIONS:")
    doc.append("-" * 80)
    doc.append("")
    
    best_acc = aggregated_analysis['best_accuracy']
    doc.append(f"For Maximum Accuracy (K={best_acc['config']['context_window']}, "
               f"rare={best_acc['config']['rare_token_threshold']}, "
               f"top_k={best_acc['config']['top_k']}):")
    doc.append(f"  - Accuracy: {best_acc['avg_accuracy'] * 100:.1f}%")
    doc.append(f"  - Speed: {best_acc['avg_speed']:.2f}ms")
    doc.append("")
    
    best_spd = aggregated_analysis['best_speed']
    doc.append(f"For Maximum Speed (K={best_spd['config']['context_window']}, "
               f"rare={best_spd['config']['rare_token_threshold']}, "
               f"top_k={best_spd['config']['top_k']}):")
    doc.append(f"  - Accuracy: {best_spd['avg_accuracy'] * 100:.1f}%")
    doc.append(f"  - Speed: {best_spd['avg_speed']:.2f}ms")
    doc.append("")
    
    doc.append("PER-DOMAIN BEST CONFIGURATIONS:")
    doc.append("-" * 80)
    doc.append("")
    
    for domain, results in all_domain_results.items():
        best_domain = results['best_accuracy_config']
        doc.append(f"{domain.replace('_', ' ').title()}:")
        doc.append(f"  Config: K={best_domain['config']['context_window']}, "
                   f"rare={best_domain['config']['rare_token_threshold']}, "
                   f"top_k={best_domain['config']['top_k']}")
        doc.append(f"  Accuracy: {best_domain['accuracy'] * 100:.1f}%")
        doc.append(f"  Speed: {best_domain['avg_inference_time_ms']:.2f}ms")
        doc.append("")
    
    doc.append("=" * 80)
    doc.append("IMPLEMENTATION NOTES")
    doc.append("=" * 80)
    doc.append("")
    doc.append("To apply these settings in SparseLogitGenerator:")
    doc.append("")
    doc.append("```python")
    doc.append("from puhl_luck._logit_generator import SparseLogitGenerator")
    doc.append("")
    doc.append("generator = SparseLogitGenerator(")
    doc.append(f"    repetition_window={config['context_window']},")
    doc.append(f"    rare_token_threshold={config['rare_token_threshold']},")
    doc.append(f"    top_k={config['top_k']},")
    doc.append(")")
    doc.append("```")
    doc.append("")
    doc.append("Note: The repetition_window parameter controls the context window")
    doc.append("      for repetition penalty, which aligns with the K parameter.")
    doc.append("")
    doc.append("=" * 80)
    
    return "\n".join(doc)


def main():
    """
    Execute Task 10.5: Run hyperparameter tuning and apply best configuration.
    
    Requirements: 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 7.1, 7.2, 7.3
    """
    print("=" * 80)
    print("TASK 10.5: HYPERPARAMETER TUNING AND OPTIMIZATION")
    print("=" * 80)
    print()
    print("This task will:")
    print("  1. Create comprehensive benchmark datasets")
    print("  2. Run full grid search (7 × 5 × 6 = 210 configs per domain)")
    print("  3. Identify optimal context_window, rare_threshold, and top_k")
    print("  4. Generate cross-domain recommendations")
    print("  5. Document optimal hyperparameters")
    print()
    print("Note: This will take several minutes to complete.")
    print()
    
    start_time = time.time()
    
    # Step 1: Create datasets
    print("Step 1: Creating benchmark datasets...")
    print("-" * 80)
    
    code_train, code_test = create_code_completion_data()
    print(f"✓ Code completion: {len(code_train)} train, {len(code_test)} test")
    
    class_train, class_test = create_classification_data()
    print(f"✓ Classification: {len(class_train)} train, {len(class_test)} test")
    
    pattern_train, pattern_test = create_pattern_matching_data()
    print(f"✓ Pattern matching: {len(pattern_train)} train, {len(pattern_test)} test")
    
    qa_train, qa_test = create_qa_data()
    print(f"✓ Q&A: {len(qa_train)} train, {len(qa_test)} test")
    print()
    
    # Step 2: Run grid searches for each domain
    print("Step 2: Running grid searches (this will take time)...")
    print("-" * 80)
    
    all_results = {}
    
    # Code completion
    all_results['code_completion'] = run_full_grid_search(
        'code_completion',
        code_train,
        code_test,
        verbose=True
    )
    
    # Classification
    all_results['classification'] = run_full_grid_search(
        'classification',
        class_train,
        class_test,
        verbose=True
    )
    
    # Pattern matching
    all_results['pattern_matching'] = run_full_grid_search(
        'pattern_matching',
        pattern_train,
        pattern_test,
        verbose=True
    )
    
    # Q&A
    all_results['qa'] = run_full_grid_search(
        'qa',
        qa_train,
        qa_test,
        verbose=True
    )
    
    # Step 3: Aggregate and analyze results
    print("Step 3: Analyzing results across all domains...")
    print("-" * 80)
    
    aggregated = aggregate_results(all_results)
    
    # Step 4: Generate documentation
    print()
    print("Step 4: Generating documentation...")
    print("-" * 80)
    
    documentation = generate_documentation(aggregated, all_results)
    
    # Save documentation
    doc_file = Path("task_10_5_optimal_hyperparameters.txt")
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(documentation)
    
    print(f"✓ Documentation saved to: {doc_file.absolute()}")
    print()
    
    # Save aggregated results
    aggregated_file = Path("task_10_5_aggregated_results.json")
    with open(aggregated_file, 'w', encoding='utf-8') as f:
        # Convert for JSON serialization
        json_safe = {
            'best_accuracy': aggregated['best_accuracy'],
            'best_speed': aggregated['best_speed'],
            'best_balanced': aggregated['best_balanced'],
            'best_robust': aggregated['best_robust'],
            'total_configurations_tested': len(aggregated['aggregated_performance']),
            'domains_tested': list(all_results.keys()),
        }
        json.dump(json_safe, f, indent=2)
    
    print(f"✓ Aggregated results saved to: {aggregated_file.absolute()}")
    print()
    
    # Step 5: Display final results
    elapsed_time = time.time() - start_time
    
    print()
    print("=" * 80)
    print("TASK 10.5 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print(f"Total execution time: {elapsed_time / 60:.1f} minutes")
    print()
    print(documentation)
    print()
    print("=" * 80)
    print("All results have been saved. Review the documentation file for")
    print("optimal hyperparameter configuration recommendations.")
    print("=" * 80)


if __name__ == '__main__':
    main()
