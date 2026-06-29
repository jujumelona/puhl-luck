"""
Validation script for Task 10.6: Overfitting Prevention

This script validates Requirements 3.1 and 3.3:
- Requirement 3.1: Accuracy degradation <5% after learning new data
- Requirement 3.3: Consistent accuracy across training phases within 10%

Tests the HDC system's ability to retain earlier learned patterns
when new training data is introduced (catastrophic forgetting prevention).
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add package to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "puhl_luck"))

from puhl_luck.brain_memory import BrainMemory


def create_dataset_a() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Create Dataset A: Basic arithmetic functions
    
    Returns:
        (training_pairs, test_pairs)
    """
    training = [
        ("def add(a, b):", "def add(a, b):\n    return a + b"),
        ("def subtract(x, y):", "def subtract(x, y):\n    return x - y"),
        ("def multiply(a, b):", "def multiply(a, b):\n    return a * b"),
        ("def divide(a, b):", "def divide(a, b):\n    return a / b"),
        ("def modulo(x, y):", "def modulo(x, y):\n    return x % y"),
    ]
    
    test = [
        ("def add(x, y):", "return x + y"),
        ("def subtract(a, b):", "return a - b"),
        ("def multiply(x, y):", "return x * y"),
    ]
    
    return training, test


def create_dataset_b() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Create Dataset B: Boolean/comparison functions
    
    Returns:
        (training_pairs, test_pairs)
    """
    training = [
        ("def is_even(x):", "def is_even(x):\n    return x % 2 == 0"),
        ("def is_odd(n):", "def is_odd(n):\n    return n % 2 == 1"),
        ("def is_positive(n):", "def is_positive(n):\n    return n > 0"),
        ("def is_negative(x):", "def is_negative(x):\n    return x < 0"),
        ("def is_zero(n):", "def is_zero(n):\n    return n == 0"),
    ]
    
    test = [
        ("def is_even(n):", "return n % 2 == 0"),
        ("def is_positive(x):", "return x > 0"),
    ]
    
    return training, test


def create_dataset_c() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Create Dataset C: Utility/helper functions
    
    Returns:
        (training_pairs, test_pairs)
    """
    training = [
        ("def square(n):", "def square(n):\n    return n * n"),
        ("def cube(x):", "def cube(x):\n    return x * x * x"),
        ("def abs_val(x):", "def abs_val(x):\n    return x if x >= 0 else -x"),
        ("def negate(n):", "def negate(n):\n    return -n"),
        ("def double(n):", "def double(n):\n    return n * 2"),
    ]
    
    test = [
        ("def square(x):", "return x * x"),
        ("def abs_val(n):", "return n if n >= 0 else -n"),
    ]
    
    return training, test


def train_on_dataset(
    brain: BrainMemory,
    dataset: List[Tuple[str, str]],
    domain: str,
    verbose: bool = True
) -> float:
    """
    Train brain on a dataset.
    
    Args:
        brain: BrainMemory instance
        dataset: List of (input, target) pairs
        domain: Domain identifier
        verbose: Whether to print progress
        
    Returns:
        Training time in milliseconds
    """
    if verbose:
        print(f"  Training on {len(dataset)} examples...")
    
    start_time = time.time()
    
    for input_text, target_text in dataset:
        brain.expose_pair(
            partial=input_text,
            complete=target_text,
            domain=domain,
            modality=domain,
        )
    
    training_time_ms = (time.time() - start_time) * 1000
    
    if verbose:
        print(f"  Training completed in {training_time_ms:.2f}ms")
    
    return training_time_ms


def evaluate_accuracy(
    brain: BrainMemory,
    test_data: List[Tuple[str, str]],
    domain: str,
    max_new_tokens: int = 64,
    verbose: bool = False
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluate brain accuracy on test dataset.
    
    Args:
        brain: Trained BrainMemory instance
        test_data: List of (input, expected_output) pairs
        domain: Domain identifier
        max_new_tokens: Maximum tokens to generate
        verbose: Whether to print per-example results
        
    Returns:
        (accuracy, details_list)
    """
    total_tests = len(test_data)
    passed = 0
    details = []
    
    for input_text, expected_output in test_data:
        try:
            # Generate output
            result = brain.generate(
                query=input_text,
                max_new_tokens=max_new_tokens,
                domain=domain,
                return_metrics=False,
            )
            
            # Extract generated text
            if isinstance(result, tuple):
                generated = result[0]
            else:
                generated = result
            
            # Evaluate correctness (token-based matching)
            expected_tokens = set(expected_output.lower().split())
            generated_tokens = set(generated.lower().split())
            overlap = len(expected_tokens & generated_tokens)
            
            # Pass if >50% token overlap
            correct = overlap >= len(expected_tokens) * 0.5
            
            if correct:
                passed += 1
            
            if verbose:
                print(f"    Input: {input_text[:50]}")
                print(f"    Expected: {expected_output[:50]}")
                print(f"    Generated: {generated[:50]}")
                print(f"    Correct: {correct}")
                print()
            
            details.append({
                "input": input_text,
                "expected": expected_output,
                "generated": generated,
                "correct": correct,
            })
            
        except Exception as e:
            if verbose:
                print(f"    Input: {input_text[:50]}")
                print(f"    Error: {e}")
                print()
            
            details.append({
                "input": input_text,
                "expected": expected_output,
                "generated": "",
                "correct": False,
                "error": str(e),
            })
    
    accuracy = passed / total_tests if total_tests > 0 else 0.0
    return accuracy, details


def validate_overfitting_prevention(verbose: bool = True) -> Dict[str, Any]:
    """
    Validate overfitting prevention by training on sequential datasets A, B, C
    and measuring accuracy degradation on dataset A.
    
    Validates:
    - Requirement 3.1: Accuracy degradation <5% after learning new data
    - Requirement 3.3: Consistent accuracy across training phases within 10%
    
    Args:
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with validation results and pass/fail status
    """
    if verbose:
        print("=" * 80)
        print("TASK 10.6: OVERFITTING PREVENTION VALIDATION")
        print("=" * 80)
        print()
    
    # Create datasets
    dataset_a_train, dataset_a_test = create_dataset_a()
    dataset_b_train, dataset_b_test = create_dataset_b()
    dataset_c_train, dataset_c_test = create_dataset_c()
    
    if verbose:
        print(f"Dataset A: {len(dataset_a_train)} training, {len(dataset_a_test)} test")
        print(f"Dataset B: {len(dataset_b_train)} training, {len(dataset_b_test)} test")
        print(f"Dataset C: {len(dataset_c_train)} training, {len(dataset_c_test)} test")
        print()
    
    # Initialize brain
    brain = BrainMemory()
    domain = "code"
    
    # ========================================================================
    # PHASE 1: Train on Dataset A and measure baseline accuracy
    # ========================================================================
    if verbose:
        print("[PHASE 1] Train on Dataset A")
        print("-" * 80)
    
    train_on_dataset(brain, dataset_a_train, domain, verbose)
    
    if verbose:
        print("  Evaluating on Dataset A test set...")
    
    accuracy_a_baseline, _ = evaluate_accuracy(brain, dataset_a_test, domain, verbose=False)
    
    if verbose:
        print(f"  Baseline accuracy on Dataset A: {accuracy_a_baseline * 100:.1f}%")
        print()
    
    # ========================================================================
    # PHASE 2: Train on Dataset B and re-measure accuracy on A
    # ========================================================================
    if verbose:
        print("[PHASE 2] Train on Dataset B")
        print("-" * 80)
    
    train_on_dataset(brain, dataset_b_train, domain, verbose)
    
    if verbose:
        print("  Evaluating on Dataset A test set (after B)...")
    
    accuracy_a_after_b, _ = evaluate_accuracy(brain, dataset_a_test, domain, verbose=False)
    
    if verbose:
        print(f"  Accuracy on Dataset A after B: {accuracy_a_after_b * 100:.1f}%")
        print(f"  Degradation: {(accuracy_a_baseline - accuracy_a_after_b) * 100:.1f}%")
    
    # Also measure accuracy on B for comparison
    if verbose:
        print("  Evaluating on Dataset B test set...")
    
    accuracy_b_after_b, _ = evaluate_accuracy(brain, dataset_b_test, domain, verbose=False)
    
    if verbose:
        print(f"  Accuracy on Dataset B: {accuracy_b_after_b * 100:.1f}%")
        print()
    
    # ========================================================================
    # PHASE 3: Train on Dataset C and re-measure accuracy on A
    # ========================================================================
    if verbose:
        print("[PHASE 3] Train on Dataset C")
        print("-" * 80)
    
    train_on_dataset(brain, dataset_c_train, domain, verbose)
    
    if verbose:
        print("  Evaluating on Dataset A test set (after C)...")
    
    accuracy_a_after_c, _ = evaluate_accuracy(brain, dataset_a_test, domain, verbose=False)
    
    if verbose:
        print(f"  Accuracy on Dataset A after C: {accuracy_a_after_c * 100:.1f}%")
        print(f"  Total degradation: {(accuracy_a_baseline - accuracy_a_after_c) * 100:.1f}%")
    
    # Also measure accuracy on B and C for comparison
    if verbose:
        print("  Evaluating on Dataset B test set (after C)...")
    
    accuracy_b_after_c, _ = evaluate_accuracy(brain, dataset_b_test, domain, verbose=False)
    
    if verbose:
        print(f"  Accuracy on Dataset B after C: {accuracy_b_after_c * 100:.1f}%")
    
    if verbose:
        print("  Evaluating on Dataset C test set...")
    
    accuracy_c_after_c, _ = evaluate_accuracy(brain, dataset_c_test, domain, verbose=False)
    
    if verbose:
        print(f"  Accuracy on Dataset C: {accuracy_c_after_c * 100:.1f}%")
        print()
    
    # ========================================================================
    # VALIDATION: Check Requirements
    # ========================================================================
    if verbose:
        print("=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
    
    # Requirement 3.1: Accuracy degradation <5%
    degradation_percentage = (accuracy_a_baseline - accuracy_a_after_c) * 100
    req_3_1_pass = degradation_percentage < 5.0
    
    if verbose:
        print(f"\nRequirement 3.1: Accuracy degradation <5%")
        print(f"  Baseline accuracy on A: {accuracy_a_baseline * 100:.1f}%")
        print(f"  Final accuracy on A: {accuracy_a_after_c * 100:.1f}%")
        print(f"  Degradation: {degradation_percentage:.1f}%")
        print(f"  Status: {'PASS' if req_3_1_pass else 'FAIL'} ✓" if req_3_1_pass else f"  Status: FAIL ✗")
    
    # Requirement 3.3: Consistent accuracy across training phases within 10%
    all_accuracies = [
        accuracy_a_after_c,  # A after C
        accuracy_b_after_c,  # B after C
        accuracy_c_after_c,  # C after C
    ]
    
    min_accuracy = min(all_accuracies)
    max_accuracy = max(all_accuracies)
    accuracy_variance = (max_accuracy - min_accuracy) * 100
    req_3_3_pass = accuracy_variance <= 10.0
    
    if verbose:
        print(f"\nRequirement 3.3: Consistent accuracy within 10% across phases")
        print(f"  Accuracy on A (after C): {accuracy_a_after_c * 100:.1f}%")
        print(f"  Accuracy on B (after C): {accuracy_b_after_c * 100:.1f}%")
        print(f"  Accuracy on C (after C): {accuracy_c_after_c * 100:.1f}%")
        print(f"  Min: {min_accuracy * 100:.1f}%, Max: {max_accuracy * 100:.1f}%")
        print(f"  Variance: {accuracy_variance:.1f}%")
        print(f"  Status: {'PASS' if req_3_3_pass else 'FAIL'} ✓" if req_3_3_pass else f"  Status: FAIL ✗")
    
    # Overall validation status
    overall_pass = req_3_1_pass and req_3_3_pass
    
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"OVERALL: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
        print(f"{'=' * 80}")
    
    # Return results
    results = {
        "overall_pass": overall_pass,
        "requirement_3_1": {
            "pass": req_3_1_pass,
            "baseline_accuracy": accuracy_a_baseline,
            "final_accuracy": accuracy_a_after_c,
            "degradation_percentage": degradation_percentage,
            "threshold": 5.0,
        },
        "requirement_3_3": {
            "pass": req_3_3_pass,
            "accuracy_a_after_c": accuracy_a_after_c,
            "accuracy_b_after_c": accuracy_b_after_c,
            "accuracy_c_after_c": accuracy_c_after_c,
            "min_accuracy": min_accuracy,
            "max_accuracy": max_accuracy,
            "variance_percentage": accuracy_variance,
            "threshold": 10.0,
        },
        "detailed_accuracies": {
            "phase_1_a_baseline": accuracy_a_baseline,
            "phase_2_a_after_b": accuracy_a_after_b,
            "phase_2_b_after_b": accuracy_b_after_b,
            "phase_3_a_after_c": accuracy_a_after_c,
            "phase_3_b_after_c": accuracy_b_after_c,
            "phase_3_c_after_c": accuracy_c_after_c,
        },
    }
    
    return results


def main():
    """Main entry point for validation script."""
    try:
        results = validate_overfitting_prevention(verbose=True)
        
        # Exit with appropriate code
        exit_code = 0 if results["overall_pass"] else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\nERROR: Validation failed with exception:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
