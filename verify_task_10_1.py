"""
Verification script for Task 10.1: Create comprehensive benchmark dataset

This script verifies that the benchmark_data.py file contains all required datasets:
- Code completion: 10 training functions, 5 test cases (Requirement 11.1)
- Sentiment classification: 20 training examples, 10 test examples (Requirement 11.2)
- Pattern matching: 15 training sequences, 8 test cases (Requirement 11.3)
- Q&A: 12 training pairs, 6 test questions (Requirement 11.4)
"""

import sys
sys.path.insert(0, 'packages/puhl_luck')

from puhl_luck.benchmarks.benchmark_data import (
    get_code_completion_data,
    get_sentiment_classification_data,
    get_pattern_matching_data,
    get_qa_data,
    get_dataset_statistics,
)


def verify_task_10_1():
    """Verify that Task 10.1 requirements are met."""
    
    print("=" * 70)
    print("Task 10.1 Verification: Create comprehensive benchmark dataset")
    print("=" * 70)
    print()
    
    # Get all datasets
    code_train, code_test = get_code_completion_data()
    sentiment_train, sentiment_test = get_sentiment_classification_data()
    pattern_train, pattern_test = get_pattern_matching_data()
    qa_train, qa_test = get_qa_data()
    
    # Track verification results
    all_passed = True
    
    # Verify code completion dataset (Requirement 11.1)
    print("1. Code Completion Dataset (Requirement 11.1)")
    print(f"   Training examples: {len(code_train)} (expected: 10)")
    print(f"   Test examples: {len(code_test)} (expected: 5)")
    if len(code_train) >= 10 and len(code_test) >= 5:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED")
        all_passed = False
    print()
    
    # Verify sentiment classification dataset (Requirement 11.2)
    print("2. Sentiment Classification Dataset (Requirement 11.2)")
    print(f"   Training examples: {len(sentiment_train)} (expected: 20)")
    print(f"   Test examples: {len(sentiment_test)} (expected: 10)")
    if len(sentiment_train) >= 20 and len(sentiment_test) >= 10:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED")
        all_passed = False
    print()
    
    # Verify pattern matching dataset (Requirement 11.3)
    print("3. Pattern Matching Dataset (Requirement 11.3)")
    print(f"   Training examples: {len(pattern_train)} (expected: 15)")
    print(f"   Test examples: {len(pattern_test)} (expected: 8)")
    if len(pattern_train) >= 15 and len(pattern_test) >= 8:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED")
        all_passed = False
    print()
    
    # Verify Q&A dataset (Requirement 11.4)
    print("4. Question & Answer Dataset (Requirement 11.4)")
    print(f"   Training examples: {len(qa_train)} (expected: 12)")
    print(f"   Test examples: {len(qa_test)} (expected: 6)")
    if len(qa_train) >= 12 and len(qa_test) >= 6:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED")
        all_passed = False
    print()
    
    # Verify data structure integrity
    print("5. Data Structure Integrity")
    structure_valid = True
    
    # Check code completion structure
    if code_train and len(code_train[0]) == 2:
        print("   Code completion structure: ✓ (input, output) tuples")
    else:
        print("   Code completion structure: ✗ Invalid format")
        structure_valid = False
    
    # Check sentiment classification structure
    if sentiment_train and len(sentiment_train[0]) == 2:
        print("   Sentiment classification structure: ✓ (text, label) tuples")
    else:
        print("   Sentiment classification structure: ✗ Invalid format")
        structure_valid = False
    
    # Check pattern matching structure
    if pattern_train and len(pattern_train[0]) == 2:
        print("   Pattern matching structure: ✓ (sequence, next) tuples")
    else:
        print("   Pattern matching structure: ✗ Invalid format")
        structure_valid = False
    
    # Check Q&A structure
    if qa_train and len(qa_train[0]) == 2:
        print("   Q&A structure: ✓ (question, answer) tuples")
    else:
        print("   Q&A structure: ✗ Invalid format")
        structure_valid = False
    
    if structure_valid:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED")
        all_passed = False
    print()
    
    # Display dataset statistics
    print("6. Dataset Statistics")
    stats = get_dataset_statistics()
    for task_name, task_stats in stats.items():
        print(f"   {task_name}: {task_stats['training_count']} training, "
              f"{task_stats['test_count']} test, "
              f"{task_stats['total_examples']} total")
    print()
    
    # Show sample data from each dataset
    print("7. Sample Data (First Example from Each Dataset)")
    print()
    print(f"   Code Completion:")
    print(f"      Input: {code_train[0][0][:50]}...")
    print(f"      Output: {code_train[0][1][:50]}...")
    print()
    print(f"   Sentiment Classification:")
    print(f"      Text: {sentiment_train[0][0][:50]}...")
    print(f"      Label: {sentiment_train[0][1]}")
    print()
    print(f"   Pattern Matching:")
    print(f"      Sequence: {pattern_train[0][0]}")
    print(f"      Next: {pattern_train[0][1]}")
    print()
    print(f"   Q&A:")
    print(f"      Question: {qa_train[0][0]}")
    print(f"      Answer: {qa_train[0][1]}")
    print()
    
    # Final result
    print("=" * 70)
    if all_passed:
        print("TASK 10.1 VERIFICATION: ✓ ALL CHECKS PASSED")
        print()
        print("The comprehensive benchmark dataset has been successfully created with:")
        print("  - 10 code completion training functions + 5 test cases")
        print("  - 20 sentiment classification training examples + 10 test examples")
        print("  - 15 pattern matching training sequences + 8 test cases")
        print("  - 12 Q&A training pairs + 6 test questions")
        print()
        print("Requirements validated: 11.1, 11.2, 11.3, 11.4")
    else:
        print("TASK 10.1 VERIFICATION: ✗ SOME CHECKS FAILED")
        return 1
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(verify_task_10_1())
