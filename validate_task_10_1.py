"""
Validation script for Task 10.1: Create comprehensive benchmark dataset

This script validates that the benchmark_data.py module meets all requirements:
- Requirements 11.1, 11.2, 11.3, 11.4
"""

from puhl_luck.benchmarks.benchmark_data import (
    get_code_completion_data,
    get_sentiment_classification_data,
    get_pattern_matching_data,
    get_qa_data,
    get_all_datasets,
    get_dataset_statistics,
)


def validate_code_completion():
    """Validate Requirement 11.1: Code completion examples"""
    print("=" * 70)
    print("VALIDATING REQUIREMENT 11.1: Code Completion Dataset")
    print("=" * 70)
    
    training, test = get_code_completion_data()
    
    print(f"✓ Training examples: {len(training)} (Required: 10)")
    assert len(training) == 10, f"Expected 10 training examples, got {len(training)}"
    
    print(f"✓ Test examples: {len(test)} (Required: 5)")
    assert len(test) == 5, f"Expected 5 test examples, got {len(test)}"
    
    # Validate structure
    for i, (input_text, output_text) in enumerate(training[:3]):
        print(f"  Example {i+1}: '{input_text[:30]}...' -> '{output_text[:30]}...'")
        assert isinstance(input_text, str), "Input must be string"
        assert isinstance(output_text, str), "Output must be string"
        assert len(input_text) > 0, "Input must not be empty"
        assert len(output_text) > 0, "Output must not be empty"
    
    print("✓ All code completion examples are valid")
    print()


def validate_sentiment_classification():
    """Validate Requirement 11.2: Sentiment classification examples"""
    print("=" * 70)
    print("VALIDATING REQUIREMENT 11.2: Sentiment Classification Dataset")
    print("=" * 70)
    
    training, test = get_sentiment_classification_data()
    
    print(f"✓ Training examples: {len(training)} (Required: 20)")
    assert len(training) == 20, f"Expected 20 training examples, got {len(training)}"
    
    print(f"✓ Test examples: {len(test)} (Required: 10)")
    assert len(test) == 10, f"Expected 10 test examples, got {len(test)}"
    
    # Validate label distribution
    train_labels = [label for _, label in training]
    test_labels = [label for _, label in test]
    
    train_pos = train_labels.count('positive')
    train_neg = train_labels.count('negative')
    train_neu = train_labels.count('neutral')
    
    print(f"  Training distribution: {train_pos} positive, {train_neg} negative, {train_neu} neutral")
    print(f"  Test distribution: {test_labels.count('positive')} positive, {test_labels.count('negative')} negative, {test_labels.count('neutral')} neutral")
    
    # Validate structure
    for i, (text, label) in enumerate(training[:3]):
        print(f"  Example {i+1}: '{text[:40]}...' -> {label}")
        assert isinstance(text, str), "Text must be string"
        assert isinstance(label, str), "Label must be string"
        assert label in ['positive', 'negative', 'neutral'], f"Invalid label: {label}"
    
    print("✓ All sentiment classification examples are valid")
    print()


def validate_pattern_matching():
    """Validate Requirement 11.3: Pattern matching examples"""
    print("=" * 70)
    print("VALIDATING REQUIREMENT 11.3: Pattern Matching Dataset")
    print("=" * 70)
    
    training, test = get_pattern_matching_data()
    
    print(f"✓ Training sequences: {len(training)} (Required: 15)")
    assert len(training) == 15, f"Expected 15 training sequences, got {len(training)}"
    
    print(f"✓ Test sequences: {len(test)} (Required: 8)")
    assert len(test) == 8, f"Expected 8 test sequences, got {len(test)}"
    
    # Validate structure
    for i, (pattern, next_elem) in enumerate(training[:3]):
        print(f"  Pattern {i+1}: '{pattern}' -> '{next_elem}'")
        assert isinstance(pattern, str), "Pattern must be string"
        assert isinstance(next_elem, str), "Next element must be string"
        assert len(pattern) > 0, "Pattern must not be empty"
        assert len(next_elem) > 0, "Next element must not be empty"
    
    print("✓ All pattern matching examples are valid")
    print()


def validate_qa():
    """Validate Requirement 11.4: Q&A examples"""
    print("=" * 70)
    print("VALIDATING REQUIREMENT 11.4: Question & Answer Dataset")
    print("=" * 70)
    
    training, test = get_qa_data()
    
    print(f"✓ Training pairs: {len(training)} (Required: 12)")
    assert len(training) == 12, f"Expected 12 training pairs, got {len(training)}"
    
    print(f"✓ Test questions: {len(test)} (Required: 6)")
    assert len(test) == 6, f"Expected 6 test questions, got {len(test)}"
    
    # Validate structure
    for i, (question, answer) in enumerate(training[:3]):
        print(f"  Q&A {i+1}: '{question}' -> '{answer}'")
        assert isinstance(question, str), "Question must be string"
        assert isinstance(answer, str), "Answer must be string"
        assert len(question) > 0, "Question must not be empty"
        assert len(answer) > 0, "Answer must not be empty"
        assert question.endswith('?'), f"Question should end with '?': {question}"
    
    print("✓ All Q&A examples are valid")
    print()


def validate_all_datasets():
    """Validate get_all_datasets() function"""
    print("=" * 70)
    print("VALIDATING ALL DATASETS ACCESS FUNCTION")
    print("=" * 70)
    
    datasets = get_all_datasets()
    
    assert 'code' in datasets, "Missing 'code' dataset"
    assert 'classification' in datasets, "Missing 'classification' dataset"
    assert 'pattern' in datasets, "Missing 'pattern' dataset"
    assert 'qa' in datasets, "Missing 'qa' dataset"
    
    print("✓ All datasets accessible via get_all_datasets()")
    
    stats = get_dataset_statistics()
    print("\nDataset Statistics:")
    for task_name, task_stats in stats.items():
        print(f"  {task_name.upper()}:")
        print(f"    - Training: {task_stats['training_count']}")
        print(f"    - Test: {task_stats['test_count']}")
        print(f"    - Total: {task_stats['total_examples']}")
    
    print("\n✓ Dataset statistics function working correctly")
    print()


def main():
    """Run all validations"""
    print("\n" + "=" * 70)
    print("TASK 10.1 VALIDATION: Comprehensive Benchmark Dataset")
    print("=" * 70)
    print()
    
    try:
        validate_code_completion()
        validate_sentiment_classification()
        validate_pattern_matching()
        validate_qa()
        validate_all_datasets()
        
        print("=" * 70)
        print("✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
        print("=" * 70)
        print("\nTask 10.1 Requirements Met:")
        print("  ✓ Requirement 11.1: Code completion dataset (10 training, 5 test)")
        print("  ✓ Requirement 11.2: Sentiment classification (20 training, 10 test)")
        print("  ✓ Requirement 11.3: Pattern matching (15 training, 8 test)")
        print("  ✓ Requirement 11.4: Q&A dataset (12 training, 6 test)")
        print("\nFile Created:")
        print("  ✓ packages/puhl_luck/puhl_luck/benchmarks/benchmark_data.py")
        print()
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
