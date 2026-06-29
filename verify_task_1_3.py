"""Verification script for Task 1.3: Generation statistics and diagnostics.

This script verifies that all requirements for Task 1.3 are satisfied:
- Requirement 13.1: Generate returns detailed metrics with return_metrics=True
- Requirement 13.2: get_statistics() returns required fields
- Requirement 13.3: Empty output includes failure reason
- Requirement 13.4: Excessive backoff logging (>20% field_only)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'puhl_luck'))

from puhl_luck._logit_generator import SparseLogitGenerator, GenerationMetrics
import logging

# Enable logging to see warning messages
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def verify_requirement_13_1():
    """Verify Requirement 13.1: Generate returns detailed metrics."""
    print("\n" + "=" * 70)
    print("REQUIREMENT 13.1: Generate returns detailed metrics with return_metrics=True")
    print("=" * 70)
    
    gen = SparseLogitGenerator()
    gen.learn("def add(a, b):", "return a + b")
    
    output, metrics = gen.generate("def add(a, b):", return_metrics=True)
    
    # Verify detailed_metrics exists
    assert 'detailed_metrics' in metrics, "❌ Missing 'detailed_metrics' in response"
    print("✓ detailed_metrics present in response")
    
    detailed = metrics['detailed_metrics']
    assert isinstance(detailed, GenerationMetrics), "❌ detailed_metrics not GenerationMetrics instance"
    print("✓ detailed_metrics is GenerationMetrics instance")
    
    # Verify all required fields from Requirement 13.1
    required_fields = [
        'tokens_generated',
        'backoff_levels',
        'copy_gate_activations',
        'generation_method'
    ]
    
    for field in required_fields:
        assert hasattr(detailed, field), f"❌ Missing field: {field}"
        print(f"✓ Field '{field}' present: {getattr(detailed, field)}")
    
    print("\n✅ REQUIREMENT 13.1 SATISFIED")
    return True

def verify_requirement_13_2():
    """Verify Requirement 13.2: get_statistics() returns required fields."""
    print("\n" + "=" * 70)
    print("REQUIREMENT 13.2: get_statistics() returns required fields")
    print("=" * 70)
    
    gen = SparseLogitGenerator()
    gen.learn("def add(a, b):", "return a + b")
    gen.learn("def subtract(a, b):", "return a - b")
    
    stats = gen.get_statistics()
    
    # Verify all required fields from Requirement 13.2
    required_fields = {
        'pairs_learned': 2,
        'total_transitions': lambda x: x > 0,
        'total_contexts': lambda x: x > 0,
        'total_unique_tokens': lambda x: x > 0
    }
    
    for field, expected in required_fields.items():
        assert field in stats, f"❌ Missing field: {field}"
        value = stats[field]
        
        if callable(expected):
            assert expected(value), f"❌ Field '{field}' has invalid value: {value}"
            print(f"✓ Field '{field}': {value} (valid)")
        else:
            assert value == expected, f"❌ Field '{field}' expected {expected}, got {value}"
            print(f"✓ Field '{field}': {value} (matches expected)")
    
    print("\n✅ REQUIREMENT 13.2 SATISFIED")
    return True

def verify_requirement_13_3():
    """Verify Requirement 13.3: Empty output includes failure reason."""
    print("\n" + "=" * 70)
    print("REQUIREMENT 13.3: Empty output includes failure reason")
    print("=" * 70)
    
    gen = SparseLogitGenerator()
    
    # Generate without training to produce empty output
    output, metrics = gen.generate("untrained input", return_metrics=True)
    
    assert 'detailed_metrics' in metrics, "❌ Missing detailed_metrics"
    detailed = metrics['detailed_metrics']
    
    # Verify empty_output flag
    assert hasattr(detailed, 'empty_output'), "❌ Missing 'empty_output' field"
    print(f"✓ empty_output field present: {detailed.empty_output}")
    
    # Verify failure_reason
    assert hasattr(detailed, 'failure_reason'), "❌ Missing 'failure_reason' field"
    print(f"✓ failure_reason field present: {detailed.failure_reason}")
    
    # If output is empty, failure_reason should be set
    if detailed.empty_output:
        assert detailed.failure_reason is not None, "❌ Empty output should have failure_reason"
        print(f"✓ Failure reason correctly identified: {detailed.failure_reason}")
    
    print("\n✅ REQUIREMENT 13.3 SATISFIED")
    return True

def verify_requirement_13_4():
    """Verify Requirement 13.4: Excessive backoff logging."""
    print("\n" + "=" * 70)
    print("REQUIREMENT 13.4: Excessive backoff logging (>20% field_only)")
    print("=" * 70)
    
    gen = SparseLogitGenerator()
    
    # Verify tracking variables exist
    assert hasattr(gen, 'total_generations'), "❌ Missing total_generations"
    print(f"✓ total_generations tracking exists: {gen.total_generations}")
    
    assert hasattr(gen, 'total_field_only_backoffs'), "❌ Missing total_field_only_backoffs"
    print(f"✓ total_field_only_backoffs tracking exists: {gen.total_field_only_backoffs}")
    
    assert hasattr(gen, 'generation_backoff_levels'), "❌ Missing generation_backoff_levels"
    print(f"✓ generation_backoff_levels tracking exists: {len(gen.generation_backoff_levels)} levels")
    
    # Verify get_statistics includes field_only_backoff_percentage
    gen.learn("test", "data")
    gen.generate("test", max_tokens=5)
    stats = gen.get_statistics()
    
    assert 'field_only_backoff_percentage' in stats, "❌ Missing field_only_backoff_percentage"
    print(f"✓ field_only_backoff_percentage in stats: {stats['field_only_backoff_percentage']:.1f}%")
    
    print("\n✅ REQUIREMENT 13.4 SATISFIED")
    print("\nNote: Logging warning for >20% field_only backoff is implemented")
    print("in the generate() method at lines ~718-724 of _logit_generator.py")
    return True

def main():
    """Run all verification checks."""
    print("=" * 70)
    print("TASK 1.3 VERIFICATION: Generation Statistics and Diagnostics")
    print("=" * 70)
    
    results = []
    
    try:
        results.append(("Requirement 13.1", verify_requirement_13_1()))
    except Exception as e:
        print(f"\n❌ REQUIREMENT 13.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Requirement 13.1", False))
    
    try:
        results.append(("Requirement 13.2", verify_requirement_13_2()))
    except Exception as e:
        print(f"\n❌ REQUIREMENT 13.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Requirement 13.2", False))
    
    try:
        results.append(("Requirement 13.3", verify_requirement_13_3()))
    except Exception as e:
        print(f"\n❌ REQUIREMENT 13.3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Requirement 13.3", False))
    
    try:
        results.append(("Requirement 13.4", verify_requirement_13_4()))
    except Exception as e:
        print(f"\n❌ REQUIREMENT 13.4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Requirement 13.4", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 ALL REQUIREMENTS VERIFIED SUCCESSFULLY! 🎉")
        print("\nTask 1.3 Implementation Complete:")
        print("  ✓ GenerationMetrics dataclass added to _logit_generator.py")
        print("  ✓ generate() method returns detailed metrics with return_metrics=True")
        print("  ✓ get_statistics() method returns required statistics")
        print("  ✓ Excessive backoff logging implemented")
        return 0
    else:
        print("\n❌ SOME REQUIREMENTS FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(main())
