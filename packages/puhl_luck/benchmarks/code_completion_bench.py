"""
Custom code completion benchmark for PUHL-LUCK.

Tests token-level generation accuracy on various Python patterns:
- Function definitions
- Loops and conditionals
- List comprehensions
- Common idioms
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from puhl_luck.brain_memory import BrainMemory


# Training examples: (partial, complete) pairs
TRAINING_EXAMPLES = [
    # Basic functions
    ("def add(a, b):", "def add(a, b):\n    return a + b"),
    ("def multiply(x, y):", "def multiply(x, y):\n    return x * y"),
    ("def square(n):", "def square(n):\n    return n * n"),
    
    # Conditionals
    ("def is_positive(x):\n    if x > 0:", "def is_positive(x):\n    if x > 0:\n        return True\n    else:\n        return False"),
    ("def max_of_two(a, b):\n    if a > b:", "def max_of_two(a, b):\n    if a > b:\n        return a\n    else:\n        return b"),
    
    # Loops
    ("def sum_list(numbers):\n    total = 0\n    for n in numbers:", "def sum_list(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total"),
    ("def count_evens(nums):\n    count = 0\n    for x in nums:\n        if x % 2 == 0:", "def count_evens(nums):\n    count = 0\n    for x in nums:\n        if x % 2 == 0:\n            count += 1\n    return count"),
    
    # List comprehensions
    ("def squares(n):\n    return [x", "def squares(n):\n    return [x**2 for x in range(n)]"),
    ("def evens(n):\n    return [x for x in range(n) if", "def evens(n):\n    return [x for x in range(n) if x % 2 == 0]"),
    
    # String operations
    ("def uppercase(s):\n    return", "def uppercase(s):\n    return s.upper()"),
    ("def reverse(s):\n    return", "def reverse(s):\n    return s[::-1]"),
    
    # Common patterns
    ("def factorial(n):\n    if n <= 1:\n        return 1\n    else:", "def factorial(n):\n    if n <= 1:\n        return 1\n    else:\n        return n * factorial(n - 1)"),
    ("def fibonacci(n):\n    if n <= 1:\n        return n\n    return", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"),
    
    # List operations
    ("def last_element(lst):\n    return lst[-1", "def last_element(lst):\n    return lst[-1]"),
    ("def first_half(lst):\n    mid = len(lst) // 2\n    return", "def first_half(lst):\n    mid = len(lst) // 2\n    return lst[:mid]"),
]


# Test cases: (partial, expected_tokens, description)
TEST_CASES = [
    {
        "partial": "def subtract(a, b):",
        "expected_contains": ["return", "a", "-", "b"],
        "description": "Basic subtraction function",
        "validate": lambda code: "return a - b" in code or "return a-b" in code,
    },
    {
        "partial": "def divide(x, y):",
        "expected_contains": ["return", "x", "/", "y"],
        "description": "Basic division function",
        "validate": lambda code: "return x / y" in code or "return x/y" in code,
    },
    {
        "partial": "def is_even(n):\n    return n % 2 ==",
        "expected_contains": ["0"],
        "description": "Even number check",
        "validate": lambda code: "== 0" in code,
    },
    {
        "partial": "def double_list(nums):\n    return [x * 2",
        "expected_contains": ["for", "x", "in", "nums"],
        "description": "List comprehension doubling",
        "validate": lambda code: "for x in nums" in code,
    },
    {
        "partial": "def abs_value(x):\n    if x < 0:\n        return",
        "expected_contains": ["-", "x"],
        "description": "Absolute value negative case",
        "validate": lambda code: "return -x" in code or "return - x" in code,
    },
]


def train_code_completion(brain: BrainMemory) -> None:
    """Train on code completion examples."""
    print(f"Training on {len(TRAINING_EXAMPLES)} code examples...")
    
    for i, (partial, complete) in enumerate(TRAINING_EXAMPLES):
        brain.expose_pair(
            partial=partial,
            complete=complete,
            domain="code",
            modality="code",
            source="code_completion_train",
        )
        
        if (i + 1) % 5 == 0:
            print(f"  Trained {i + 1}/{len(TRAINING_EXAMPLES)}")
    
    # Print token stats
    if hasattr(brain._transition_layer, "get_token_transition_stats"):
        stats = brain._transition_layer.get_token_transition_stats()
        print(f"\nToken transition stats:")
        print(f"  Contexts: {stats['total_contexts']}")
        print(f"  Transitions: {stats['total_transitions']}")
        print(f"  Avg fanout: {stats['avg_fanout']}")


def evaluate_completion(
    brain: BrainMemory,
    test_case: Dict[str, Any],
    use_token_generation: bool = True,
) -> Tuple[bool, str, str]:
    """
    Evaluate a single completion test case.
    
    Returns:
        (passed, status, generated_code)
    """
    partial = test_case["partial"]
    expected_contains = test_case["expected_contains"]
    validate_fn = test_case.get("validate")
    
    # Generate completion
    try:
        if use_token_generation:
            generated = brain.generate_code(
                prompt=partial,
                max_tokens=128,
                validate_syntax=False,  # Allow partial completions
            )
        else:
            generated = brain.generate(
                prompt=partial,
                max_new_tokens=128,
            )
    except Exception as e:
        return False, "generation_error", str(e)
    
    # Check if expected tokens present
    tokens_present = sum(1 for tok in expected_contains if tok in generated)
    token_coverage = tokens_present / len(expected_contains) if expected_contains else 0.0
    
    # Custom validation if provided
    if validate_fn:
        try:
            if validate_fn(generated):
                return True, "pass", generated
        except Exception:
            pass
    
    # Check token coverage
    if token_coverage >= 0.75:
        return True, "pass", generated
    elif token_coverage >= 0.5:
        return False, "partial", generated
    else:
        return False, "fail", generated


def run_code_completion_benchmark(
    use_token_generation: bool = True,
) -> Dict[str, Any]:
    """Run custom code completion benchmark."""
    print("=" * 60)
    print("Code Completion Benchmark")
    print("=" * 60)
    
    # Initialize brain
    brain = BrainMemory()
    
    # Train
    train_code_completion(brain)
    
    # Evaluate
    print(f"\nEvaluating {len(TEST_CASES)} test cases...")
    
    results = {
        "total": len(TEST_CASES),
        "passed": 0,
        "partial": 0,
        "failed": 0,
        "generation_errors": 0,
        "details": [],
    }
    
    for i, test_case in enumerate(TEST_CASES):
        description = test_case["description"]
        passed, status, generated = evaluate_completion(
            brain, test_case, use_token_generation
        )
        
        results["details"].append({
            "description": description,
            "partial": test_case["partial"],
            "generated": generated,
            "status": status,
        })
        
        if status == "pass":
            results["passed"] += 1
        elif status == "partial":
            results["partial"] += 1
        elif status == "generation_error":
            results["generation_errors"] += 1
        else:
            results["failed"] += 1
        
        print(f"  [{i+1}/{len(TEST_CASES)}] {description}: {status}")
        if status != "pass":
            print(f"    Generated: {generated[:80]}...")
    
    # Metrics
    results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Total: {results['total']}")
    print(f"  Passed: {results['passed']} ({results['pass_rate']:.1%})")
    print(f"  Partial: {results['partial']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Generation errors: {results['generation_errors']}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_code_completion_benchmark(use_token_generation=True)
