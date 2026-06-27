"""
HumanEval benchmark for PUHL-LUCK code generation.

HumanEval: 164 hand-written programming problems with unit tests.
Paper: "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from puhl_luck.brain_memory import BrainMemory


def load_humaneval_problems(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load HumanEval problems from JSONL file.
    
    If no path provided, returns a small subset of sample problems.
    
    Returns:
        List of problem dicts with keys: task_id, prompt, canonical_solution, test, entry_point
    """
    if path and path.exists():
        problems = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        return problems
    
    # Sample problems for testing
    return [
        {
            "task_id": "HumanEval/0",
            "prompt": '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
            "entry_point": "has_close_elements",
            "canonical_solution": '''    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
''',
            "test": '''

def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False


check(has_close_elements)
''',
        },
        {
            "task_id": "HumanEval/1",
            "prompt": '''from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
''',
            "entry_point": "separate_paren_groups",
            "canonical_solution": '''    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()

    return result
''',
            "test": '''

def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == [
        '(()())', '((()))', '()', '((())()())'
    ]
    assert candidate('() (()) ((())) (((())))') == [
        '()', '(())', '((()))', '(((())))'
    ]
    assert candidate('(()(())((())))') == [
        '(()(())((()))))'
    ]
    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']

check(separate_paren_groups)
''',
        },
        {
            "task_id": "HumanEval/2",
            "prompt": '''

def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
''',
            "entry_point": "truncate_number",
            "canonical_solution": '''    return number % 1.0
''',
            "test": '''

def check(candidate):
    assert candidate(3.5) == 0.5
    assert abs(candidate(1.33) - 0.33) < 1e-6
    assert abs(candidate(123.456) - 0.456) < 1e-6

check(truncate_number)
''',
        },
    ]


def train_on_problems(brain: BrainMemory, problems: List[Dict[str, Any]], num_train: int = 10) -> None:
    """
    Train the memory system on problem-solution pairs.
    
    Args:
        brain: BrainMemory instance
        problems: List of HumanEval problems
        num_train: Number of problems to use for training
    """
    print(f"Training on {num_train} problems...")
    
    for i, problem in enumerate(problems[:num_train]):
        prompt = problem["prompt"]
        solution = problem["canonical_solution"]
        
        # Store as code transition
        brain.expose_pair(
            partial=prompt,
            complete=solution,
            domain="code",
            modality="code",
            source="humaneval_train",
        )
        
        if (i + 1) % 5 == 0:
            print(f"  Trained {i + 1}/{num_train} problems")
    
    print(f"Training complete. Memory stats:")
    print(f"  Transitions: {len(brain._transition_layer.transitions)}")
    if hasattr(brain._transition_layer, "get_token_transition_stats"):
        token_stats = brain._transition_layer.get_token_transition_stats()
        print(f"  Token contexts: {token_stats['total_contexts']}")
        print(f"  Token transitions: {token_stats['total_transitions']}")


def evaluate_problem(
    brain: BrainMemory,
    problem: Dict[str, Any],
    use_token_generation: bool = True,
) -> Tuple[bool, str, Optional[str]]:
    """
    Evaluate a single HumanEval problem.
    
    Args:
        brain: BrainMemory instance
        problem: Problem dict
        use_token_generation: Whether to use token-level generation
        
    Returns:
        (passed, status, error_message)
        - passed: True if tests passed
        - status: "pass", "syntax_error", "runtime_error", "assertion_error"
        - error_message: Error details if failed
    """
    prompt = problem["prompt"]
    test_code = problem["test"]
    entry_point = problem["entry_point"]
    
    # Generate solution
    try:
        if use_token_generation:
            # Use token-level generation
            generated = brain.generate_code(
                prompt=prompt,
                max_tokens=256,
                validate_syntax=True,
            )
        else:
            # Use field-based generation
            generated = brain.generate(
                prompt=prompt,
                max_new_tokens=256,
            )
    except Exception as e:
        return False, "generation_error", str(e)
    
    # Combine prompt + generated
    full_code = prompt + "\n" + generated
    
    # Check syntax
    try:
        ast.parse(full_code)
    except SyntaxError as e:
        return False, "syntax_error", str(e)
    
    # Execute test
    try:
        # Create execution namespace
        exec_globals = {}
        
        # Execute generated code
        exec(full_code, exec_globals)
        
        # Execute test code
        exec(test_code, exec_globals)
        
        return True, "pass", None
    
    except AssertionError as e:
        return False, "assertion_error", str(e)
    
    except Exception as e:
        return False, "runtime_error", f"{type(e).__name__}: {e}"


def run_humaneval_benchmark(
    problems_path: Optional[Path] = None,
    num_train: int = 10,
    num_eval: int = 20,
    use_token_generation: bool = True,
) -> Dict[str, Any]:
    """
    Run full HumanEval benchmark.
    
    Args:
        problems_path: Path to HumanEval JSONL file (uses samples if None)
        num_train: Number of problems for training
        num_eval: Number of problems for evaluation
        use_token_generation: Whether to use token-level generation
        
    Returns:
        Results dict with metrics
    """
    print("=" * 60)
    print("HumanEval Benchmark")
    print("=" * 60)
    
    # Load problems
    problems = load_humaneval_problems(problems_path)
    print(f"Loaded {len(problems)} problems")
    
    # Split train/eval
    train_problems = problems[:num_train]
    eval_problems = problems[num_train:num_train + num_eval]
    
    # Initialize brain
    brain = BrainMemory()
    
    # Train
    train_on_problems(brain, train_problems, num_train)
    
    # Evaluate
    print(f"\nEvaluating on {len(eval_problems)} problems...")
    
    results = {
        "total": len(eval_problems),
        "passed": 0,
        "syntax_errors": 0,
        "runtime_errors": 0,
        "assertion_errors": 0,
        "generation_errors": 0,
        "details": [],
    }
    
    for i, problem in enumerate(eval_problems):
        task_id = problem["task_id"]
        passed, status, error = evaluate_problem(brain, problem, use_token_generation)
        
        results["details"].append({
            "task_id": task_id,
            "passed": passed,
            "status": status,
            "error": error,
        })
        
        if passed:
            results["passed"] += 1
        elif status == "syntax_error":
            results["syntax_errors"] += 1
        elif status == "runtime_error":
            results["runtime_errors"] += 1
        elif status == "assertion_error":
            results["assertion_errors"] += 1
        else:
            results["generation_errors"] += 1
        
        print(f"  [{i+1}/{len(eval_problems)}] {task_id}: {status}")
    
    # Compute metrics
    results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Total: {results['total']}")
    print(f"  Passed: {results['passed']} ({results['pass_rate']:.1%})")
    print(f"  Syntax errors: {results['syntax_errors']}")
    print(f"  Runtime errors: {results['runtime_errors']}")
    print(f"  Assertion errors: {results['assertion_errors']}")
    print(f"  Generation errors: {results['generation_errors']}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Run benchmark with default settings
    run_humaneval_benchmark(
        num_train=10,
        num_eval=20,
        use_token_generation=True,
    )
