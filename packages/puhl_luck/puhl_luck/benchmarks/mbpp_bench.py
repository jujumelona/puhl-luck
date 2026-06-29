"""
MBPP (Mostly Basic Python Problems) benchmark for PUHL-LUCK.

MBPP: 974 crowd-sourced Python programming problems designed to be solvable
by entry-level programmers. Simpler than HumanEval.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from puhl_luck.brain_memory import BrainMemory


def load_mbpp_problems(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load MBPP problems.
    
    Returns sample problems if no path provided.
    """
    if path and path.exists():
        problems = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                problems = data
            elif isinstance(data, dict):
                problems = data.get("problems", [])
        return problems
    
    # Sample problems
    return [
        {
            "task_id": 1,
            "text": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].",
            "code": "def min_cost(cost, m, n):\n    R = 3\n    C = 3\n    tc = [[0 for x in range(C)] for x in range(R)]\n    tc[0][0] = cost[0][0]\n    for i in range(1, m+1):\n        tc[i][0] = tc[i-1][0] + cost[i][0]\n    for j in range(1, n+1):\n        tc[0][j] = tc[0][j-1] + cost[0][j]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j]\n    return tc[m][n]",
            "test_list": [
                "assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8",
                "assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12",
                "assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16",
            ],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\n    return tuple(set(test_tup1) & set(test_tup2))",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "def is_not_prime(n):\n    if n == 1:\n        return True\n    for i in range(2, n):\n        if n % i == 0:\n            return True\n    return False",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
                "assert is_not_prime(37) == False",
            ],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
        {
            "task_id": 4,
            "text": "Write a function to find the n largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\ndef heap_queue_largest(nums, n):\n    return hq.nlargest(n, nums)",
            "test_list": [
                "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]",
                "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]",
                "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]",
            ],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
    ]


def train_on_mbpp(brain: BrainMemory, problems: List[Dict[str, Any]], num_train: int = 20) -> None:
    """Train on MBPP problems."""
    print(f"Training on {num_train} MBPP problems...")
    
    for i, problem in enumerate(problems[:num_train]):
        # Create prompt from problem text
        prompt = f"# {problem['text']}\ndef "
        solution = problem["code"]
        
        # Store transition
        brain.expose_pair(
            partial=prompt,
            complete=solution,
            domain="code",
            modality="code",
            source="mbpp_train",
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Trained {i + 1}/{num_train} problems")
    
    print("Training complete.")


def evaluate_mbpp_problem(
    brain: BrainMemory,
    problem: Dict[str, Any],
    use_token_generation: bool = True,
) -> Tuple[bool, str, Optional[str]]:
    """Evaluate a single MBPP problem."""
    text = problem["text"]
    test_list = problem["test_list"]
    setup_code = problem.get("test_setup_code", "")
    
    # Create prompt
    prompt = f"# {text}\ndef "
    
    # Generate solution
    try:
        if use_token_generation:
            generated = brain.generate_code(
                prompt=prompt,
                max_tokens=256,
                validate_syntax=True,
            )
        else:
            generated = brain.generate(
                prompt=prompt,
                max_new_tokens=256,
            )
    except Exception as e:
        return False, "generation_error", str(e)
    
    # Syntax check
    try:
        ast.parse(generated)
    except SyntaxError as e:
        return False, "syntax_error", str(e)
    
    # Execute tests
    try:
        exec_globals = {}
        
        # Setup code
        if setup_code:
            exec(setup_code, exec_globals)
        
        # Generated code
        exec(generated, exec_globals)
        
        # Run tests
        for test in test_list:
            exec(test, exec_globals)
        
        return True, "pass", None
    
    except AssertionError as e:
        return False, "assertion_error", str(e)
    
    except Exception as e:
        return False, "runtime_error", f"{type(e).__name__}: {e}"


def run_mbpp_benchmark(
    problems_path: Optional[Path] = None,
    num_train: int = 20,
    num_eval: int = 30,
    use_token_generation: bool = True,
) -> Dict[str, Any]:
    """Run MBPP benchmark."""
    print("=" * 60)
    print("MBPP Benchmark")
    print("=" * 60)
    
    # Load problems
    problems = load_mbpp_problems(problems_path)
    print(f"Loaded {len(problems)} problems")
    
    # Split
    train_problems = problems[:num_train]
    eval_problems = problems[num_train:num_train + num_eval]
    
    # Initialize brain
    brain = BrainMemory()
    
    # Train
    train_on_mbpp(brain, train_problems, num_train)
    
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
        passed, status, error = evaluate_mbpp_problem(brain, problem, use_token_generation)
        
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
        
        print(f"  [{i+1}/{len(eval_problems)}] Task {task_id}: {status}")
    
    # Metrics
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
    run_mbpp_benchmark(
        num_train=20,
        num_eval=30,
        use_token_generation=True,
    )
