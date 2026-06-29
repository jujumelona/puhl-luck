"""
Run all code generation benchmarks and compare results.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from humaneval_bench import run_humaneval_benchmark
from mbpp_bench import run_mbpp_benchmark
from code_completion_bench import run_code_completion_benchmark


def run_all_benchmarks(use_token_generation: bool = True):
    """
    Run all benchmarks and report combined results.
    
    Args:
        use_token_generation: Whether to use token-level generation
    """
    print("\n" + "=" * 60)
    print("PUHL-LUCK Code Generation Benchmark Suite")
    print("=" * 60)
    print(f"Generation mode: {'Token-level' if use_token_generation else 'Field-based'}")
    print()
    
    all_results = {}
    
    # 1. Code Completion Benchmark (fastest, most basic)
    print("\n[1/3] Running Code Completion Benchmark...")
    start = time.time()
    try:
        cc_results = run_code_completion_benchmark(use_token_generation)
        all_results["code_completion"] = cc_results
    except Exception as e:
        print(f"ERROR: {e}")
        all_results["code_completion"] = {"error": str(e)}
    print(f"Time: {time.time() - start:.1f}s")
    
    # 2. MBPP Benchmark (medium difficulty)
    print("\n[2/3] Running MBPP Benchmark...")
    start = time.time()
    try:
        mbpp_results = run_mbpp_benchmark(
            num_train=20,
            num_eval=30,
            use_token_generation=use_token_generation,
        )
        all_results["mbpp"] = mbpp_results
    except Exception as e:
        print(f"ERROR: {e}")
        all_results["mbpp"] = {"error": str(e)}
    print(f"Time: {time.time() - start:.1f}s")
    
    # 3. HumanEval Benchmark (hardest)
    print("\n[3/3] Running HumanEval Benchmark...")
    start = time.time()
    try:
        humaneval_results = run_humaneval_benchmark(
            num_train=10,
            num_eval=20,
            use_token_generation=use_token_generation,
        )
        all_results["humaneval"] = humaneval_results
    except Exception as e:
        print(f"ERROR: {e}")
        all_results["humaneval"] = {"error": str(e)}
    print(f"Time: {time.time() - start:.1f}s")
    
    # Combined summary
    print("\n" + "=" * 60)
    print("COMBINED RESULTS")
    print("=" * 60)
    
    for benchmark_name, results in all_results.items():
        if "error" in results:
            print(f"\n{benchmark_name.upper()}: ERROR")
            print(f"  {results['error']}")
            continue
        
        print(f"\n{benchmark_name.upper()}:")
        print(f"  Pass rate: {results.get('pass_rate', 0):.1%}")
        print(f"  Passed: {results.get('passed', 0)}/{results.get('total', 0)}")
        
        if "syntax_errors" in results:
            print(f"  Syntax errors: {results['syntax_errors']}")
        if "assertion_errors" in results:
            print(f"  Assertion errors: {results['assertion_errors']}")
        if "runtime_errors" in results:
            print(f"  Runtime errors: {results['runtime_errors']}")
    
    # Overall metrics
    total_problems = sum(r.get("total", 0) for r in all_results.values() if "error" not in r)
    total_passed = sum(r.get("passed", 0) for r in all_results.values() if "error" not in r)
    
    print(f"\nOVERALL:")
    print(f"  Total problems: {total_problems}")
    print(f"  Total passed: {total_passed}")
    print(f"  Overall pass rate: {total_passed / total_problems:.1%}" if total_problems > 0 else "  N/A")
    
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run PUHL-LUCK code benchmarks")
    parser.add_argument(
        "--mode",
        choices=["token", "field"],
        default="token",
        help="Generation mode: token-level or field-based",
    )
    
    args = parser.parse_args()
    
    run_all_benchmarks(use_token_generation=(args.mode == "token"))
