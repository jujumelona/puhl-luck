"""
P52 Sample Benchmark - Quick validation on 10 problems.
"""

from puhl_luck.brain_memory import BrainMemory
import ast

def test_problems():
    """Return 10 simple test problems."""
    return [
        {
            "id": 1,
            "prompt": "# Write a function to add two numbers\ndef add(a, b):",
            "expected": "return a + b",
            "test": "assert add(2, 3) == 5",
        },
        {
            "id": 2,
            "prompt": "# Write a function to subtract two numbers\ndef subtract(a, b):",
            "expected": "return a - b",
            "test": "assert subtract(5, 3) == 2",
        },
        {
            "id": 3,
            "prompt": "# Write a function to multiply two numbers\ndef multiply(a, b):",
            "expected": "return a * b",
            "test": "assert multiply(4, 5) == 20",
        },
        {
            "id": 4,
            "prompt": "# Write a function to check if number is even\ndef is_even(n):",
            "expected": "return n % 2 == 0",
            "test": "assert is_even(4) == True and is_even(3) == False",
        },
        {
            "id": 5,
            "prompt": "# Write a function to get the first element\ndef first(lst):",
            "expected": "return lst[0]",
            "test": "assert first([1, 2, 3]) == 1",
        },
        {
            "id": 6,
            "prompt": "# Write a function to get the last element\ndef last(lst):",
            "expected": "return lst[-1]",
            "test": "assert last([1, 2, 3]) == 3",
        },
        {
            "id": 7,
            "prompt": "# Write a function to get length of list\ndef length(lst):",
            "expected": "return len(lst)",
            "test": "assert length([1, 2, 3, 4]) == 4",
        },
        {
            "id": 8,
            "prompt": "# Write a function to check if list is empty\ndef is_empty(lst):",
            "expected": "return len(lst) == 0",
            "test": "assert is_empty([]) == True and is_empty([1]) == False",
        },
        {
            "id": 9,
            "prompt": "# Write a function to reverse a string\ndef reverse(s):",
            "expected": "return s[::-1]",
            "test": "assert reverse('hello') == 'olleh'",
        },
        {
            "id": 10,
            "prompt": "# Write a function to uppercase a string\ndef uppercase(s):",
            "expected": "return s.upper()",
            "test": "assert uppercase('hello') == 'HELLO'",
        },
    ]


def run_benchmark():
    """Run P52 benchmark on 10 problems."""
    print("=" * 70)
    print("P52 SAMPLE BENCHMARK - 10 Problems")
    print("=" * 70)
    
    brain = BrainMemory()
    problems = test_problems()
    
    # Phase 1: Training (first 5 problems)
    print("\n[PHASE 1] Training on 5 problems...")
    train_problems = problems[:5]
    for p in train_problems:
        brain.expose_pair(p["prompt"], p["expected"], domain="code")
        print(f"  ✓ Learned problem {p['id']}")
    
    print(f"\nTraining complete:")
    stats = brain._logit_generator.get_statistics()
    print(f"  Pairs learned: {stats['pairs_learned']}")
    print(f"  Vocab size: {stats['vocab_size']}")
    print(f"  Transitions: {stats['transition_entries']}")
    
    # Phase 2: Evaluation (last 5 problems)
    print("\n[PHASE 2] Evaluating on 5 problems...")
    eval_problems = problems[5:]
    
    results = {
        "total": len(eval_problems),
        "passed": 0,
        "syntax_ok": 0,
        "runtime_ok": 0,
        "details": []
    }
    
    for p in eval_problems:
        print(f"\n--- Problem {p['id']} ---")
        print(f"Prompt: {p['prompt'][:50]}...")
        
        # Generate
        try:
            output = brain.generate(p["prompt"], max_new_tokens=30)
            print(f"Generated: {output[:60]}...")
        except Exception as e:
            print(f"❌ Generation error: {e}")
            results["details"].append({
                "id": p["id"],
                "status": "generation_error",
                "error": str(e)
            })
            continue
        
        # Check syntax
        full_code = p["prompt"] + "\n    " + output
        try:
            ast.parse(full_code)
            results["syntax_ok"] += 1
            print("✓ Syntax valid")
        except SyntaxError as e:
            print(f"❌ Syntax error: {e}")
            results["details"].append({
                "id": p["id"],
                "status": "syntax_error",
                "error": str(e)
            })
            continue
        
        # Run test
        try:
            exec_globals = {}
            exec(full_code, exec_globals)
            exec(p["test"], exec_globals)
            
            results["runtime_ok"] += 1
            results["passed"] += 1
            print("✅ Test PASSED")
            
            results["details"].append({
                "id": p["id"],
                "status": "passed",
                "output": output
            })
            
        except AssertionError as e:
            print(f"❌ Test failed: Wrong answer")
            results["details"].append({
                "id": p["id"],
                "status": "assertion_error",
                "error": "Wrong answer"
            })
            
        except Exception as e:
            print(f"❌ Runtime error: {e}")
            results["details"].append({
                "id": p["id"],
                "status": "runtime_error",
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total evaluated: {results['total']}")
    print(f"Tests passed:    {results['passed']} ({results['passed']/results['total']*100:.1f}%)")
    print(f"Syntax valid:    {results['syntax_ok']} ({results['syntax_ok']/results['total']*100:.1f}%)")
    print(f"Runtime ok:      {results['runtime_ok']} ({results['runtime_ok']/results['total']*100:.1f}%)")
    print("=" * 70)
    
    # Detailed results
    print("\nDetailed Results:")
    for detail in results["details"]:
        status_symbol = "✅" if detail["status"] == "passed" else "❌"
        print(f"  {status_symbol} Problem {detail['id']}: {detail['status']}")
        if detail["status"] != "passed":
            print(f"     Error: {detail.get('error', 'N/A')}")
    
    print("\n" + "=" * 70)
    
    if results["passed"] >= 2:
        print("✅ P52 IS WORKING! Pass rate >= 40%")
    else:
        print("⚠️  P52 needs tuning. Pass rate < 40%")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_benchmark()
