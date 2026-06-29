"""
Parameter tuning script for sparse autoregressive generator.

Tests different hyperparameter values to find optimal configuration.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from puhl_luck.brain_memory import BrainMemory
from puhl_luck._memory_sparse_generator import SparseGeneratorConfig


def create_test_data() -> List[Tuple[str, str]]:
    """Create test training data."""
    return [
        ("def add a b", "return a + b"),
        ("def sub a b", "return a - b"),
        ("def mul a b", "return a * b"),
        ("def div a b", "return a / b"),
        ("def mod a b", "return a % b"),
        ("def max a b", "return a if a > b else b"),
        ("def min a b", "return a if a < b else b"),
        ("def abs x", "return x if x >= 0 else -x"),
        ("def square x", "return x * x"),
        ("def double x", "return x + x"),
    ]


def evaluate_config(
    config: SparseGeneratorConfig,
    train_data: List[Tuple[str, str]],
    test_queries: List[str]
) -> Dict[str, float]:
    """Evaluate a configuration."""
    brain = BrainMemory()
    
    # Override sparse generator config
    brain._sparse_generator.config = config
    brain._sparse_generator.context_sketch.K = config.context_k
    brain._sparse_generator.copy_gate.rare_token_threshold = config.rare_token_threshold
    brain._sparse_generator.autoregressive_loop.max_tokens = config.max_tokens
    brain._sparse_generator.autoregressive_loop.top_k = config.top_k
    
    # Train
    for input_text, target_text in train_data:
        brain.expose_pair(input_text, target_text, domain="code")
    
    # Evaluate
    total_tokens = 0
    empty_count = 0
    backoff_usage = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    copy_activations = 0
    
    for query in test_queries:
        output, metrics = brain.generate(
            query,
            max_new_tokens=20,
            domain="code",
            return_metrics=True
        )
        
        total_tokens += metrics.tokens_generated
        if metrics.empty_output:
            empty_count += 1
        
        for level, count in metrics.backoff_levels.items():
            backoff_usage[level] += count
        
        copy_activations += metrics.copy_gate_activations
    
    # Compute metrics
    avg_tokens = total_tokens / len(test_queries) if test_queries else 0
    empty_rate = empty_count / len(test_queries) if test_queries else 0
    
    # Backoff efficiency (lower levels are better)
    total_backoffs = sum(backoff_usage.values())
    backoff_score = sum(level * count for level, count in backoff_usage.items())
    avg_backoff_level = backoff_score / total_backoffs if total_backoffs > 0 else 0
    
    return {
        "avg_tokens": avg_tokens,
        "empty_rate": empty_rate,
        "avg_backoff_level": avg_backoff_level,
        "copy_activations": copy_activations,
        "score": avg_tokens * (1 - empty_rate) * (1 / (1 + avg_backoff_level))
    }


def tune_context_k(train_data: List[Tuple[str, str]], test_queries: List[str]):
    """Tune context window size K."""
    print("=" * 60)
    print("Tuning Context Window Size (K)")
    print("=" * 60)
    
    k_values = [5, 10, 15, 20]
    results = []
    
    for k in k_values:
        config = SparseGeneratorConfig(
            context_k=k,
            rare_token_threshold=3,
            max_tokens=512,
            top_k=5
        )
        
        metrics = evaluate_config(config, train_data, test_queries)
        results.append((k, metrics))
        
        print(f"\nK={k}:")
        print(f"  Avg tokens: {metrics['avg_tokens']:.2f}")
        print(f"  Empty rate: {metrics['empty_rate']:.1%}")
        print(f"  Avg backoff level: {metrics['avg_backoff_level']:.2f}")
        print(f"  Score: {metrics['score']:.3f}")
    
    # Find best
    best_k, best_metrics = max(results, key=lambda x: x[1]['score'])
    print(f"\n??Best K: {best_k} (score: {best_metrics['score']:.3f})")
    
    return best_k


def tune_rare_threshold(train_data: List[Tuple[str, str]], test_queries: List[str], best_k: int):
    """Tune rare token threshold."""
    print("\n" + "=" * 60)
    print("Tuning Rare Token Threshold")
    print("=" * 60)
    
    thresholds = [2, 3, 5]
    results = []
    
    for threshold in thresholds:
        config = SparseGeneratorConfig(
            context_k=best_k,
            rare_token_threshold=threshold,
            max_tokens=512,
            top_k=5
        )
        
        metrics = evaluate_config(config, train_data, test_queries)
        results.append((threshold, metrics))
        
        print(f"\nThreshold={threshold}:")
        print(f"  Avg tokens: {metrics['avg_tokens']:.2f}")
        print(f"  Copy activations: {metrics['copy_activations']}")
        print(f"  Score: {metrics['score']:.3f}")
    
    # Find best
    best_threshold, best_metrics = max(results, key=lambda x: x[1]['score'])
    print(f"\n??Best Threshold: {best_threshold} (score: {best_metrics['score']:.3f})")
    
    return best_threshold


def tune_top_k(train_data: List[Tuple[str, str]], test_queries: List[str], best_context_k: int, best_threshold: int):
    """Tune top-k candidates."""
    print("\n" + "=" * 60)
    print("Tuning Top-K Candidates")
    print("=" * 60)
    
    top_k_values = [3, 5, 10]
    results = []
    
    for top_k in top_k_values:
        config = SparseGeneratorConfig(
            context_k=best_context_k,
            rare_token_threshold=best_threshold,
            max_tokens=512,
            top_k=top_k
        )
        
        metrics = evaluate_config(config, train_data, test_queries)
        results.append((top_k, metrics))
        
        print(f"\nTop-K={top_k}:")
        print(f"  Avg tokens: {metrics['avg_tokens']:.2f}")
        print(f"  Empty rate: {metrics['empty_rate']:.1%}")
        print(f"  Score: {metrics['score']:.3f}")
    
    # Find best
    best_top_k, best_metrics = max(results, key=lambda x: x[1]['score'])
    print(f"\n??Best Top-K: {best_top_k} (score: {best_metrics['score']:.3f})")
    
    return best_top_k


def run_parameter_tuning():
    """Run full parameter tuning."""
    print("=" * 60)
    print("SPARSE AUTOREGRESSIVE GENERATOR")
    print("Parameter Tuning")
    print("=" * 60)
    
    # Prepare data
    train_data = create_test_data()
    test_queries = [
        "def add x y",
        "def sub x y",
        "def mul x y",
    ]
    
    # Tune parameters
    best_k = tune_context_k(train_data, test_queries)
    best_threshold = tune_rare_threshold(train_data, test_queries, best_k)
    best_top_k = tune_top_k(train_data, test_queries, best_k, best_threshold)
    
    # Final results
    print("\n" + "=" * 60)
    print("OPTIMAL PARAMETERS")
    print("=" * 60)
    print(f"Context Window (K): {best_k}")
    print(f"Rare Token Threshold: {best_threshold}")
    print(f"Top-K Candidates: {best_top_k}")
    print(f"Max Tokens: 512 (default)")
    print("=" * 60)
    
    # Test with optimal config
    print("\nTesting with optimal configuration...")
    config = SparseGeneratorConfig(
        context_k=best_k,
        rare_token_threshold=best_threshold,
        max_tokens=512,
        top_k=best_top_k
    )
    
    final_metrics = evaluate_config(config, train_data, test_queries)
    print(f"Final Score: {final_metrics['score']:.3f}")
    print(f"Avg Tokens: {final_metrics['avg_tokens']:.2f}")
    print(f"Empty Rate: {final_metrics['empty_rate']:.1%}")
    print(f"Avg Backoff Level: {final_metrics['avg_backoff_level']:.2f}")
    
    return {
        "context_k": best_k,
        "rare_token_threshold": best_threshold,
        "top_k": best_top_k,
        "max_tokens": 512
    }


if __name__ == "__main__":
    optimal = run_parameter_tuning()
    
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 60)
    print("```python")
    print("from puhl_luck._memory_sparse_generator import SparseGeneratorConfig")
    print("")
    print("config = SparseGeneratorConfig(")
    print(f"    context_k={optimal['context_k']},")
    print(f"    rare_token_threshold={optimal['rare_token_threshold']},")
    print(f"    max_tokens={optimal['max_tokens']},")
    print(f"    top_k={optimal['top_k']}")
    print(")")
    print("```")
    print("=" * 60)

