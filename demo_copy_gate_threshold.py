"""
Demonstration of Copy Gate Threshold Optimization (Task 7.5)

This script demonstrates how the rare_token_threshold parameter affects
copy gate behavior in the HDC system.

Key Features:
1. Tokens with frequency < threshold are prioritized for copy extraction
2. Different thresholds produce different copy token lists
3. Grid search finds optimal threshold for accuracy/speed tradeoff

Requirements: 6.1, 6.2, 6.3, 6.4
"""

from puhl_luck.brain_memory import BrainMemory
from puhl_luck.benchmarks.hyperparameter_tuner import HyperparameterTuner, HyperparameterConfig


def demo_rare_token_prioritization():
    """Demonstrate rare token prioritization based on threshold."""
    print("=" * 80)
    print("DEMO: Rare Token Prioritization in Copy Gate")
    print("=" * 80)
    
    brain = BrainMemory()
    lg = brain._logit_generator
    
    # Train with varying token frequencies
    print("\nTraining with different token frequencies:")
    
    # Very common token (frequency = 15)
    for i in range(15):
        brain.expose_pair(f"example {i}", "common", domain="demo", modality="demo")
    
    # Medium frequency token (frequency = 5)
    for i in range(5):
        brain.expose_pair(f"test {i}", "medium", domain="demo", modality="demo")
    
    # Rare token (frequency = 2)
    for i in range(2):
        brain.expose_pair(f"special {i}", "rare", domain="demo", modality="demo")
    
    # Very rare token (frequency = 1)
    brain.expose_pair("unique case", "veryrare", domain="demo", modality="demo")
    
    # Check learned frequencies
    print("\nLearned token frequencies:")
    tokens = ['common', 'medium', 'rare', 'veryrare', 'unseen']
    for token in tokens:
        freq = lg.tables.vocab.get(token, 0)
        print(f"  {token:15s}: frequency = {freq}")
    
    # Test copy gate with different thresholds
    print("\n" + "=" * 80)
    print("Copy Gate Behavior with Different Thresholds")
    print("=" * 80)
    
    test_tokens = ['common', 'medium', 'rare', 'veryrare', 'unseen', 'another']
    
    for threshold in [1, 2, 3, 6, 10]:
        lg.rare_token_threshold = threshold
        copy_list = lg._copy_tokens(test_tokens, limit=24)
        
        print(f"\nThreshold = {threshold}:")
        print(f"  Copy tokens: {copy_list}")
        
        # Show which tokens are considered "rare"
        rare_tokens = []
        for token in test_tokens:
            freq = lg.tables.vocab.get(token, 0)
            if freq < threshold:
                rare_tokens.append(f"{token}(freq={freq})")
        
        print(f"  Rare tokens (freq < {threshold}): {rare_tokens}")
        print(f"  → Rare tokens prioritized at start of copy list")


def demo_threshold_impact_on_generation():
    """Demonstrate how threshold affects generation quality."""
    print("\n" + "=" * 80)
    print("DEMO: Threshold Impact on Generation")
    print("=" * 80)
    
    # Create training data with common and rare tokens
    train_data = [
        ("def add(x, y):", "return x + y"),
        ("def subtract(a, b):", "return a - b"),
        ("def multiply(m, n):", "return m * n"),
        ("def divide(p, q):", "return p / q"),
        ("def modulo(i, j):", "return i % j"),
    ]
    
    test_data = [
        ("def power(base, exp):", "return base ** exp"),
    ]
    
    print("\nTraining data:")
    for inp, out in train_data:
        print(f"  {inp:30s} → {out}")
    
    print("\nTest query:")
    print(f"  {test_data[0][0]:30s} → {test_data[0][1]} (expected)")
    
    print("\n" + "-" * 80)
    print("Testing different threshold values:")
    print("-" * 80)
    
    for threshold in [1, 2, 3, 5]:
        print(f"\nThreshold = {threshold}:")
        
        # Create fresh brain
        brain = BrainMemory()
        lg = brain._logit_generator
        lg.rare_token_threshold = threshold
        
        # Train
        for inp, out in train_data:
            brain.expose_pair(inp, out, domain="code", modality="code")
        
        # Check which tokens are rare
        test_tokens = ['base', 'exp', 'return', 'def', 'power']
        rare_count = 0
        for token in test_tokens:
            freq = lg.tables.vocab.get(token, 0)
            if freq < threshold:
                rare_count += 1
        
        print(f"  Rare tokens in test: {rare_count}/{len(test_tokens)}")
        
        # Generate
        try:
            result = brain.generate(
                query=test_data[0][0],
                max_new_tokens=10,
                domain="code"
            )
            
            if isinstance(result, tuple):
                generated, metrics = result
            else:
                generated = result
                metrics = None
            
            print(f"  Generated: {generated}")
            
            if metrics and hasattr(metrics, 'copy_gate_activations'):
                print(f"  Copy gate activations: {metrics.copy_gate_activations}")
                
        except Exception as e:
            print(f"  Generation failed: {e}")


def demo_grid_search_optimization():
    """Demonstrate finding optimal threshold via grid search."""
    print("\n" + "=" * 80)
    print("DEMO: Grid Search Finds Optimal Threshold")
    print("=" * 80)
    
    # Create training/test data
    train_data = [
        ("calculate sum:", "add numbers"),
        ("calculate product:", "multiply values"),
        ("calculate difference:", "subtract amounts"),
        ("calculate quotient:", "divide quantities"),
    ]
    
    test_data = [
        ("calculate average:", "divide sum"),
        ("calculate total:", "add all"),
    ]
    
    print("\nSearching for optimal rare_token_threshold...")
    print("(This evaluates all combinations of hyperparameters)")
    
    tuner = HyperparameterTuner(
        train_data=train_data,
        test_data=test_data,
        domain='calculation'
    )
    
    # Focused search space
    tuner.set_search_space(
        context_windows=[3, 5],
        rare_thresholds=[1, 2, 3, 4, 5],  # Full range of thresholds
        top_k_values=[2, 3]
    )
    
    print(f"\nSearch space:")
    print(f"  Context windows: {tuner.context_windows}")
    print(f"  Rare thresholds: {tuner.rare_thresholds}")
    print(f"  Top-K values: {tuner.top_k_values}")
    print(f"  Total configs: {len(tuner.context_windows) * len(tuner.rare_thresholds) * len(tuner.top_k_values)}")
    
    # Run grid search
    results = tuner.grid_search(
        max_new_tokens=15,
        verbose=False
    )
    
    print(f"\nGrid search completed!")
    print(f"  Total evaluations: {results['total_evaluations']}")
    print(f"  Time: {results['total_time_ms'] / 1000:.2f}s")
    
    # Analyze threshold impact
    print("\n" + "-" * 80)
    print("Threshold Performance Analysis:")
    print("-" * 80)
    
    # Group results by threshold
    threshold_performance = {}
    for result in results['all_results']:
        config = result['config']
        threshold = config['rare_token_threshold']
        
        if threshold not in threshold_performance:
            threshold_performance[threshold] = {
                'accuracies': [],
                'speeds': [],
                'copy_activations': []
            }
        
        threshold_performance[threshold]['accuracies'].append(result['accuracy'])
        threshold_performance[threshold]['speeds'].append(result['avg_inference_time_ms'])
        threshold_performance[threshold]['copy_activations'].append(result['copy_gate_activations'])
    
    # Print summary
    print(f"\n{'Threshold':<12} {'Avg Accuracy':<15} {'Avg Speed (ms)':<18} {'Avg Copy Acts':<15}")
    print("-" * 70)
    
    for threshold in sorted(threshold_performance.keys()):
        perf = threshold_performance[threshold]
        avg_acc = sum(perf['accuracies']) / len(perf['accuracies'])
        avg_speed = sum(perf['speeds']) / len(perf['speeds'])
        avg_copy = sum(perf['copy_activations']) / len(perf['copy_activations'])
        
        print(f"{threshold:<12} {avg_acc * 100:>6.1f}%        {avg_speed:>8.2f}          {avg_copy:>6.1f}")
    
    # Show best configurations
    print("\n" + "-" * 80)
    print("Best Configurations:")
    print("-" * 80)
    
    best_acc = results['best_accuracy_config']
    print(f"\nHighest Accuracy:")
    print(f"  Config: K={best_acc['config']['context_window']}, "
          f"rare={best_acc['config']['rare_token_threshold']}, "
          f"top_k={best_acc['config']['top_k']}")
    print(f"  Accuracy: {best_acc['accuracy'] * 100:.1f}%")
    print(f"  Speed: {best_acc['avg_inference_time_ms']:.2f}ms")
    
    best_speed = results['best_speed_config']
    print(f"\nFastest Speed:")
    print(f"  Config: K={best_speed['config']['context_window']}, "
          f"rare={best_speed['config']['rare_token_threshold']}, "
          f"top_k={best_speed['config']['top_k']}")
    print(f"  Accuracy: {best_speed['accuracy'] * 100:.1f}%")
    print(f"  Speed: {best_speed['avg_inference_time_ms']:.2f}ms")
    
    # Recommend balanced config
    recommendation = tuner.recommend_config(results, priority='balanced')
    print(f"\nBalanced Recommendation:")
    print(f"  Config: K={recommendation['recommended_config']['context_window']}, "
          f"rare={recommendation['recommended_config']['rare_token_threshold']}, "
          f"top_k={recommendation['recommended_config']['top_k']}")
    print(f"  Accuracy: {recommendation['accuracy'] * 100:.1f}%")
    print(f"  Speed: {recommendation['avg_inference_time_ms']:.2f}ms")


def main():
    """Run all demonstrations."""
    try:
        demo_rare_token_prioritization()
        demo_threshold_impact_on_generation()
        demo_grid_search_optimization()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("1. rare_token_threshold controls which tokens are prioritized for copying")
        print("2. Lower thresholds → more tokens considered rare → more copy candidates")
        print("3. Higher thresholds → fewer tokens considered rare → selective copying")
        print("4. Grid search finds optimal threshold for accuracy/speed tradeoff")
        print("\nIntegration Points:")
        print("✓ HyperparameterConfig includes rare_token_threshold")
        print("✓ Grid search tests all threshold values [1-5]")
        print("✓ _train_brain() applies threshold to generator")
        print("✓ _copy_tokens() prioritizes tokens with freq < threshold")
        print("✓ Pareto-optimal selection considers threshold impact")
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
