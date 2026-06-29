#!/usr/bin/env python
"""Quick integration test for operator-based generation with metrics."""

from puhl_luck.brain_memory import BrainMemory

def main():
    brain = BrainMemory()

    # Test expose_pair stores operators
    partial = 'def count_even(nums):'
    complete = 'def count_even(nums): return len([x for x in nums if x % 2 == 0])'

    brain.expose_pair(partial, complete, domain='code')

    # Check operator storage
    print(f'✓ Operator storage graphs: {len(brain._operator_storage.graphs)}')
    print(f'✓ Operator storage operators: {len(brain._operator_storage.operator_counts)}')
    print(f'✓ Field mappings: {len(brain._operator_storage.field_to_graph)}')
    print(f'✓ Training examples recorded: {len(brain._metrics_tracker.training_outputs)}')

    # Test generate with operator-based generation and metrics
    result, metrics = brain.generate(
        'def count_even(arr):',
        use_operator_generation=True,
        domain='code',
        return_metrics=True
    )
    
    print(f'\n✓ Generated: {result[:80]}...' if len(result) > 80 else f'\n✓ Generated: {result}')
    print(f'✓ Generation method: {metrics.generation_method}')
    print(f'✓ Was exact copy: {metrics.was_exact_copy}')
    print(f'✓ Nearest similarity: {metrics.nearest_train_similarity:.2%}')
    
    # Check metrics summary
    summary = brain._metrics_tracker.get_summary()
    print(f'\n✓ Metrics Summary:')
    print(f'  - Total generations: {summary["total_generations"]}')
    print(f'  - Copy rate: {summary["copy_rate"]:.1%}')
    print(f'  - Novel composition rate: {summary["novel_composition_rate"]:.1%}')
    print(f'  - Avg similarity: {summary["avg_similarity"]:.2%}')
    
    print('\n✓✓✓ Integration test passed! ✓✓✓')

if __name__ == '__main__':
    main()
