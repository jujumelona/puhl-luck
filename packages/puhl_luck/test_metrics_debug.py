#!/usr/bin/env python
"""Debug metrics tracking."""

from puhl_luck.brain_memory import BrainMemory

def main():
    brain = BrainMemory()
    
    # Train
    brain.expose_pair(
        "def add(a, b):",
        "def add(a, b): return a + b",
        domain="code"
    )
    
    print(f"Training outputs: {len(brain._metrics_tracker.training_outputs)}")
    print(f"Training operators: {len(brain._metrics_tracker.training_operators)}")
    
    # Generate with metrics
    result = brain.generate(
        "def add(x, y):",
        use_operator_generation=True,
        domain="code",
        return_metrics=True
    )
    
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    
    if isinstance(result, tuple) and len(result) == 2:
        output, metrics = result
        print(f"Output: {output}")
        print(f"Metrics: {metrics}")
        print(f"Generation method: {metrics.generation_method}")
    
    # Check summary
    summary = brain._metrics_tracker.get_summary()
    print(f"\nSummary: {summary}")

if __name__ == '__main__':
    main()
