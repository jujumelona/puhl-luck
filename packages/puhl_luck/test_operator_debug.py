#!/usr/bin/env python
"""Debug operator-based generation."""

from puhl_luck.brain_memory import BrainMemory

def main():
    brain = BrainMemory()
    
    # Train
    partial = 'def count_even(nums):'
    complete = 'def count_even(nums): return len([x for x in nums if x % 2 == 0])'
    
    brain.expose_pair(partial, complete, domain='code')
    
    print(f"✓ Training complete")
    print(f"  Graphs: {len(brain._operator_storage.graphs)}")
    print(f"  Operators: {len(brain._operator_storage.operator_counts)}")
    print(f"  Operator counts: {dict(brain._operator_storage.operator_counts)}")
    
    # Check field_to_graph
    print(f"\n✓ Field mappings:")
    for field_sig, graph_sigs in brain._operator_storage.field_to_graph.items():
        print(f"  {field_sig[:80]}... → {len(graph_sigs)} graphs")
    
    # Generate with detailed output
    query = 'def count_even(arr):'
    print(f"\n✓ Generating for: {query}")
    
    # Get field features
    query_features = brain.features_for_query(query)
    print(f"  Query features (first 10): {query_features[:10]}")
    
    # Form field signature
    from puhl_luck._memory_operator_activation import OperatorActivation
    activation = OperatorActivation()
    field_sig = activation._form_field_signature(query_features[:20])
    print(f"  Field signature: {field_sig[:80]}...")
    
    # Activate operators
    activated = activation.activate(query_features, brain._operator_storage)
    print(f"\n✓ Activated operators: {len(activated)}")
    for i, act_op in enumerate(activated[:5]):
        print(f"  {i+1}. {act_op.operator.op_type} (score={act_op.activation_score:.3f})")
    
    # Generate
    result, metrics = brain.generate(
        query,
        use_operator_generation=True,
        domain='code',
        return_metrics=True
    )
    
    print(f"\n✓ Generated: {result}")
    print(f"  Method: {metrics.generation_method}")
    print(f"  Was copy: {metrics.was_exact_copy}")
    print(f"  Operators used: {metrics.operators_used}")

if __name__ == '__main__':
    main()
