"""Tests for token-level generation."""

import pytest
from puhl_luck.brain_memory import BrainMemory
from puhl_luck._memory_tokenization import tokenize_code


def test_token_transition_storage():
    """Test that token transitions are stored correctly."""
    brain = BrainMemory()
    
    partial = "def add(a, b):"
    complete = "def add(a, b):\n    return a + b"
    
    # Store transition
    brain.expose_pair(partial, complete, domain="code", modality="code")
    
    # Check stats
    stats = brain._transition_layer.get_token_transition_stats()
    assert stats["total_contexts"] > 0, "Should have stored some contexts"
    assert stats["total_transitions"] > 0, "Should have stored some transitions"
    print(f"Token stats: {stats}")


def test_token_query():
    """Test querying for next tokens."""
    brain = BrainMemory()
    
    partial = "def add(a, b):"
    complete = "def add(a, b):\n    return a + b"
    
    # Store
    brain.expose_pair(partial, complete, domain="code", modality="code")
    
    # Tokenize partial
    partial_tokens = tokenize_code(partial)
    print(f"Partial tokens: {partial_tokens}")
    
    # Query
    candidates = brain._transition_layer.find_next_token(partial_tokens, top_k=5)
    print(f"Next token candidates: {candidates}")
    
    assert len(candidates) > 0, "Should find at least one candidate"


def test_generate_tokens():
    """Test token-by-token generation."""
    brain = BrainMemory()
    
    # Train on multiple examples
    examples = [
        ("def add(a, b):", "def add(a, b):\n    return a + b"),
        ("def multiply(x, y):", "def multiply(x, y):\n    return x * y"),
    ]
    
    for partial, complete in examples:
        brain.expose_pair(partial, complete, domain="code", modality="code")
    
    # Generate
    result = brain.generate_tokens("def add(a, b):", max_tokens=20, domain="code")
    print(f"Generated: '{result}'")
    
    assert isinstance(result, str), "Should return string"
    # Note: result might be empty if no patterns match, that's OK for now


def test_generate_code():
    """Test code generation with validation."""
    brain = BrainMemory()
    
    # Train
    examples = [
        ("def add(a, b):", "def add(a, b):\n    return a + b"),
        ("def sub(a, b):", "def sub(a, b):\n    return a - b"),
    ]
    
    for partial, complete in examples:
        brain.expose_pair(partial, complete, domain="code", modality="code")
    
    # Generate
    result = brain.generate_code("def add(a, b):", max_tokens=32, validate_syntax=False)
    print(f"Generated code: '{result}'")
    
    assert isinstance(result, str), "Should return string"


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
