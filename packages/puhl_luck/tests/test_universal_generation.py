"""
Test universal surface sequence generation.
"""

import pytest
from puhl_luck.brain_memory import BrainMemory


def test_surface_sequence_storage():
    """Test that surface sequences are stored correctly."""
    brain = BrainMemory()
    
    # Train: input → target
    brain.expose_pair(
        partial="def add(a, b):",
        complete="return a + b",
        domain="code",
    )
    
    # Check storage
    stats = brain._surface_storage.get_stats()
    assert stats["total_sequences"] > 0, "Should store surface sequence"
    print(f"Storage stats: {stats}")


def test_exact_retrieval():
    """Test exact input match retrieval."""
    brain = BrainMemory()
    
    # Train
    input_text = "def add(a, b):"
    target_text = "return a + b"
    
    brain.expose_pair(input_text, target_text, domain="code")
    
    # Query with exact input
    result = brain.generate(input_text)
    
    print(f"Input: {input_text}")
    print(f"Generated: {result}")
    print(f"Expected: {target_text}")
    
    assert result == target_text, f"Should return exact target, got: {result}"


def test_multi_target_retrieval():
    """Test retrieving from multiple stored targets."""
    brain = BrainMemory()
    
    # Train multiple pairs
    pairs = [
        ("classify: 정치 뉴스", "정치"),
        ("classify: 경제 뉴스", "경제"),
        ("classify: IT 뉴스", "IT과학"),
    ]
    
    for inp, tgt in pairs:
        brain.expose_pair(inp, tgt, domain="classification")
    
    # Query
    result = brain.generate("classify: IT 뉴스")
    
    print(f"Result: {result}")
    assert result == "IT과학", f"Should retrieve IT과학, got: {result}"


def test_token_fallback():
    """Test token generation fallback when no stored sequence matches."""
    brain = BrainMemory()
    
    # Train similar but not exact
    brain.expose_pair("def add(a, b):", "return a + b", domain="code")
    brain.expose_pair("def sub(a, b):", "return a - b", domain="code")
    
    # Query with new input (should use token fallback)
    result = brain.generate("def multiply(x, y):")
    
    print(f"Fallback result: {result}")
    # Should generate something (not empty)
    assert isinstance(result, str), "Should return string"


def test_universal_domains():
    """Test that same storage works for all domains."""
    brain = BrainMemory()
    
    # Mix different domains
    brain.expose_pair("코드:", "def foo(): pass", domain="code")
    brain.expose_pair("분류:", "IT과학", domain="classification")
    brain.expose_pair("질문:", "정답입니다", domain="qa")
    
    # All should retrieve correctly
    assert brain.generate("코드:") == "def foo(): pass"
    assert brain.generate("분류:") == "IT과학"
    assert brain.generate("질문:") == "정답입니다"
    
    print("All domains work correctly!")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
