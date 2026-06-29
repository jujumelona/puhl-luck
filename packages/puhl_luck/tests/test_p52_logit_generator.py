"""
Tests for P52 Sparse Logit Generator
"""

import pytest
from puhl_luck._logit_generator import SparseLogitGenerator


class TestSparseLogitGenerator:
    """Test sparse logit generator."""
    
    def test_initialization(self):
        """Should initialize generator."""
        gen = SparseLogitGenerator()
        
        assert gen.tables is not None
        assert gen.scorer is not None
        assert gen.pairs_learned == 0
    
    def test_learn_simple(self):
        """Should learn from (input, target) pair."""
        gen = SparseLogitGenerator()
        
        stats = gen.learn("hello", "world")
        
        assert stats["input_tokens"] > 0
        assert stats["target_tokens"] > 0
        assert stats["transitions_added"] > 0
        assert gen.pairs_learned == 1
    
    def test_learn_multiple_pairs(self):
        """Should learn from multiple pairs."""
        gen = SparseLogitGenerator()
        
        gen.learn("def add x y", "return x + y")
        gen.learn("def sub x y", "return x - y")
        gen.learn("def mul x y", "return x * y")
        
        assert gen.pairs_learned == 3
        assert len(gen.tables.vocab) > 5
    
    def test_generate_after_learning(self):
        """Should generate after learning."""
        gen = SparseLogitGenerator()
        
        # Learn pattern
        for _ in range(5):
            gen.learn("input", "output")
        
        # Generate
        output, metrics = gen.generate("input", max_tokens=10)
        
        assert isinstance(output, str)
        assert metrics["generation_method"] == "sparse_logit"
        assert metrics["output_tokens"] >= 0
    
    def test_no_repetition(self):
        """Should not repeat tokens infinitely."""
        gen = SparseLogitGenerator()
        
        # Learn pattern
        gen.learn("test", "result")
        
        # Generate
        output, metrics = gen.generate("test", max_tokens=20)
        
        # Check no single token repeats more than 3 times
        tokens = gen._tokenize(output)
        for token in set(tokens):
            count = tokens.count(token)
            assert count < 5, f"Token '{token}' repeated {count} times"
    
    def test_generates_diverse_outputs(self):
        """Should generate different outputs on multiple calls."""
        gen = SparseLogitGenerator()
        
        # Learn multiple patterns
        gen.learn("def func x", "return x")
        gen.learn("def func y", "return y + 1")
        gen.learn("def func z", "return z * 2")
        
        # Generate multiple times
        outputs = []
        for _ in range(5):
            output, _ = gen.generate("def func", max_tokens=10)
            outputs.append(output)
        
        # Should have at least 2 different outputs
        unique_outputs = len(set(outputs))
        assert unique_outputs >= 2, f"Only {unique_outputs} unique outputs"
    
    def test_respects_max_tokens(self):
        """Should respect max_tokens limit."""
        gen = SparseLogitGenerator()
        
        gen.learn("test", "a b c d e f g h i j k l m n o p")
        
        output, metrics = gen.generate("test", max_tokens=5)
        
        assert metrics["output_tokens"] <= 5
    
    def test_copy_boost_small(self):
        """Copy boost should be small, not dominate."""
        gen = SparseLogitGenerator(copy_boost_weight=0.1)
        
        # Learn pattern that doesn't include input token
        for _ in range(10):
            gen.learn("input", "different output here")
        
        # Generate - should NOT just copy "input"
        output, _ = gen.generate("input", max_tokens=10)
        
        # Output should not be just "input input input..."
        assert output != "input " * 5
    
    def test_statistics(self):
        """Should track statistics."""
        gen = SparseLogitGenerator()
        
        gen.learn("a", "b")
        gen.learn("c", "d")
        
        stats = gen.get_statistics()
        
        assert stats["pairs_learned"] == 2
        assert stats["tokens_learned"] > 0
        assert stats["vocab_size"] > 0
    
    def test_clear(self):
        """Should clear all data."""
        gen = SparseLogitGenerator()
        
        gen.learn("a", "b")
        gen.clear()
        
        assert gen.pairs_learned == 0
        assert gen.tokens_learned == 0
        assert len(gen.tables.vocab) == 0


class TestIntegrationWithBrain:
    """Test generator integrated with BrainMemory."""
    
    def test_brain_has_logit_generator(self):
        """Should have logit generator."""
        from puhl_luck.brain_memory import BrainMemory
        
        brain = BrainMemory()
        
        assert hasattr(brain, '_logit_generator')
        assert brain._logit_generator is not None
    
    def test_expose_pair_learns(self):
        """Should learn from expose_pair."""
        from puhl_luck.brain_memory import BrainMemory
        
        brain = BrainMemory()
        
        initial_pairs = brain._logit_generator.pairs_learned
        
        brain.expose_pair("def add x y", "return x + y", domain="code")
        
        assert brain._logit_generator.pairs_learned == initial_pairs + 1
    
    def test_generate_uses_logits(self):
        """Should generate using logit tables."""
        from puhl_luck.brain_memory import BrainMemory
        
        brain = BrainMemory()
        
        # Learn
        brain.expose_pair("test input", "test output", domain="code")
        
        # Generate
        output, metrics = brain.generate("test input", max_new_tokens=10, return_metrics=True)
        
        assert isinstance(output, str)
        assert metrics.generation_method == "sparse_logit"
    
    def test_no_python_repetition(self):
        """Should NOT produce 'Python Python Python...'"""
        from puhl_luck.brain_memory import BrainMemory
        
        brain = BrainMemory()
        
        # Learn pattern (even with "Python" in input)
        brain.expose_pair("Write Python code", "def hello(): print('hello')", domain="code")
        
        # Generate
        output = brain.generate("Write Python code", max_new_tokens=20)
        
        # Check no infinite repetition
        if "Python" in output:
            count = output.lower().count("python")
            assert count < 5, f"'Python' appears {count} times - repetition detected!"
