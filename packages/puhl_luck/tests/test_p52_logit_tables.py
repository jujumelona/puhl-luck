"""
Tests for P52 Sparse Logit Tables
"""

import pytest
from puhl_luck._logit_tables import SparseLogitTables


class TestSparseLogitTables:
    """Test sparse logit tables."""
    
    def test_initialization(self):
        """Should initialize empty tables."""
        tables = SparseLogitTables()
        
        assert len(tables.vocab) == 0
        assert len(tables.unigram) == 0
        assert len(tables.transition) == 0
        assert tables.unigram_total == 0
        assert tables.updates == 0
    
    def test_update_unigram(self):
        """Should update unigram counts."""
        tables = SparseLogitTables()
        
        tables.update([], "hello")
        tables.update([], "hello")
        tables.update([], "world")
        
        assert tables.unigram["hello"] == 2
        assert tables.unigram["world"] == 1
        assert tables.unigram_total == 3
        assert len(tables.vocab) == 2
    
    def test_update_transition(self):
        """Should update transition counts."""
        tables = SparseLogitTables()
        
        tables.update(["hello"], "world")
        tables.update(["hello"], "world")
        tables.update(["hello"], "there")
        
        assert tables.transition[("hello", "world")] == 2
        assert tables.transition[("hello", "there")] == 1
        assert tables.transition_marginal["hello"] == 3
    
    def test_update_bigram(self):
        """Should update bigram counts."""
        tables = SparseLogitTables()
        
        tables.update(["hello", "my"], "friend")
        tables.update(["hello", "my"], "friend")
        
        assert tables.bigram[("hello", "my", "friend")] == 2
        assert tables.bigram_marginal[("hello", "my")] == 2
    
    def test_update_trigram(self):
        """Should update trigram counts."""
        tables = SparseLogitTables()
        
        tables.update(["hello", "my", "dear"], "friend")
        
        assert tables.trigram[("hello", "my", "dear", "friend")] == 1
        assert tables.trigram_marginal[("hello", "my", "dear")] == 1
    
    def test_update_field_token(self):
        """Should update field-token counts."""
        tables = SparseLogitTables()
        
        tables.update([], "hello", field_features=["feature1", "feature2"])
        
        assert tables.field_token[("feature1", "hello")] == 1
        assert tables.field_token[("feature2", "hello")] == 1
        assert tables.field_marginal["feature1"] == 1
    
    def test_update_position_token(self):
        """Should update position-token counts."""
        tables = SparseLogitTables()
        
        tables.update([], "hello", position_label="start")
        tables.update([], "world", position_label="end")
        
        assert tables.position_token[("start", "hello")] == 1
        assert tables.position_token[("end", "world")] == 1
    
    def test_get_log_prob_unigram(self):
        """Should compute unigram log probability."""
        tables = SparseLogitTables()
        
        tables.update([], "hello")
        tables.update([], "hello")
        tables.update([], "world")
        
        # hello: 2/3, world: 1/3
        score_hello = tables.get_log_prob("hello", [])
        score_world = tables.get_log_prob("world", [])
        
        assert score_hello > score_world  # hello is more frequent
    
    def test_get_log_prob_additive(self):
        """Should sum scores from multiple tables."""
        tables = SparseLogitTables()
        
        # Setup: "hello" → "world" transition
        tables.update([], "hello")
        tables.update(["hello"], "world")
        tables.update(["hello"], "world")
        
        # Get score for "world" with context "hello"
        score_with_context = tables.get_log_prob("world", ["hello"])
        
        # Get score for "world" without context
        score_no_context = tables.get_log_prob("world", [])
        
        # With context should be higher (unigram + transition)
        assert score_with_context > score_no_context
    
    def test_get_top_candidates(self):
        """Should return top candidates."""
        tables = SparseLogitTables()
        
        # Add some data
        tables.update([], "a")
        tables.update([], "a")
        tables.update([], "a")
        tables.update([], "b")
        tables.update([], "c")
        
        candidates = tables.get_top_candidates([], top_k=3)
        
        assert len(candidates) <= 3
        assert candidates[0][0] == "a"  # Most frequent
        assert all(isinstance(score, float) for _, score in candidates)
    
    def test_statistics(self):
        """Should return statistics."""
        tables = SparseLogitTables()
        
        tables.update([], "hello")
        tables.update(["hello"], "world")
        
        stats = tables.get_statistics()
        
        assert stats["vocab_size"] == 2
        assert stats["unigram_entries"] == 2
        assert stats["transition_entries"] == 1
        assert stats["total_updates"] == 2
    
    def test_clear(self):
        """Should clear all tables."""
        tables = SparseLogitTables()
        
        tables.update([], "hello")
        tables.update(["hello"], "world")
        
        tables.clear()
        
        assert len(tables.vocab) == 0
        assert len(tables.unigram) == 0
        assert tables.unigram_total == 0
        assert tables.updates == 0
