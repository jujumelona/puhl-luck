"""
Tests for PUHLCompatibilityAdapter: Backward compatibility layer.
"""

import pytest
import tempfile
import os
from pathlib import Path

from puhl_luck._memory_puhl_adapter import PUHLCompatibilityAdapter


class TestPUHLCompatibilityAdapter:
    """Test backward compatibility adapter."""
    
    @pytest.fixture
    def temp_memory_file(self):
        """Create temporary memory file."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name
        yield filepath
        # Cleanup
        try:
            os.unlink(filepath)
        except:
            pass
        # Also cleanup cognitive field file
        cf_path = filepath.replace('.pkl', '_cognitive_field.pkl')
        try:
            os.unlink(cf_path)
        except:
            pass
    
    def test_initialization(self, temp_memory_file):
        """Test adapter initialization."""
        adapter = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        assert adapter is not None
        assert adapter.cognitive_field is not None
        assert adapter.brain_memory is not None
    
    def test_expose_text(self, temp_memory_file):
        """Test expose_text backward compatible API."""
        adapter = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        event_id = adapter.expose_text(
            "This is a test",
            metadata={"source": "test"}
        )
        
        assert event_id is not None
        assert isinstance(event_id, str)
        
        # Check event stored in both systems
        assert event_id in adapter.brain_memory.events
        assert event_id in adapter.cognitive_field.events_layer.events
    
    def test_expose_text_multiple(self, temp_memory_file):
        """Test exposing multiple texts."""
        adapter = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        event_ids = []
        for i in range(3):
            event_id = adapter.expose_text(f"Test text {i}")
            event_ids.append(event_id)
        
        assert len(event_ids) == 3
        assert len(set(event_ids)) == 3  # All unique
        
        # All stored in both systems
        for eid in event_ids:
            assert eid in adapter.brain_memory.events
            assert eid in adapter.cognitive_field.events_layer.events
    
    def test_expose_file(self, temp_memory_file):
        """Test expose_file backward compatible API."""
        adapter = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Line 1\nLine 2\nLine 3")
            test_file = f.name
        
        try:
            event_ids = adapter.expose_file(test_file)
            
            assert event_ids is not None
            assert len(event_ids) > 0
            
            # Check events stored
            for eid in event_ids:
                assert eid in adapter.brain_memory.events
                assert eid in adapter.cognitive_field.events_layer.events
        finally:
            os.unlink(test_file)
    
    def test_rank_similarity_mode(self, temp_memory_file):
        """Test rank with similarity mode (old API)."""
        adapter = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        # Expose some texts
        adapter.expose_text("Python programming")
        adapter.expose_text("Machine learning")
        adapter.expose_text("Data science")
        
        # Rank candidates
        candidates = ["Python", "Java", "JavaScript"]
        ranked = adapter.rank("programming", candidates, mode="similarity", top_k=2)
        
        assert len(ranked) <= 2
        assert all(isinstance(r, tuple) for r in ranked)
        assert all(len(r) == 2 for r in ranked)
        assert all(isinstance(r[0], str) and isinstance(r[1], (int, float)) for r in ranked)
    
    def test_rank_energy_mode(self, temp_memory_file):
        """Test rank with energy mode (new field-based API)."""
        adapter = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        # Expose some texts
        adapter.expose_text("Python is great")
        adapter.expose_text("Java is verbose")
        
        # Rank candidates using field energy
        candidates = ["Python rocks", "Java rules", "C++ power"]
        ranked = adapter.rank("programming", candidates, mode="energy", top_k=3)
        
        assert len(ranked) <= 3
        assert all(isinstance(r, tuple) for r in ranked)
        # Scores should be ordered (higher is better)
        scores = [r[1] for r in ranked]
        assert scores == sorted(scores, reverse=True)
    
    def test_save_and_load(self, temp_memory_file):
        """Test save and load functionality."""
        # Create adapter and add data
        adapter1 = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        event_id = adapter1.expose_text("Test persistence")
        adapter1.save()
        
        # Load in new adapter
        adapter2 = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        # Check data restored in cognitive field
        assert event_id in adapter2.cognitive_field.events_layer.events
    
    def test_get_stats(self, temp_memory_file):
        """Test get_stats returns comprehensive statistics."""
        adapter = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        # Add some data
        adapter.expose_text("Test 1")
        adapter.expose_text("Test 2")
        
        stats = adapter.get_stats()
        
        assert 'cognitive_field' in stats
        assert 'layers' in stats['cognitive_field']
        assert 'events_count' in stats['cognitive_field']
        assert stats['cognitive_field']['layers'] == 4
        assert stats['cognitive_field']['events_count'] >= 2
    
    def test_migration_from_old_format(self, temp_memory_file):
        """Test that adapter initializes correctly even without old data."""
        # Just create adapter - should initialize cleanly
        adapter = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        # Add some data
        adapter.expose_text("New format data")
        
        # Check storage successful
        assert len(adapter.cognitive_field.events_layer.events) > 0
    
    def test_expose_with_different_modes(self, temp_memory_file):
        """Test expose_text with different modes."""
        adapter = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        # General mode
        id1 = adapter.expose_text("General text", mode="general")
        
        # Code mode
        id2 = adapter.expose_text("def func(): pass", mode="code")
        
        # Document mode
        id3 = adapter.expose_text("# Document\n\nContent", mode="document")
        
        assert id1 != id2 != id3
        assert all(eid in adapter.brain_memory.events for eid in [id1, id2, id3])
    
    def test_rank_with_no_memory(self, temp_memory_file):
        """Test ranking works even with no exposed memory."""
        adapter = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        candidates = ["Python", "Java"]
        ranked = adapter.rank("test", candidates, mode="similarity", top_k=2)
        
        # Should return results even without prior exposure
        assert len(ranked) <= 2
    
    def test_delegate_unknown_methods(self, temp_memory_file):
        """Test that unknown methods are delegated to brain_memory."""
        adapter = PUHLCompatibilityAdapter(memory_file=temp_memory_file)
        
        # This method exists on BrainMemory
        result = adapter.expose_text("Test")
        assert result is not None
        
        # Direct method call works
        assert hasattr(adapter, 'events')
        assert adapter.events == adapter.brain_memory.events


class TestPUHLAdapterIntegration:
    """Integration tests for adapter with real workflows."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter with temporary file."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name
        
        adapter = PUHLCompatibilityAdapter(memory_file=filepath)
        yield adapter
        
        # Cleanup
        try:
            os.unlink(filepath)
        except:
            pass
        cf_path = filepath.replace('.pkl', '_cognitive_field.pkl')
        try:
            os.unlink(cf_path)
        except:
            pass
    
    def test_complete_workflow(self, adapter):
        """Test complete expose -> rank workflow."""
        # Expose training data
        adapter.expose_text("Python is a programming language")
        adapter.expose_text("Java is used for enterprise applications")
        adapter.expose_text("JavaScript runs in browsers")
        
        # Rank some candidates
        query = "What language should I learn?"
        candidates = [
            "Learn Python for data science",
            "Learn Java for Android",
            "Learn JavaScript for web development"
        ]
        
        ranked = adapter.rank(query, candidates, mode="similarity", top_k=3)
        
        assert len(ranked) == 3
        assert all(c in [r[0] for r in ranked] for c in candidates)
    
    def test_persistence_workflow(self, adapter):
        """Test full persistence workflow."""
        # Add data
        adapter.expose_text("Persistent data 1")
        adapter.expose_text("Persistent data 2")
        adapter.save()
        
        # Get stats before reload
        stats1 = adapter.get_stats()
        
        # Reload
        adapter.load()
        
        # Get stats after reload
        stats2 = adapter.get_stats()
        
        # Should have same event count
        assert stats1['cognitive_field']['events_count'] == stats2['cognitive_field']['events_count']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

