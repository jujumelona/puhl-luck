"""
Tests for Memory Management with Rust optimization.
"""

import pytest
import time
from puhl_luck._memory_management_field import MemoryManager, RUST_AVAILABLE


class TestMemoryManager:
    """Test MemoryManager initialization and configuration."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        manager = MemoryManager()
        
        assert manager.event_capacity == 10000
        assert manager.operator_capacity == 1000
        assert manager.transition_capacity == 5000
        assert manager.min_operator_confidence == 0.3
        assert manager.auto_prune is True
    
    def test_initialization_custom(self):
        """Test custom configuration."""
        manager = MemoryManager(
            event_capacity=5000,
            operator_capacity=500,
            transition_capacity=2000,
            min_operator_confidence=0.5,
            auto_prune=False,
        )
        
        assert manager.event_capacity == 5000
        assert manager.operator_capacity == 500
        assert manager.transition_capacity == 2000
        assert manager.min_operator_confidence == 0.5
        assert manager.auto_prune is False
    
    def test_statistics_initialized(self):
        """Test statistics counters initialized to zero."""
        manager = MemoryManager()
        
        assert manager.total_events_pruned == 0
        assert manager.total_operators_pruned == 0
        assert manager.total_transitions_pruned == 0
        assert manager.last_prune_time == 0


class TestEventPruning:
    """Test event pruning functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create manager with small capacity for testing."""
        return MemoryManager(event_capacity=5)
    
    def test_prune_events_below_capacity(self, manager):
        """Test that no events pruned when below capacity."""
        event_ids = ["e1", "e2", "e3"]
        event_novelty = {"e1": 0.8, "e2": 0.6, "e3": 0.4}
        event_last_accessed = {"e1": int(time.time()), "e2": int(time.time()), "e3": int(time.time())}
        event_activation = {"e1": 0.9, "e2": 0.5, "e3": 0.3}
        
        to_prune = manager.prune_events(
            event_ids,
            event_novelty,
            event_last_accessed,
            event_activation,
        )
        
        assert len(to_prune) == 0
    
    def test_prune_events_above_capacity(self, manager):
        """Test pruning when above capacity."""
        event_ids = ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]
        event_novelty = {eid: 0.5 for eid in event_ids}
        current_time = int(time.time())
        
        # Make some events old (low recency)
        event_last_accessed = {
            "e1": current_time - 86400 * 30,  # 30 days old
            "e2": current_time - 86400 * 20,  # 20 days old
            "e3": current_time,               # Recent
            "e4": current_time,               # Recent
            "e5": current_time - 86400 * 10,  # 10 days old
            "e6": current_time,               # Recent
            "e7": current_time - 86400 * 40,  # 40 days old
        }
        
        event_activation = {eid: 0.5 for eid in event_ids}
        
        to_prune = manager.prune_events(
            event_ids,
            event_novelty,
            event_last_accessed,
            event_activation,
        )
        
        # Should prune 2 events (7 - 5 = 2)
        assert len(to_prune) == 2
        
        # Should prune old events (e1, e7)
        assert all(eid in ["e1", "e2", "e7"] for eid in to_prune)
    
    def test_protected_events_not_pruned(self, manager):
        """Test that protected events are never pruned."""
        event_ids = ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]
        event_novelty = {eid: 0.1 for eid in event_ids}  # All low novelty
        current_time = int(time.time())
        event_last_accessed = {eid: current_time - 86400 * 100 for eid in event_ids}  # All old
        event_activation = {eid: 0.1 for eid in event_ids}  # All low activation
        
        # Protect some events
        protected = {"e1", "e2"}
        
        to_prune = manager.prune_events(
            event_ids,
            event_novelty,
            event_last_accessed,
            event_activation,
            protected_ids=protected,
        )
        
        # Should prune 2 events, but not protected ones
        assert len(to_prune) == 2
        assert "e1" not in to_prune
        assert "e2" not in to_prune
    
    def test_high_novelty_events_retained(self, manager):
        """Test that high novelty events are more likely to be retained."""
        event_ids = ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]
        current_time = int(time.time())
        
        # e1 has high novelty, others low
        event_novelty = {
            "e1": 0.9,  # High novelty
            "e2": 0.1, "e3": 0.1, "e4": 0.1, "e5": 0.1, "e6": 0.1, "e7": 0.1,
        }
        
        event_last_accessed = {eid: current_time - 86400 * 30 for eid in event_ids}  # All same age
        event_activation = {eid: 0.3 for eid in event_ids}  # All same activation
        
        to_prune = manager.prune_events(
            event_ids,
            event_novelty,
            event_last_accessed,
            event_activation,
        )
        
        # Should prune 2, but not e1 (high novelty)
        assert len(to_prune) == 2
        assert "e1" not in to_prune
    
    def test_statistics_updated(self, manager):
        """Test that pruning statistics are updated."""
        event_ids = ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]
        event_novelty = {eid: 0.5 for eid in event_ids}
        current_time = int(time.time())
        event_last_accessed = {eid: current_time for eid in event_ids}
        event_activation = {eid: 0.5 for eid in event_ids}
        
        initial_pruned = manager.total_events_pruned
        
        to_prune = manager.prune_events(
            event_ids,
            event_novelty,
            event_last_accessed,
            event_activation,
        )
        
        assert manager.total_events_pruned == initial_pruned + len(to_prune)
        assert manager.last_prune_time > 0


class TestOperatorPruning:
    """Test operator pruning functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create manager with small operator capacity."""
        return MemoryManager(
            operator_capacity=100,
            min_operator_confidence=0.4,
            min_operator_uses=3,
        )
    
    def test_prune_low_confidence_operators(self, manager):
        """Test pruning operators with low confidence."""
        operator_ids = ["op1", "op2", "op3"]
        operator_confidence = {"op1": 0.2, "op2": 0.5, "op3": 0.3}  # op1, op3 low
        operator_use_count = {"op1": 1, "op2": 10, "op3": 2}  # op1, op3 low use
        operator_success_count = {"op1": 1, "op2": 8, "op3": 1}
        current_time = int(time.time())
        operator_last_used = {eid: current_time for eid in operator_ids}
        
        to_prune = manager.prune_operators(
            operator_ids,
            operator_confidence,
            operator_use_count,
            operator_success_count,
            operator_last_used,
        )
        
        # Should prune op1 and op3 (low confidence AND low use count)
        assert len(to_prune) == 2
        assert "op1" in to_prune
        assert "op3" in to_prune
        assert "op2" not in to_prune
    
    def test_prune_old_low_success_operators(self, manager):
        """Test pruning operators with old timestamp and low success rate."""
        operator_ids = ["op1", "op2"]
        operator_confidence = {"op1": 0.6, "op2": 0.6}  # Both decent confidence
        operator_use_count = {"op1": 10, "op2": 10}
        operator_success_count = {"op1": 2, "op2": 8}  # op1 low success rate (20%)
        current_time = int(time.time())
        operator_last_used = {
            "op1": current_time - 86400 * 200,  # Very old
            "op2": current_time,                 # Recent
        }
        
        to_prune = manager.prune_operators(
            operator_ids,
            operator_confidence,
            operator_use_count,
            operator_success_count,
            operator_last_used,
        )
        
        # Should prune op1 (old AND low success rate)
        assert "op1" in to_prune
        assert "op2" not in to_prune
    
    def test_never_used_low_confidence_operators(self, manager):
        """Test pruning operators that were never used and have low confidence."""
        operator_ids = ["op1", "op2", "op3"]
        operator_confidence = {"op1": 0.3, "op2": 0.7, "op3": 0.2}  # op1, op3 low
        operator_use_count = {"op1": 0, "op2": 0, "op3": 0}  # All never used
        operator_success_count = {"op1": 0, "op2": 0, "op3": 0}
        current_time = int(time.time())
        operator_last_used = {eid: current_time for eid in operator_ids}
        
        to_prune = manager.prune_operators(
            operator_ids,
            operator_confidence,
            operator_use_count,
            operator_success_count,
            operator_last_used,
        )
        
        # Should prune op1 and op3 (never used AND low confidence)
        # Note: threshold is min_confidence * 1.5 = 0.6 for never-used operators
        assert "op1" in to_prune
        assert "op3" in to_prune
        assert "op2" not in to_prune  # High confidence, so kept even if never used


class TestTransitionPruning:
    """Test transition pruning functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create manager with small transition capacity."""
        return MemoryManager(transition_capacity=3)
    
    def test_prune_transitions_below_capacity(self, manager):
        """Test no pruning when below capacity."""
        transition_ids = ["t1", "t2"]
        transition_relevance = {"t1": 0.8, "t2": 0.6}
        transition_match_count = {"t1": 10, "t2": 5}
        current_time = int(time.time())
        transition_last_matched = {tid: current_time for tid in transition_ids}
        
        to_prune = manager.prune_transitions(
            transition_ids,
            transition_relevance,
            transition_match_count,
            transition_last_matched,
        )
        
        assert len(to_prune) == 0
    
    def test_prune_transitions_above_capacity(self, manager):
        """Test pruning when above capacity."""
        transition_ids = ["t1", "t2", "t3", "t4", "t5"]
        transition_relevance = {tid: 0.5 for tid in transition_ids}
        transition_match_count = {
            "t1": 20, "t2": 15, "t3": 10, "t4": 5, "t5": 1  # t5 least matched
        }
        current_time = int(time.time())
        transition_last_matched = {tid: current_time for tid in transition_ids}
        
        to_prune = manager.prune_transitions(
            transition_ids,
            transition_relevance,
            transition_match_count,
            transition_last_matched,
        )
        
        # Should prune 2 (5 - 3 = 2)
        assert len(to_prune) == 2
        
        # Should prune least matched (t4, t5)
        assert "t5" in to_prune
        assert "t4" in to_prune
    
    def test_prune_low_relevance_transitions(self, manager):
        """Test pruning prioritizes low relevance."""
        transition_ids = ["t1", "t2", "t3", "t4", "t5"]
        transition_relevance = {
            "t1": 0.9, "t2": 0.8, "t3": 0.7, "t4": 0.2, "t5": 0.1  # t4, t5 low
        }
        transition_match_count = {tid: 10 for tid in transition_ids}  # All same
        current_time = int(time.time())
        transition_last_matched = {tid: current_time for tid in transition_ids}
        
        to_prune = manager.prune_transitions(
            transition_ids,
            transition_relevance,
            transition_match_count,
            transition_last_matched,
        )
        
        # Should prune 2, prioritizing low relevance
        assert len(to_prune) == 2
        assert "t5" in to_prune
        assert "t4" in to_prune


class TestHealthMetrics:
    """Test memory health computation."""
    
    @pytest.fixture
    def manager(self):
        """Create manager."""
        return MemoryManager()
    
    def test_compute_health_basic(self, manager):
        """Test basic health computation."""
        metrics = manager.compute_health_metrics(
            event_count=100,
            operator_count=10,
            transition_count=50,
            event_novelty_values=[0.5, 0.6, 0.7],
            operator_confidence_values=[0.8, 0.9],
            transition_relevance_values=[0.6, 0.7, 0.8],
        )
        
        assert metrics['event_count'] == 100
        assert metrics['operator_count'] == 10
        assert metrics['transition_count'] == 50
        assert metrics['total_memory_items'] == 160
        
        assert 'avg_event_novelty' in metrics
        assert 'avg_operator_confidence' in metrics
        assert 'avg_transition_relevance' in metrics
        assert 'memory_health_score' in metrics
        assert 'needs_pruning' in metrics
    
    def test_health_needs_pruning_when_over_capacity(self, manager):
        """Test health indicates pruning needed when over capacity."""
        metrics = manager.compute_health_metrics(
            event_count=15000,  # Over capacity (10000)
            operator_count=10,
            transition_count=50,
            event_novelty_values=[0.5],
            operator_confidence_values=[0.8],
            transition_relevance_values=[0.6],
        )
        
        assert metrics['needs_pruning'] is True
    
    def test_health_no_pruning_when_under_capacity(self, manager):
        """Test health indicates no pruning when under capacity."""
        metrics = manager.compute_health_metrics(
            event_count=100,
            operator_count=10,
            transition_count=50,
            event_novelty_values=[0.5],
            operator_confidence_values=[0.8],
            transition_relevance_values=[0.6],
        )
        
        assert metrics['needs_pruning'] is False
    
    def test_health_includes_pruning_statistics(self):
        """Test health metrics include pruning statistics."""
        # Create manager with small capacity to force pruning
        manager = MemoryManager(event_capacity=5)
        
        # Do some pruning first
        event_ids = ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]
        manager.prune_events(
            event_ids,
            {eid: 0.5 for eid in event_ids},
            {eid: int(time.time()) for eid in event_ids},
            {eid: 0.5 for eid in event_ids},
        )
        
        metrics = manager.compute_health_metrics(
            event_count=5,
            operator_count=10,
            transition_count=50,
            event_novelty_values=[0.5],
            operator_confidence_values=[0.8],
            transition_relevance_values=[0.6],
        )
        
        assert 'total_events_pruned' in metrics
        assert 'total_operators_pruned' in metrics
        assert 'total_transitions_pruned' in metrics
        assert 'last_prune_time' in metrics
        assert metrics['total_events_pruned'] > 0


class TestAutoPruning:
    """Test automatic pruning trigger."""
    
    def test_should_auto_prune_when_over_capacity(self):
        """Test auto-prune triggered when over capacity."""
        manager = MemoryManager(
            event_capacity=100,
            operator_capacity=50,
            transition_capacity=200,
            auto_prune=True,
        )
        
        # Event count over capacity
        assert manager.should_auto_prune(
            event_count=150,
            operator_count=10,
            transition_count=50,
        ) is True
        
        # Operator count over capacity
        assert manager.should_auto_prune(
            event_count=50,
            operator_count=60,
            transition_count=50,
        ) is True
        
        # Transition count over capacity
        assert manager.should_auto_prune(
            event_count=50,
            operator_count=10,
            transition_count=250,
        ) is True
    
    def test_should_not_auto_prune_when_under_capacity(self):
        """Test auto-prune not triggered when under capacity."""
        manager = MemoryManager(
            event_capacity=100,
            operator_capacity=50,
            transition_capacity=200,
            auto_prune=True,
        )
        
        assert manager.should_auto_prune(
            event_count=50,
            operator_count=20,
            transition_count=100,
        ) is False
    
    def test_auto_prune_disabled(self):
        """Test auto-prune disabled when configured."""
        manager = MemoryManager(
            event_capacity=100,
            auto_prune=False,  # Disabled
        )
        
        # Should return False even when over capacity
        assert manager.should_auto_prune(
            event_count=150,
            operator_count=10,
            transition_count=50,
        ) is False


class TestRustAvailability:
    """Test Rust availability and fallback."""
    
    def test_rust_availability_reported(self):
        """Test that Rust availability is correctly reported."""
        # Should be True or False, not None
        assert RUST_AVAILABLE in [True, False]
    
    def test_manager_works_regardless_of_rust(self):
        """Test that manager works with or without Rust."""
        manager = MemoryManager(event_capacity=5)
        
        event_ids = ["e1", "e2", "e3", "e4", "e5", "e6"]
        event_novelty = {eid: 0.5 for eid in event_ids}
        current_time = int(time.time())
        event_last_accessed = {eid: current_time for eid in event_ids}
        event_activation = {eid: 0.5 for eid in event_ids}
        
        # Should work regardless of Rust availability
        to_prune = manager.prune_events(
            event_ids,
            event_novelty,
            event_last_accessed,
            event_activation,
        )
        
        assert len(to_prune) == 1  # 6 - 5 = 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
