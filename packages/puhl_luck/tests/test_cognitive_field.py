"""
Unit tests for CognitiveField class (Task 9.1).

Tests the main orchestrator that coordinates all four memory layers
and the recursive stabilization loop.
"""

import numpy as np
import pytest
import tempfile
import os

from puhl_luck._memory_cognitive_field import CognitiveField, GenerationResult
from puhl_luck._memory_field_core import InputContext


class TestCognitiveField:
    """Test suite for CognitiveField class."""
    
    def test_initialization(self):
        """Test that CognitiveField initializes correctly with all layers."""
        field = CognitiveField(
            window_size=50,
            decay=0.9,
            hdc_dimensions=5000,
            max_stabilization_iterations=10,
            convergence_threshold=0.1,
        )
        
        assert field.window_size == 50
        assert field.decay == 0.9
        assert field.hdc_dimensions == 5000
        assert field.max_stabilization_iterations == 10
        assert field.convergence_threshold == 0.1
        
        # Check all layers initialized
        assert field.events_layer is not None
        assert field.state_field_layer is not None
        assert field.operators_layer is not None
        assert field.transitions_layer is not None
        
        # Check process components
        assert field.field_formation is not None
        assert field.energy_computer is not None
        assert field.candidate_emergence is not None
        assert field.recursive_stabilization is not None
        
        # Check statistics
        assert field.total_generations == 0
        assert field.total_exposures == 0
    
    def test_expose_text(self):
        """Test exposing text for learning."""
        field = CognitiveField()
        
        event_id = field.expose_text(
            "This is a test sentence.",
            metadata={"source": "test"}
        )
        
        assert isinstance(event_id, str)
        assert len(event_id) > 0
        assert field.total_exposures == 1
        
        # Check event stored in Layer 1
        assert event_id in field.events_layer.events
    
    def test_expose_multiple_texts(self):
        """Test exposing multiple texts."""
        field = CognitiveField()
        
        texts = [
            "First sentence about AI.",
            "Second sentence about machine learning.",
            "Third sentence about neural networks.",
        ]
        
        event_ids = []
        for text in texts:
            event_id = field.expose_text(text)
            event_ids.append(event_id)
        
        assert len(event_ids) == 3
        assert len(set(event_ids)) == 3  # All unique
        assert field.total_exposures == 3
    
    def test_form_field(self):
        """Test field formation from input context."""
        field = CognitiveField()
        
        # Expose some text first
        field.expose_text("Python is a programming language.")
        field.expose_text("Machine learning uses Python.")
        
        # Form field
        context = InputContext(
            query_text="What is Python?",
            query_features=["what", "python"],
            query_hv=np.zeros(100, dtype=np.int8),
            modality="text",
            domain="conversation",
        )
        
        state_field = field.form_field(context)
        
        assert state_field is not None
        assert len(state_field.query_features) > 0
        # Should have activated some events
        assert isinstance(state_field.activated_events, dict)
    
    def test_generate_basic(self):
        """Test basic generation workflow."""
        field = CognitiveField(max_stabilization_iterations=3)
        
        # Expose some knowledge
        field.expose_text("Paris is the capital of France.")
        field.expose_text("France is in Europe.")
        
        # Generate
        result = field.generate(
            query="What is the capital of France?",
            modality="text",
            domain="conversation",
        )
        
        assert isinstance(result, GenerationResult)
        assert isinstance(result.output, str)
        assert isinstance(result.converged, bool)
        assert result.iterations > 0
        assert result.iterations <= 3
        assert isinstance(result.final_energy, float)
        assert len(result.energy_history) == result.iterations
        assert field.total_generations == 1
    
    def test_generate_with_empty_memory(self):
        """Test generation with no exposed knowledge."""
        field = CognitiveField(max_stabilization_iterations=3)
        
        result = field.generate(
            query="What is machine learning?",
            max_iterations=2,
        )
        
        # Should still complete without error
        assert isinstance(result, GenerationResult)
        assert result.iterations <= 2
    
    def test_rank_candidates(self):
        """Test ranking candidates using field-based scoring."""
        field = CognitiveField()
        
        # Expose relevant knowledge
        field.expose_text("Python is a high-level programming language.")
        field.expose_text("JavaScript is used for web development.")
        
        # Rank candidates
        scores = field.rank(
            query="Which language is better for beginners?",
            candidates=[
                "Python is easier to learn",
                "C++ has more features",
                "Assembly is fast",
            ],
        )
        
        assert len(scores) == 3
        assert all(isinstance(s, (int, float)) for s in scores)
    
    def test_save_and_load(self):
        """Test saving and loading cognitive field state."""
        field1 = CognitiveField()
        
        # Add some data
        field1.expose_text("Test content 1")
        field1.expose_text("Test content 2")
        result1 = field1.generate("test query", max_iterations=2)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "cognitive_field.pkl")
            field1.save(save_path)
            
            # Load into new instance
            field2 = CognitiveField()
            field2.load(save_path)
            
            # Check state restored
            assert field2.total_exposures == field1.total_exposures
            assert field2.window_size == field1.window_size
            assert field2.decay == field1.decay
            assert len(field2.events_layer.events) == len(field1.events_layer.events)
    
    def test_persistence_preserves_layers(self):
        """Test that save/load preserves all layer data."""
        field1 = CognitiveField()
        
        # Populate layers
        field1.expose_text("Layer 1 content")
        
        # Manually add operator to Layer 3 (simplified)
        from puhl_luck._memory_field_core import (
            OperatorRecord, OperatorType, StatePattern,
            TransformationRule
        )
        
        operator = OperatorRecord(
            operator_id="test_op",
            operator_type=OperatorType.COMPLETION,
            pattern=StatePattern(
                required_features={"test"},
                required_concepts=set(),
                incompleteness_markers=[],
                goal_patterns=[],
            ),
            preconditions=[],
            transformation=TransformationRule(
                rule_type="template",
                parameters={},
                confidence_threshold=0.5,
            ),
            completion_template="test template",
            confidence=0.8,
            usage_count=0,
            success_rate=0.0,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        field1.operators_layer.store_operator(operator)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "field.pkl")
            field1.save(save_path)
            
            field2 = CognitiveField()
            field2.load(save_path)
            
            # Check operator restored
            assert "test_op" in field2.operators_layer.operators
            assert field2.operators_layer.operators["test_op"].confidence == 0.8
    
    def test_backward_compatible_api(self):
        """Test backward compatible PUHL API methods."""
        field = CognitiveField()
        
        # expose_text
        event_id1 = field.expose_text("Test text", {"label": "test"})
        assert isinstance(event_id1, str)
        
        # rank
        scores = field.rank(
            query="test query",
            candidates=["option1", "option2"],
        )
        assert len(scores) == 2
    
    def test_multiple_generations(self):
        """Test multiple generations update statistics correctly."""
        field = CognitiveField(max_stabilization_iterations=2)
        
        field.expose_text("Knowledge base content")
        
        # Generate multiple times
        for i in range(3):
            result = field.generate(f"Query {i}", max_iterations=2)
            assert field.total_generations == i + 1
    
    def test_layer_coordination(self):
        """Test that all layers coordinate correctly."""
        field = CognitiveField()
        
        # Expose creates Layer 1 entries
        event_id = field.expose_text("Coordination test")
        assert event_id in field.events_layer.events
        
        # Form field activates Layer 1 and creates Layer 2 state
        context = InputContext(
            query_text="test",
            query_features=["test"],
            query_hv=np.zeros(100, dtype=np.int8),
            modality="text",
            domain="test",
        )
        state = field.form_field(context)
        assert state is not None
        
        # Layer 3 (operators) and Layer 4 (transitions) available
        assert field.operators_layer is not None
        assert field.transitions_layer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
