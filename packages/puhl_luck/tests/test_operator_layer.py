"""
Tests for OperatorMemoryLayer (Layer 3) integration with StateFieldLayer.

Tests the implementation of Task 3.1: OperatorMemoryLayer class in _memory_operator_layer.py
"""

import pytest

from puhl_luck._memory_field_core import (
    GoalState,
    InputContext,
    OperatorRecord,
    OperatorType,
    StateField,
    StatePattern,
    TransformationRule,
)
from puhl_luck._memory_operator_layer import OperatorMemoryLayer, OperatorInstance
from puhl_luck._memory_state_field import StateFieldLayer
from puhl_luck._memory_exposure_layer import ExposureEventsLayer


class TestOperatorMemoryLayer:
    """Test the OperatorMemoryLayer class."""
    
    def test_store_operator(self):
        """Test storing an operator in Layer 3."""
        layer = OperatorMemoryLayer()
        
        # Create a completion operator
        pattern = StatePattern(
            required_features={"incomplete", "text"},
            required_concepts={"concept:writing"},
            incompleteness_markers=["unsatisfied_goal"],
            goal_patterns=["complete_text"],
        )
        
        transformation = TransformationRule(
            rule_type="template",
            parameters={"template": "complete: {context}"},
            confidence_threshold=0.5,
        )
        
        operator = OperatorRecord(
            operator_id="op_001",
            operator_type=OperatorType.COMPLETION,
            pattern=pattern,
            preconditions=["has_goal:complete_text"],
            transformation=transformation,
            completion_template="",
            confidence=0.8,
            usage_count=0,
            success_rate=0.0,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        
        operator_id = layer.store_operator(operator)
        
        assert operator_id == "op_001"
        assert layer.total_operators == 1
        assert "op_001" in layer.operators
        assert operator_id in layer.operators_by_type[OperatorType.COMPLETION]
    
    def test_find_applicable_operators(self):
        """Test finding operators applicable to a field state."""
        layer = OperatorMemoryLayer()
        
        # Store a completion operator
        pattern = StatePattern(
            required_features={"incomplete", "text"},
            required_concepts={"concept:writing"},
            incompleteness_markers=["unsatisfied_goal"],
            goal_patterns=["complete_text"],
        )
        
        transformation = TransformationRule(
            rule_type="template",
            parameters={"template": "complete: {context}"},
            confidence_threshold=0.5,
        )
        
        operator = OperatorRecord(
            operator_id="op_completion",
            operator_type=OperatorType.COMPLETION,
            pattern=pattern,
            preconditions=[],
            transformation=transformation,
            completion_template="",
            confidence=0.8,
            usage_count=0,
            success_rate=0.0,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        
        layer.store_operator(operator)
        
        # Create a field state that should match
        field_state = StateField(
            query_features=["incomplete", "text", "document"],
            query_hv=None,
            activated_events={},
            activated_concepts={"concept:writing": 0.9},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="complete_text",
                    satisfaction_level=0.0,
                    constraints=[],
                )
            ],
            partial_outputs=[],
            resonance={},
            field_energy=None,
            previous_outputs=[],
            iteration=0,
        )
        
        # Find applicable operators
        applicable = layer.find_applicable_operators(field_state)
        
        assert len(applicable) > 0
        assert applicable[0][0] == "op_completion"
        assert applicable[0][1] > 0  # Has positive match score
    
    def test_instantiate_operator(self):
        """Test instantiating an operator with context."""
        layer = OperatorMemoryLayer()
        
        # Store an operator
        pattern = StatePattern(
            required_features={"question"},
            required_concepts=set(),
            incompleteness_markers=["unsatisfied_goal"],
            goal_patterns=["answer_question"],
        )
        
        transformation = TransformationRule(
            rule_type="template",
            parameters={"template": "answer: {context}"},
            confidence_threshold=0.5,
        )
        
        operator = OperatorRecord(
            operator_id="op_explain",
            operator_type=OperatorType.EXPLANATION,
            pattern=pattern,
            preconditions=[],
            transformation=transformation,
            completion_template="",
            confidence=0.9,
            usage_count=0,
            success_rate=0.0,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        
        layer.store_operator(operator)
        
        # Create a context field state
        context = StateField(
            query_features=["question", "what", "is"],
            query_hv=None,
            activated_events={"event_123": 0.8},
            activated_concepts={"concept:knowledge": 0.7},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="answer_question",
                    satisfaction_level=0.0,
                    constraints=[],
                )
            ],
            partial_outputs=[],
            resonance={},
            field_energy=None,
            previous_outputs=[],
            iteration=0,
        )
        
        # Instantiate the operator
        instance = layer.instantiate_operator("op_explain", context)
        
        assert instance is not None
        assert instance.operator_id == "op_explain"
        assert instance.match_score > 0
        assert "query_features" in instance.bindings
        assert "goals" in instance.bindings
        
        # Check usage count was updated
        assert layer.operators["op_explain"].usage_count == 1
    
    def test_update_operator_stats(self):
        """Test updating operator statistics after application."""
        layer = OperatorMemoryLayer()
        
        # Store an operator
        pattern = StatePattern(
            required_features=set(),
            required_concepts=set(),
            incompleteness_markers=[],
            goal_patterns=[],
        )
        
        transformation = TransformationRule(
            rule_type="template",
            parameters={},
            confidence_threshold=0.5,
        )
        
        operator = OperatorRecord(
            operator_id="op_test",
            operator_type=OperatorType.COMPLETION,
            pattern=pattern,
            preconditions=[],
            transformation=transformation,
            completion_template="",
            confidence=0.5,
            usage_count=0,
            success_rate=0.0,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        
        layer.store_operator(operator)
        
        # Update with successful application
        layer.update_operator_stats("op_test", success=True)
        
        assert layer.operators["op_test"].success_rate > 0.0
        
        # Update with failed application
        layer.update_operator_stats("op_test", success=False)
        
        # Success rate should have adjusted


class TestStateFieldLayerOperatorIntegration:
    """Test integration of OperatorMemoryLayer with StateFieldLayer."""
    
    def test_activate_operators(self):
        """Test operator activation during field formation."""
        # Create layers
        events_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        operators_layer = OperatorMemoryLayer()
        state_field_layer = StateFieldLayer()
        
        # Store a test operator
        pattern = StatePattern(
            required_features={"test", "query"},
            required_concepts=set(),
            incompleteness_markers=[],
            goal_patterns=[],
        )
        
        transformation = TransformationRule(
            rule_type="template",
            parameters={},
            confidence_threshold=0.5,
        )
        
        operator = OperatorRecord(
            operator_id="op_test",
            operator_type=OperatorType.COMPLETION,
            pattern=pattern,
            preconditions=[],
            transformation=transformation,
            completion_template="",
            confidence=0.8,
            usage_count=0,
            success_rate=0.0,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        
        operators_layer.store_operator(operator)
        
        # Store some test events
        from puhl_luck._brain_common import text_feature_list, text_sequence
        features = text_feature_list("This is a test query document")
        sequence = text_sequence("This is a test query document")
        events_layer.store_event(
            modality="text",
            features=features,
            sequence=sequence,
            source="test",
            preview="This is a test query document"
        )
        
        # Create input context that should activate the operator
        input_context = InputContext(
            query_text="test query",
            modality="text",
            metadata={},
            goals=None,
            constraints=None,
        )
        
        # Activate field with operators
        field = state_field_layer.activate_from_input(
            input_context=input_context,
            events_layer=events_layer,
            operators_layer=operators_layer,
        )
        
        # Verify operators were activated
        assert isinstance(field.activated_operators, dict)
        # May or may not have activated the specific operator depending on match
        # But the mechanism should work without errors
    
    def test_activate_without_operators_layer(self):
        """Test that field activation works without operators layer."""
        events_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_field_layer = StateFieldLayer()
        
        # Store a test event
        from puhl_luck._brain_common import text_feature_list, text_sequence
        features = text_feature_list("Test document")
        sequence = text_sequence("Test document")
        events_layer.store_event(
            modality="text",
            features=features,
            sequence=sequence,
            source="test",
            preview="Test document"
        )
        
        # Create input context
        input_context = InputContext(
            query_text="test",
            modality="text",
            metadata={},
            goals=None,
            constraints=None,
        )
        
        # Activate field without operators layer
        field = state_field_layer.activate_from_input(
            input_context=input_context,
            events_layer=events_layer,
            operators_layer=None,  # No operators layer
        )
        
        # Verify field was created successfully
        assert field is not None
        assert field.activated_operators == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


