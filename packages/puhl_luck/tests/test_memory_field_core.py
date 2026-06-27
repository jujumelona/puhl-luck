"""Tests for core data models in _memory_field_core.py.

Validates that all data structures for Task 1.2 can be properly instantiated
and meet the requirements for operator and transition memory layers.

Requirements tested: 1.7, 1.8, 4.1, 5.1, 5.2, 5.3, 5.4
"""

import numpy as np
import pytest

from puhl_luck._memory_field_core import (
    CandidateSource,
    CognitiveFieldSnapshot,
    CompletionPattern,
    ConflictMarker,
    ConflictType,
    FieldEnergy,
    GoalState,
    InputContext,
    OperatorRecord,
    OperatorType,
    StateField,
    StatePattern,
    StateTransition,
    TensionSource,
    TensionType,
    TransformationRule,
)


def test_operator_record_instantiation():
    """Test that OperatorRecord can be instantiated with required fields.
    
    Validates Requirement 1.7: Operator storage with transformation types.
    Validates Requirement 4.1: Storage as state transformation operators.
    """
    pattern = StatePattern(
        required_features={"question", "context"},
        required_concepts={"query"},
        incompleteness_markers=["missing_answer"],
        goal_patterns=["answer_question"],
    )
    
    transformation = TransformationRule(
        rule_type="template",
        parameters={"template": "The answer is {answer}"},
        confidence_threshold=0.8,
    )
    
    operator = OperatorRecord(
        operator_id="op_001",
        operator_type=OperatorType.EXPLANATION,
        pattern=pattern,
        preconditions=["has_question"],
        transformation=transformation,
        completion_template="Answer: {content}",
        confidence=0.85,
        usage_count=10,
        success_rate=0.90,
        generalization_level=2,
        induced_from=["trans_001", "trans_002"],
        timestamp=1234567890.0,
    )
    
    assert operator.operator_id == "op_001"
    assert operator.operator_type == OperatorType.EXPLANATION
    assert operator.confidence == 0.85
    assert operator.generalization_level == 2
    assert len(operator.induced_from) == 2


def test_operator_types_complete():
    """Test that all required operator types are available.
    
    Validates Requirement 1.7: Transformation types including completion_operator,
    repair_operator, explanation_operator, comparison_operator, transformation_operator.
    """
    required_types = [
        OperatorType.COMPLETION,
        OperatorType.REPAIR,
        OperatorType.EXPLANATION,
        OperatorType.COMPARISON,
        OperatorType.TRANSFORMATION,
        OperatorType.COMPOSITION,
    ]
    
    # Verify all types exist
    assert len(required_types) == 6
    
    # Verify enum values match expected strings
    assert OperatorType.COMPLETION.value == "completion"
    assert OperatorType.REPAIR.value == "repair"
    assert OperatorType.EXPLANATION.value == "explanation"
    assert OperatorType.COMPARISON.value == "comparison"
    assert OperatorType.TRANSFORMATION.value == "transformation"
    assert OperatorType.COMPOSITION.value == "composition"


def test_state_transition_partial_to_complete():
    """Test that StateTransition represents S_partial → S_complete.
    
    Validates Requirement 1.8: Store S_partial → S_complete pairs, not S_before → S_after.
    Validates Requirement 5.1: Transitions stored as S_partial → S_complete.
    """
    # Create a partial state (incomplete)
    partial_state = StateField(
        query_features=["code", "function"],
        query_hv=np.random.randn(1000),
        activated_events={"evt_1": 0.8},
        activated_concepts={"programming": 0.7},
        activated_operators={},
        conflict_markers=[],
        goal_states=[
            GoalState(
                goal_id="g1",
                goal_description="complete_code",
                satisfaction_level=0.0,  # Unsatisfied - incomplete
                constraints=[],
            )
        ],
        partial_outputs=["def calculate("],  # Incomplete code
    )
    
    # Create a complete state
    complete_state = StateField(
        query_features=["code", "function"],
        query_hv=np.random.randn(1000),
        activated_events={"evt_1": 0.8},
        activated_concepts={"programming": 0.7},
        activated_operators={},
        conflict_markers=[],
        goal_states=[
            GoalState(
                goal_id="g1",
                goal_description="complete_code",
                satisfaction_level=1.0,  # Satisfied - complete
                constraints=[],
            )
        ],
        partial_outputs=[],  # No partial outputs - complete
        previous_outputs=["def calculate(x, y):\n    return x + y"],
    )
    
    # Create transition
    transition = StateTransition(
        transition_id="trans_001",
        partial_state=partial_state,
        complete_state=complete_state,
        completion_vector=np.random.randn(1000),
        completion_features=["parameters", "return_statement"],
        modality="code",
        domain="code",
        timestamp=1234567890.0,
        relevance_count=5,
    )
    
    # Verify it represents partial → complete
    assert transition.partial_state.goal_states[0].satisfaction_level == 0.0
    assert transition.complete_state.goal_states[0].satisfaction_level == 1.0
    assert len(transition.partial_state.partial_outputs) > 0
    assert len(transition.complete_state.partial_outputs) == 0
    assert transition.domain == "code"


def test_state_transition_conversation_domain():
    """Test StateTransition for conversation domain.
    
    Validates Requirement 5.2: Record incomplete_context → completed_context.
    """
    partial_state = StateField(
        query_features=["question", "context"],
        query_hv=np.random.randn(1000),
        activated_events={"evt_1": 0.7},
        activated_concepts={"conversation": 0.8},
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
        partial_outputs=["The answer is"],
    )
    
    complete_state = StateField(
        query_features=["question", "context"],
        query_hv=np.random.randn(1000),
        activated_events={"evt_1": 0.7},
        activated_concepts={"conversation": 0.8},
        activated_operators={},
        conflict_markers=[],
        goal_states=[
            GoalState(
                goal_id="g1",
                goal_description="answer_question",
                satisfaction_level=1.0,
                constraints=[],
            )
        ],
        partial_outputs=[],
        previous_outputs=["The answer is Paris"],
    )
    
    transition = StateTransition(
        transition_id="trans_conv_001",
        partial_state=partial_state,
        complete_state=complete_state,
        completion_vector=np.random.randn(1000),
        completion_features=["answer", "location"],
        modality="text",
        domain="conversation",
        timestamp=1234567890.0,
        relevance_count=3,
    )
    
    assert transition.domain == "conversation"
    assert transition.modality == "text"


def test_state_transition_code_domain():
    """Test StateTransition for code domain.
    
    Validates Requirement 5.3: Record incomplete_code → completed_code.
    """
    transition = StateTransition(
        transition_id="trans_code_001",
        partial_state=StateField(
            query_features=["function"],
            query_hv=np.random.randn(1000),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=["def func("],
        ),
        complete_state=StateField(
            query_features=["function"],
            query_hv=np.random.randn(1000),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
            previous_outputs=["def func(x):\n    return x * 2"],
        ),
        completion_vector=np.random.randn(1000),
        completion_features=["parameters", "body"],
        modality="code",
        domain="code",
        timestamp=1234567890.0,
        relevance_count=7,
    )
    
    assert transition.domain == "code"
    assert transition.modality == "code"


def test_state_transition_reasoning_domain():
    """Test StateTransition for reasoning domain.
    
    Validates Requirement 5.4: Record incomplete_reasoning → completed_reasoning.
    """
    transition = StateTransition(
        transition_id="trans_reason_001",
        partial_state=StateField(
            query_features=["logic", "inference"],
            query_hv=np.random.randn(1000),
            activated_events={},
            activated_concepts={"reasoning": 0.9},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="complete_inference",
                    satisfaction_level=0.3,
                    constraints=[],
                )
            ],
            partial_outputs=["If A then B. A is true. Therefore"],
        ),
        complete_state=StateField(
            query_features=["logic", "inference"],
            query_hv=np.random.randn(1000),
            activated_events={},
            activated_concepts={"reasoning": 0.9},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="complete_inference",
                    satisfaction_level=1.0,
                    constraints=[],
                )
            ],
            partial_outputs=[],
            previous_outputs=["If A then B. A is true. Therefore B is true."],
        ),
        completion_vector=np.random.randn(1000),
        completion_features=["conclusion", "modus_ponens"],
        modality="text",
        domain="reasoning",
        timestamp=1234567890.0,
        relevance_count=4,
    )
    
    assert transition.domain == "reasoning"
    assert transition.modality == "text"


def test_completion_pattern_instantiation():
    """Test that CompletionPattern can be instantiated.
    
    Validates that completion patterns can be extracted and represented.
    """
    pattern = CompletionPattern(
        added_features=["answer", "explanation"],
        added_concepts={"knowledge", "facts"},
        completion_type="elaboration",
        modality="text",
        domain="conversation",
        completion_vector=np.random.randn(1000),
        reliability=0.92,
    )
    
    assert pattern.completion_type == "elaboration"
    assert pattern.reliability == 0.92
    assert len(pattern.added_features) == 2
    assert "knowledge" in pattern.added_concepts


def test_cognitive_field_snapshot_serialization_structure():
    """Test that CognitiveFieldSnapshot has all required fields.
    
    Validates that the snapshot can store all four memory layers.
    """
    snapshot = CognitiveFieldSnapshot(
        version="1.0.0",
        timestamp=1234567890.0,
        events={"evt_1": "event_data"},
        edges={(1, 2): 0.5, (2, 3): 0.8},
        feature_to_id={"feature1": 1, "feature2": 2},
        id_to_feature=["", "feature1", "feature2"],
        event_hv={"evt_1": np.random.randn(1000)},
        operators={
            "op_1": OperatorRecord(
                operator_id="op_1",
                operator_type=OperatorType.COMPLETION,
                pattern=StatePattern(
                    required_features=set(),
                    required_concepts=set(),
                    incompleteness_markers=[],
                    goal_patterns=[],
                ),
                preconditions=[],
                transformation=TransformationRule(
                    rule_type="template",
                    parameters={},
                    confidence_threshold=0.7,
                ),
                completion_template="",
                confidence=0.85,
                usage_count=5,
                success_rate=0.9,
                generalization_level=1,
                induced_from=[],
                timestamp=1234567890.0,
            )
        },
        transitions={
            "trans_1": StateTransition(
                transition_id="trans_1",
                partial_state=StateField(
                    query_features=[],
                    query_hv=np.random.randn(1000),
                    activated_events={},
                    activated_concepts={},
                    activated_operators={},
                    conflict_markers=[],
                    goal_states=[],
                    partial_outputs=[],
                ),
                complete_state=StateField(
                    query_features=[],
                    query_hv=np.random.randn(1000),
                    activated_events={},
                    activated_concepts={},
                    activated_operators={},
                    conflict_markers=[],
                    goal_states=[],
                    partial_outputs=[],
                ),
                completion_vector=np.random.randn(1000),
                completion_features=[],
                modality="text",
                domain="conversation",
                timestamp=1234567890.0,
                relevance_count=0,
            )
        },
        total_exposures=100,
        total_operators_induced=10,
        total_transitions_stored=50,
        window_size=1000,
        decay=0.95,
        hdc_dimensions=1000,
    )
    
    # Verify all layers are present
    assert snapshot.version == "1.0.0"
    assert len(snapshot.events) == 1  # Layer 1
    assert len(snapshot.operators) == 1  # Layer 3
    assert len(snapshot.transitions) == 1  # Layer 4
    assert snapshot.total_exposures == 100
    assert snapshot.hdc_dimensions == 1000


def test_state_pattern_and_transformation_rule():
    """Test StatePattern and TransformationRule dataclasses.
    
    Validates Requirement 4.1: Patterns as state transformation operators.
    """
    pattern = StatePattern(
        required_features={"query", "context", "incomplete"},
        required_concepts={"question", "knowledge"},
        incompleteness_markers=["missing_answer", "partial_response"],
        goal_patterns=["answer_question", "provide_explanation"],
    )
    
    assert len(pattern.required_features) == 3
    assert "question" in pattern.required_concepts
    assert "missing_answer" in pattern.incompleteness_markers
    
    rule = TransformationRule(
        rule_type="learned_transition",
        parameters={
            "transition_ids": ["trans_1", "trans_2"],
            "confidence": 0.85,
        },
        confidence_threshold=0.75,
    )
    
    assert rule.rule_type == "learned_transition"
    assert rule.confidence_threshold == 0.75
    assert len(rule.parameters["transition_ids"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
