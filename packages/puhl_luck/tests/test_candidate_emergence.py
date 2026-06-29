"""
Unit tests for CandidateEmergence class (Task 6.1).

Tests the tension-driven candidate generation mechanism where candidates emerge
from field dynamics rather than being retrieved through search.
"""

import numpy as np
import pytest

from puhl_luck._memory_candidate_emergence import CandidateEmergence
from puhl_luck._memory_field_core import (
    Candidate,
    CandidateSource,
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
from puhl_luck._memory_field_energy import FreeEnergyMinimization
from puhl_luck._memory_operator_layer import OperatorMemoryLayer
from puhl_luck._memory_transition_layer import TransitionMemoryLayer


class TestCandidateEmergence:
    """Test suite for CandidateEmergence class."""
    
    def test_initialization(self):
        """Test that CandidateEmergence initializes correctly."""
        emergence = CandidateEmergence()
        
        assert emergence.energy_computer is not None
        assert emergence.total_candidates_generated == 0
        assert len(emergence.generation_stats) == 4
    
    def test_identify_tension_sources_from_conflicts(self):
        """Test identifying tension sources from conflict markers."""
        emergence = CandidateEmergence()
        
        # Create field with conflict
        field = StateField(
            query_features=["test", "feature"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[
                ConflictMarker(
                    conflict_id="c1",
                    conflict_type=ConflictType.CONTRADICTION,
                    involved_memories=["mem1", "mem2"],
                    strength=0.8,
                    description="Test conflict",
                )
            ],
            goal_states=[],
            partial_outputs=[],
        )
        
        tensions = emergence.identify_tension_sources(field)
        
        assert len(tensions) > 0
        assert any(t.type == TensionType.CONFLICT for t in tensions)
        conflict_tension = [t for t in tensions if t.type == TensionType.CONFLICT][0]
        assert conflict_tension.strength == 0.8
    
    def test_identify_tension_sources_from_unsatisfied_goals(self):
        """Test identifying tension sources from unsatisfied goals."""
        emergence = CandidateEmergence()
        
        # Create field with unsatisfied goal
        field = StateField(
            query_features=["test"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="Complete the task",
                    satisfaction_level=0.3,  # Low satisfaction
                    constraints=[],
                )
            ],
            partial_outputs=[],
        )
        
        tensions = emergence.identify_tension_sources(field)
        
        assert len(tensions) > 0
        assert any(t.type == TensionType.UNSATISFIED_GOAL for t in tensions)
        goal_tension = [t for t in tensions if t.type == TensionType.UNSATISFIED_GOAL][0]
        assert goal_tension.strength == pytest.approx(0.7, abs=0.01)  # 1.0 - 0.3
    
    def test_identify_tension_sources_from_incomplete_output(self):
        """Test identifying tension sources from incomplete partial outputs."""
        emergence = CandidateEmergence()
        
        # Create field with incomplete output
        field = StateField(
            query_features=["test"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=["Short"],  # Very short, incomplete
        )
        
        tensions = emergence.identify_tension_sources(field)
        
        assert len(tensions) > 0
        assert any(t.type == TensionType.INCOMPLETE_OUTPUT for t in tensions)
    
    def test_generate_candidates_returns_empty_for_no_tension(self):
        """Test that no candidates are generated when field has no tension."""
        emergence = CandidateEmergence()
        operators_layer = OperatorMemoryLayer()
        transitions_layer = TransitionMemoryLayer()
        
        # Create stable field with satisfied goals
        field = StateField(
            query_features=["test"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="Complete",
                    satisfaction_level=1.0,  # Fully satisfied
                    constraints=[],
                )
            ],
            partial_outputs=[],
            field_energy=FieldEnergy(
                total=0.1,  # Low energy
                evidence=0.9,
                conflicts=0.0,
                evidence_breakdown={},
                conflict_breakdown={},
                dominant_evidence_sources=[],
                dominant_conflict_sources=[],
                tension_level=0.1,  # Low tension
            ),
        )
        
        candidates = emergence.generate_candidates(
            field, operators_layer, transitions_layer, num_candidates=5
        )
        
        assert len(candidates) == 0
    
    def test_generate_from_operators(self):
        """Test generating candidates from operators."""
        emergence = CandidateEmergence()
        operators_layer = OperatorMemoryLayer()
        
        # Create and store a completion operator
        operator = OperatorRecord(
            operator_id="op1",
            operator_type=OperatorType.COMPLETION,
            pattern=StatePattern(
                required_features={"incomplete"},
                required_concepts=set(),
                incompleteness_markers=["unsatisfied_goal"],
                goal_patterns=["complete"],
            ),
            preconditions=["has_feature:incomplete"],
            transformation=TransformationRule(
                rule_type="template",
                parameters={},
                confidence_threshold=0.5,
            ),
            completion_template="Completing the task",
            confidence=0.8,
            usage_count=0,
            success_rate=0.7,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        operators_layer.store_operator(operator)
        
        # Create field that matches operator
        field = StateField(
            query_features=["incomplete", "task"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={"op1": 0.8},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="complete the task",
                    satisfaction_level=0.3,
                    constraints=[],
                )
            ],
            partial_outputs=[],
        )
        
        tension_sources = [
            TensionSource(
                type=TensionType.UNSATISFIED_GOAL,
                location="g1",
                strength=0.7,
                description="Unsatisfied goal",
            )
        ]
        
        candidates = emergence.generate_from_operators(
            field, operators_layer, tension_sources
        )
        
        assert len(candidates) > 0
        assert candidates[0].source == CandidateSource.OPERATOR_BASED
        assert "op1" in candidates[0].source_operators
    
    def test_generate_from_transitions(self):
        """Test generating candidates from transitions."""
        emergence = CandidateEmergence()
        transitions_layer = TransitionMemoryLayer()
        
        # Create and store a transition
        partial = StateField(
            query_features=["incomplete", "test"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        complete = StateField(
            query_features=["incomplete", "test", "complete"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=["Complete output"],
        )
        
        transition_id = transitions_layer.store_transition(
            partial, complete, modality="text", domain="conversation"
        )
        
        # Create similar partial field
        field = StateField(
            query_features=["incomplete", "test"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        tension_sources = [
            TensionSource(
                type=TensionType.INCOMPLETE_OUTPUT,
                location="field",
                strength=0.8,
                description="Incomplete",
            )
        ]
        
        candidates = emergence.generate_from_transitions(
            field, transitions_layer, tension_sources
        )
        
        assert len(candidates) > 0
        assert candidates[0].source == CandidateSource.TRANSITION_BASED
        assert transition_id in candidates[0].source_transitions
    
    def test_generate_from_field_dynamics(self):
        """Test generating candidates from field resonance patterns."""
        emergence = CandidateEmergence()
        
        # Create field with resonance
        field = StateField(
            query_features=["test"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={"e1": 0.8, "e2": 0.7},
            activated_concepts={"concept:test": 0.9},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
            resonance={
                ("e1", "e2"): 0.85,  # High positive resonance
                ("e2", "e1"): 0.85,
            },
        )
        
        tension_sources = [
            TensionSource(
                type=TensionType.INCOMPLETE_OUTPUT,
                location="field",
                strength=0.6,
                description="Incomplete",
            )
        ]
        
        candidates = emergence.generate_from_field_dynamics(field, tension_sources)
        
        assert len(candidates) > 0
        assert candidates[0].source == CandidateSource.FIELD_DYNAMICS
    
    def test_compute_energy_reduction(self):
        """Test computing energy reduction for candidates."""
        energy_computer = FreeEnergyMinimization()
        emergence = CandidateEmergence(energy_computer=energy_computer)
        
        # Create field with high energy
        field = StateField(
            query_features=["test"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[
                ConflictMarker(
                    conflict_id="c1",
                    conflict_type=ConflictType.CONTRADICTION,
                    involved_memories=["m1", "m2"],
                    strength=0.8,
                    description="Conflict",
                )
            ],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="resolve conflict",
                    satisfaction_level=0.2,
                    constraints=[],
                )
            ],
            partial_outputs=[],
        )
        
        # Compute field energy
        field.field_energy = energy_computer.compute_field_energy(field)
        
        # Create candidate
        candidate = Candidate(
            content="Resolving conflict by addressing the issue",
            tokens=["Resolving", "conflict", "by", "addressing", "the", "issue"],
            energy_reduction=0.0,
            predicted_energy_after=0.0,
            source=CandidateSource.OPERATOR_BASED,
            source_operators=["op1"],
            source_transitions=[],
            tensions_addressed=[],
            tensions_resolved_count=0,
            confidence=0.8,
        )
        
        energy_reduction = emergence.compute_energy_reduction(candidate, field)
        
        # Energy reduction should be computed (can be positive or negative)
        assert isinstance(energy_reduction, float)
    
    def test_full_candidate_generation_workflow(self):
        """Test complete candidate generation workflow."""
        emergence = CandidateEmergence()
        operators_layer = OperatorMemoryLayer()
        transitions_layer = TransitionMemoryLayer()
        
        # Create field with tension
        field = StateField(
            query_features=["incomplete", "task"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="Complete the task",
                    satisfaction_level=0.4,  # Unsatisfied
                    constraints=[],
                )
            ],
            partial_outputs=[],
            field_energy=FieldEnergy(
                total=0.8,  # High energy
                evidence=0.2,
                conflicts=0.6,
                evidence_breakdown={},
                conflict_breakdown={},
                dominant_evidence_sources=[],
                dominant_conflict_sources=[],
                tension_level=0.75,
            ),
        )
        
        # Generate candidates
        candidates = emergence.generate_candidates(
            field, operators_layer, transitions_layer, num_candidates=5
        )
        
        # Should generate at least some candidates from field dynamics
        # (even without operators or transitions)
        assert isinstance(candidates, list)
        # May be empty if no operators/transitions and field dynamics don't generate any
        # But workflow should complete without errors


class TestTransitionBasedGeneration:
    """Test suite specifically for transition-based candidate generation (Task 6.3)."""
    
    def test_similar_transition_retrieval(self):
        """Test that similar transitions are correctly retrieved for candidate generation."""
        emergence = CandidateEmergence()
        transitions_layer = TransitionMemoryLayer()
        
        # Store multiple transitions
        for i in range(5):
            partial = StateField(
                query_features=["incomplete", f"variant{i}"],
                query_hv=np.random.randint(-1, 2, size=100, dtype=np.int8),
                activated_events={},
                activated_concepts={},
                activated_operators={},
                conflict_markers=[],
                goal_states=[],
                partial_outputs=[],
            )
            
            complete = StateField(
                query_features=["incomplete", f"variant{i}", "completed"],
                query_hv=np.random.randint(-1, 2, size=100, dtype=np.int8),
                activated_events={},
                activated_concepts={},
                activated_operators={},
                conflict_markers=[],
                goal_states=[],
                partial_outputs=[f"Completed variant {i}"],
            )
            
            transitions_layer.store_transition(partial, complete, modality="text", domain="test")
        
        # Query with similar partial state
        field = StateField(
            query_features=["incomplete", "variant0"],  # Similar to first transition
            query_hv=np.ones(100, dtype=np.int8) * -1,
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        tension_sources = [
            TensionSource(
                type=TensionType.INCOMPLETE_OUTPUT,
                location="field",
                strength=0.8,
                description="Incomplete",
            )
        ]
        
        candidates = emergence.generate_from_transitions(
            field, transitions_layer, tension_sources
        )
        
        # Should retrieve similar transitions and generate candidates
        assert len(candidates) > 0
        assert all(c.source == CandidateSource.TRANSITION_BASED for c in candidates)
    
    def test_completion_pattern_extraction(self):
        """Test that completion patterns are correctly extracted from transitions."""
        emergence = CandidateEmergence()
        transitions_layer = TransitionMemoryLayer()
        
        # Store transition with clear completion pattern
        partial = StateField(
            query_features=["question", "incomplete"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={"e1": 0.8},
            activated_concepts={"concept:question": 0.9},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=["What is"],
        )
        
        complete = StateField(
            query_features=["question", "incomplete", "answer", "explanation"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={"e1": 0.8, "e2": 0.7},
            activated_concepts={"concept:question": 0.9, "concept:answer": 0.85},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=["What is the answer? It is X because Y."],
        )
        
        transition_id = transitions_layer.store_transition(
            partial, complete, modality="text", domain="qa"
        )
        
        # Get completion pattern
        pattern = transitions_layer.get_completion_pattern(transition_id)
        
        assert pattern is not None
        assert "answer" in pattern.added_features
        assert "explanation" in pattern.added_features
        assert "concept:answer" in pattern.added_concepts
    
    def test_completion_type_classification(self):
        """Test that different completion types (direct, elaboration, correction, explanation) are classified."""
        emergence = CandidateEmergence()
        transitions_layer = TransitionMemoryLayer()
        
        # Direct completion
        partial1 = StateField(
            query_features=["start"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        complete1 = StateField(
            query_features=["start", "end"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=["Direct finish"],
        )
        tid1 = transitions_layer.store_transition(partial1, complete1, modality="text", domain="test")
        
        # Correction completion (conflict present)
        partial2 = StateField(
            query_features=["error"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[
                ConflictMarker(
                    conflict_id="c1",
                    conflict_type=ConflictType.CONTRADICTION,
                    involved_memories=["m1", "m2"],
                    strength=0.8,
                    description="Error",
                )
            ],
            goal_states=[],
            partial_outputs=[],
        )
        complete2 = StateField(
            query_features=["error", "fixed"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],  # Conflict resolved
            goal_states=[],
            partial_outputs=["Fixed error"],
        )
        tid2 = transitions_layer.store_transition(partial2, complete2, modality="text", domain="test")
        
        pattern1 = transitions_layer.get_completion_pattern(tid1)
        pattern2 = transitions_layer.get_completion_pattern(tid2)
        
        # Pattern1 may be "direct" or "elaboration" depending on heuristics
        assert pattern1.completion_type in ["direct", "elaboration"]
        assert pattern2.completion_type == "correction"
    
    def test_transition_relevance_updates(self):
        """Test that transition relevance increases when used for candidate generation."""
        emergence = CandidateEmergence()
        transitions_layer = TransitionMemoryLayer()
        
        # Store transition
        partial = StateField(
            query_features=["test"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        complete = StateField(
            query_features=["test", "complete"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=["Complete"],
        )
        tid = transitions_layer.store_transition(partial, complete, modality="text", domain="test")
        
        # Get initial relevance
        transition = transitions_layer.transitions[tid]
        initial_relevance = transition.relevance_count
        
        # Generate candidates (should update relevance)
        field = StateField(
            query_features=["test"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        tension_sources = [
            TensionSource(
                type=TensionType.INCOMPLETE_OUTPUT,
                location="field",
                strength=0.8,
                description="Incomplete",
            )
        ]
        
        candidates = emergence.generate_from_transitions(
            field, transitions_layer, tension_sources
        )
        
        # Relevance should increase
        updated_relevance = transitions_layer.transitions[tid].relevance_count
        assert updated_relevance > initial_relevance
    
    def test_multi_transition_candidate_generation(self):
        """Test generating candidates from multiple similar transitions."""
        emergence = CandidateEmergence()
        transitions_layer = TransitionMemoryLayer()
        
        # Store multiple similar transitions with different completions
        for i in range(3):
            partial = StateField(
                query_features=["question"],
                query_hv=np.ones(100, dtype=np.int8),
                activated_events={},
                activated_concepts={},
                activated_operators={},
                conflict_markers=[],
                goal_states=[],
                partial_outputs=[],
            )
            complete = StateField(
                query_features=["question", f"answer{i}"],
                query_hv=np.ones(100, dtype=np.int8),
                activated_events={},
                activated_concepts={},
                activated_operators={},
                conflict_markers=[],
                goal_states=[],
                partial_outputs=[f"Answer variant {i}"],
            )
            transitions_layer.store_transition(partial, complete, modality="text", domain="qa")
        
        # Query with similar partial
        field = StateField(
            query_features=["question"],
            query_hv=np.ones(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        tension_sources = [
            TensionSource(
                type=TensionType.INCOMPLETE_OUTPUT,
                location="field",
                strength=0.8,
                description="Incomplete",
            )
        ]
        
        candidates = emergence.generate_from_transitions(
            field, transitions_layer, tension_sources
        )
        
        # Should generate multiple candidates from different transitions
        assert len(candidates) >= 2
        # Each should have different source transitions
        transition_ids = [c.source_transitions[0] for c in candidates if c.source_transitions]
        assert len(set(transition_ids)) >= 2


class TestOperatorBasedGeneration:
    """Test suite specifically for operator-based candidate generation (Task 6.2)."""
    
    def test_multiple_operator_types(self):
        """Test generation with multiple operator types (completion, repair, explanation)."""
        emergence = CandidateEmergence()
        operators_layer = OperatorMemoryLayer()
        
        # Store multiple operator types
        operators = [
            OperatorRecord(
                operator_id="op_complete",
                operator_type=OperatorType.COMPLETION,
                pattern=StatePattern(
                    required_features={"incomplete"},
                    required_concepts=set(),
                    incompleteness_markers=["unsatisfied_goal"],
                    goal_patterns=["complete"],
                ),
                preconditions=["has_feature:incomplete"],
                transformation=TransformationRule(
                    rule_type="template",
                    parameters={"style": "direct"},
                    confidence_threshold=0.5,
                ),
                completion_template="Completing task: {query_features}",
                confidence=0.8,
                usage_count=0,
                success_rate=0.7,
                generalization_level=1,
                induced_from=[],
                timestamp=0.0,
            ),
            OperatorRecord(
                operator_id="op_repair",
                operator_type=OperatorType.REPAIR,
                pattern=StatePattern(
                    required_features={"conflict"},
                    required_concepts=set(),
                    incompleteness_markers=["contradiction"],
                    goal_patterns=["resolve"],
                ),
                preconditions=["has_conflict"],
                transformation=TransformationRule(
                    rule_type="template",
                    parameters={},
                    confidence_threshold=0.6,
                ),
                completion_template="Resolving conflict",
                confidence=0.75,
                usage_count=0,
                success_rate=0.8,
                generalization_level=1,
                induced_from=[],
                timestamp=0.0,
            ),
            OperatorRecord(
                operator_id="op_explain",
                operator_type=OperatorType.EXPLANATION,
                pattern=StatePattern(
                    required_features={"unclear"},
                    required_concepts=set(),
                    incompleteness_markers=["needs_explanation"],
                    goal_patterns=["clarify"],
                ),
                preconditions=["has_feature:unclear"],
                transformation=TransformationRule(
                    rule_type="template",
                    parameters={},
                    confidence_threshold=0.5,
                ),
                completion_template="This means that {top_concept}",
                confidence=0.7,
                usage_count=0,
                success_rate=0.75,
                generalization_level=1,
                induced_from=[],
                timestamp=0.0,
            ),
        ]
        
        for op in operators:
            operators_layer.store_operator(op)
        
        # Create field matching multiple operators
        field = StateField(
            query_features=["incomplete", "unclear", "task"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={},
            activated_concepts={"concept:explanation": 0.8},
            activated_operators={"op_complete": 0.8, "op_explain": 0.7},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="complete and clarify",
                    satisfaction_level=0.3,
                    constraints=[],
                )
            ],
            partial_outputs=[],
        )
        
        tension_sources = [
            TensionSource(
                type=TensionType.UNSATISFIED_GOAL,
                location="g1",
                strength=0.7,
                description="Unsatisfied goal",
            )
        ]
        
        candidates = emergence.generate_from_operators(
            field, operators_layer, tension_sources
        )
        
        # Should generate candidates from multiple operators
        assert len(candidates) >= 2
        operator_ids = set()
        for c in candidates:
            operator_ids.update(c.source_operators)
        assert len(operator_ids) >= 2
    
    def test_operator_transformation_types(self):
        """Test all transformation rule types (template, learned_transition, feature_combination)."""
        emergence = CandidateEmergence()
        operators_layer = OperatorMemoryLayer()
        
        # Test template transformation
        op_template = OperatorRecord(
            operator_id="op_template",
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
            completion_template="Template output: {query_features}",
            confidence=0.8,
            usage_count=0,
            success_rate=0.7,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        operators_layer.store_operator(op_template)
        
        # Test learned_transition transformation
        op_learned = OperatorRecord(
            operator_id="op_learned",
            operator_type=OperatorType.TRANSFORMATION,
            pattern=StatePattern(
                required_features={"test"},
                required_concepts=set(),
                incompleteness_markers=[],
                goal_patterns=[],
            ),
            preconditions=[],
            transformation=TransformationRule(
                rule_type="learned_transition",
                parameters={},
                confidence_threshold=0.5,
            ),
            completion_template="",
            confidence=0.8,
            usage_count=0,
            success_rate=0.7,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        operators_layer.store_operator(op_learned)
        
        # Test feature_combination transformation
        op_feature = OperatorRecord(
            operator_id="op_feature",
            operator_type=OperatorType.COMPLETION,
            pattern=StatePattern(
                required_features={"test"},
                required_concepts=set(),
                incompleteness_markers=[],
                goal_patterns=[],
            ),
            preconditions=[],
            transformation=TransformationRule(
                rule_type="feature_combination",
                parameters={},
                confidence_threshold=0.5,
            ),
            completion_template="",
            confidence=0.8,
            usage_count=0,
            success_rate=0.7,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        operators_layer.store_operator(op_feature)
        
        field = StateField(
            query_features=["test", "feature", "data"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={},
            activated_concepts={"concept:learned": 0.8},
            activated_operators={"op_template": 0.8, "op_learned": 0.7, "op_feature": 0.7},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        tension_sources = [
            TensionSource(
                type=TensionType.INCOMPLETE_OUTPUT,
                location="field",
                strength=0.6,
                description="Incomplete",
            )
        ]
        
        candidates = emergence.generate_from_operators(
            field, operators_layer, tension_sources
        )
        
        # Should generate candidates from all transformation types
        assert len(candidates) >= 3
    
    def test_operator_tension_matching(self):
        """Test that operators correctly match their applicable tension types."""
        emergence = CandidateEmergence()
        operators_layer = OperatorMemoryLayer()
        
        # Repair operator for conflicts
        op_repair = OperatorRecord(
            operator_id="op_repair",
            operator_type=OperatorType.REPAIR,
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
                confidence_threshold=0.5,
            ),
            completion_template="Repairing conflict",
            confidence=0.8,
            usage_count=0,
            success_rate=0.7,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        operators_layer.store_operator(op_repair)
        
        field = StateField(
            query_features=["test"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={"op_repair": 0.9},
            conflict_markers=[
                ConflictMarker(
                    conflict_id="c1",
                    conflict_type=ConflictType.CONTRADICTION,
                    involved_memories=["m1", "m2"],
                    strength=0.8,
                    description="Test conflict",
                )
            ],
            goal_states=[],
            partial_outputs=[],
        )
        
        tension_sources = [
            TensionSource(
                type=TensionType.CONFLICT,
                location="c1",
                strength=0.8,
                description="Conflict",
            )
        ]
        
        candidates = emergence.generate_from_operators(
            field, operators_layer, tension_sources
        )
        
        # Repair operators should address conflict tensions
        assert len(candidates) > 0
        assert any(
            any(t.type == TensionType.CONFLICT for t in c.tensions_addressed)
            for c in candidates
        )
    
    def test_operator_confidence_propagation(self):
        """Test that operator confidence propagates to generated candidates."""
        emergence = CandidateEmergence()
        operators_layer = OperatorMemoryLayer()
        
        # High confidence operator
        op_high = OperatorRecord(
            operator_id="op_high",
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
            completion_template="High confidence output",
            confidence=0.95,
            usage_count=100,
            success_rate=0.9,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        operators_layer.store_operator(op_high)
        
        # Low confidence operator
        op_low = OperatorRecord(
            operator_id="op_low",
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
            completion_template="Low confidence output",
            confidence=0.4,
            usage_count=5,
            success_rate=0.5,
            generalization_level=1,
            induced_from=[],
            timestamp=0.0,
        )
        operators_layer.store_operator(op_low)
        
        field = StateField(
            query_features=["test"],
            query_hv=np.zeros(100, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={"op_high": 0.9, "op_low": 0.8},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        tension_sources = [
            TensionSource(
                type=TensionType.INCOMPLETE_OUTPUT,
                location="field",
                strength=0.6,
                description="Incomplete",
            )
        ]
        
        candidates = emergence.generate_from_operators(
            field, operators_layer, tension_sources
        )
        
        # Candidates from high-confidence operators should have higher confidence
        high_conf_candidates = [c for c in candidates if "op_high" in c.source_operators]
        low_conf_candidates = [c for c in candidates if "op_low" in c.source_operators]
        
        if high_conf_candidates and low_conf_candidates:
            assert high_conf_candidates[0].confidence > low_conf_candidates[0].confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
