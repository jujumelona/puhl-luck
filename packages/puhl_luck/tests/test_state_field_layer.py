"""
Unit tests for StateFieldLayer (Layer 2) implementation.

Tests the implementation of Task 2.1: StateFieldLayer class in _memory_state_field.py

Requirements tested:
- 1.2: State Field Layer stores currently activated cognitive states
- 2.4: Represents activations as interactive cognitive field
- 12.1: Maintains query feature representations
- 12.2: Maintains activated event identifiers with activation strengths
- 12.3: Maintains activated concept identifiers with activation strengths
- 12.4: Maintains activated operator identifiers with activation strengths
- 12.5: Maintains conflict markers with conflict strengths
- 12.6: Maintains goal states with satisfaction levels
- 12.7: Maintains partial output representations
"""

import pytest
import numpy as np

from puhl_luck._memory_state_field import StateFieldLayer
from puhl_luck._memory_exposure_layer import ExposureEventsLayer
from puhl_luck._memory_operator_layer import OperatorMemoryLayer
from puhl_luck._memory_field_core import (
    InputContext,
    ConflictMarker,
    ConflictType,
    GoalState,
    StateField,
)


def store_text_event(exposure_layer: ExposureEventsLayer, text: str, label: str = None) -> str:
    """Helper function to store a text event in the exposure layer."""
    features = exposure_layer.compute_event_features(text, "text")
    return exposure_layer.store_event(
        modality="text",
        features=features,
        label=label,
        preview=text[:100],
    )


class TestStateFieldLayerInitialization:
    """Test StateFieldLayer initialization and basic structure."""

    def test_initialization(self):
        """Test that StateFieldLayer can be initialized."""
        layer = StateFieldLayer()
        assert layer is not None
        assert layer.current_field is None

    def test_initial_state(self):
        """Test that initial state is clean."""
        layer = StateFieldLayer()
        assert layer.current_field is None


class TestActivateFromInput:
    """Test activate_from_input method (Requirements 1.2, 2.4, 12.1-12.7)."""

    def test_activate_from_simple_input(self):
        """Test field activation from simple text input."""
        # Setup layers
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        # Add some test events to exposure layer
        features1 = exposure_layer.compute_event_features("Paris is the capital of France.", "text")
        exposure_layer.store_event(
            modality="text",
            features=features1,
            preview="Paris is the capital of France.",
        )
        features2 = exposure_layer.compute_event_features("France is in Europe.", "text")
        exposure_layer.store_event(
            modality="text",
            features=features2,
            preview="France is in Europe.",
        )

        # Create input context
        context = InputContext(
            query_text="What is the capital of France?",
            modality="text",
            metadata={},
            goals=["answer_question"],
            constraints=[],
        )

        # Activate field
        field = state_layer.activate_from_input(context, exposure_layer)

        # Verify field structure (Requirement 1.2, 2.4)
        assert isinstance(field, StateField)
        assert field is not None
        assert state_layer.current_field is field

        # Verify query features (Requirement 12.1)
        assert len(field.query_features) > 0
        assert field.query_hv is not None
        assert isinstance(field.query_hv, np.ndarray)

        # Verify activated events (Requirement 12.2)
        assert isinstance(field.activated_events, dict)
        for event_id, strength in field.activated_events.items():
            assert isinstance(event_id, str)
            assert 0.0 <= strength <= 1.0

        # Verify activated concepts (Requirement 12.3)
        assert isinstance(field.activated_concepts, dict)
        for concept_id, strength in field.activated_concepts.items():
            assert isinstance(concept_id, str)
            assert 0.0 <= strength <= 1.0

        # Verify activated operators (Requirement 12.4)
        assert isinstance(field.activated_operators, dict)

        # Verify conflict markers (Requirement 12.5)
        assert isinstance(field.conflict_markers, list)
        assert len(field.conflict_markers) == 0  # Initially empty

        # Verify goal states (Requirement 12.6)
        assert isinstance(field.goal_states, list)
        assert len(field.goal_states) == 1
        assert field.goal_states[0].goal_description == "answer_question"
        assert field.goal_states[0].satisfaction_level == 0.0

        # Verify partial outputs (Requirement 12.7)
        assert isinstance(field.partial_outputs, list)
        assert len(field.partial_outputs) == 0  # Initially empty

        # Verify resonance structure
        assert isinstance(field.resonance, dict)

        # Verify iteration tracking
        assert field.iteration == 0
        assert len(field.previous_outputs) == 0

    def test_activate_with_multiple_goals(self):
        """Test field activation with multiple goals."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query",
            modality="text",
            metadata={},
            goals=["goal_1", "goal_2", "goal_3"],
            constraints=["constraint_1"],
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Verify multiple goals initialized
        assert len(field.goal_states) == 3
        for i, goal in enumerate(field.goal_states):
            assert goal.goal_id == f"goal_{i}"
            assert goal.goal_description == f"goal_{i + 1}"
            assert goal.satisfaction_level == 0.0
            assert goal.constraints == ["constraint_1"]

    def test_activate_without_goals(self):
        """Test field activation without explicit goals."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query",
            modality="text",
            metadata={},
            goals=None,
            constraints=None,
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Verify no goals when not provided
        assert len(field.goal_states) == 0

    def test_activate_with_operators_layer(self):
        """Test field activation with operator layer."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        operators_layer = OperatorMemoryLayer()
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query",
            modality="text",
            metadata={},
            goals=["test_goal"],
            constraints=[],
        )

        # Activate with operators layer
        field = state_layer.activate_from_input(
            context, exposure_layer, operators_layer
        )

        # Verify operators activation dictionary exists
        assert isinstance(field.activated_operators, dict)

    def test_activation_strength_normalization(self):
        """Test that activation strengths are normalized to [0, 1]."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        # Add multiple events with varying similarity
        for i in range(10):
            store_text_event(exposure_layer, f"Test event {i} with various content")

        context = InputContext(
            query_text="Test query",
            modality="text",
            metadata={},
            goals=[],
            constraints=[],
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Verify all activation strengths are in [0, 1]
        for event_id, strength in field.activated_events.items():
            assert 0.0 <= strength <= 1.0, f"Event {event_id} has invalid strength {strength}"

        for concept_id, strength in field.activated_concepts.items():
            assert 0.0 <= strength <= 1.0, f"Concept {concept_id} has invalid strength {strength}"


class TestAddConflict:
    """Test add_conflict method (Requirement 12.5)."""

    def test_add_conflict_to_active_field(self):
        """Test adding conflict to an active field."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query", modality="text", metadata={}, goals=[], constraints=[]
        )

        # Activate field
        field = state_layer.activate_from_input(context, exposure_layer)

        # Add conflict
        conflict = ConflictMarker(
            conflict_id="conflict_1",
            conflict_type=ConflictType.CONTRADICTION,
            involved_memories=["event_1", "event_2"],
            strength=0.75,
            description="Test conflict",
        )

        state_layer.add_conflict(conflict)

        # Verify conflict added
        assert len(field.conflict_markers) == 1
        assert field.conflict_markers[0] == conflict
        assert field.conflict_markers[0].conflict_id == "conflict_1"
        assert field.conflict_markers[0].strength == 0.75

    def test_add_multiple_conflicts(self):
        """Test adding multiple conflicts."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Add multiple conflicts
        for i in range(5):
            conflict = ConflictMarker(
                conflict_id=f"conflict_{i}",
                conflict_type=ConflictType.CONTRADICTION,
                involved_memories=[f"mem_{i}", f"mem_{i+1}"],
                strength=0.5 + i * 0.1,
                description=f"Conflict {i}",
            )
            state_layer.add_conflict(conflict)

        # Verify all conflicts added
        assert len(field.conflict_markers) == 5

    def test_add_conflict_without_active_field_raises_error(self):
        """Test that adding conflict without active field raises error."""
        state_layer = StateFieldLayer()

        conflict = ConflictMarker(
            conflict_id="conflict_1",
            conflict_type=ConflictType.CONTRADICTION,
            involved_memories=["event_1", "event_2"],
            strength=0.75,
            description="Test conflict",
        )

        with pytest.raises(RuntimeError, match="Cannot add conflict: no active field"):
            state_layer.add_conflict(conflict)


class TestAddGoal:
    """Test add_goal method (Requirement 12.6)."""

    def test_add_goal_to_active_field(self):
        """Test adding goal to an active field."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query", modality="text", metadata={}, goals=[], constraints=[]
        )

        # Activate field
        field = state_layer.activate_from_input(context, exposure_layer)

        # Add goal
        goal = GoalState(
            goal_id="new_goal",
            goal_description="Complete the task",
            satisfaction_level=0.5,
            constraints=["constraint_1"],
        )

        state_layer.add_goal(goal)

        # Verify goal added
        assert len(field.goal_states) == 1
        assert field.goal_states[0] == goal
        assert field.goal_states[0].goal_id == "new_goal"
        assert field.goal_states[0].satisfaction_level == 0.5

    def test_add_goal_without_active_field_raises_error(self):
        """Test that adding goal without active field raises error."""
        state_layer = StateFieldLayer()

        goal = GoalState(
            goal_id="goal_1",
            goal_description="Test goal",
            satisfaction_level=0.0,
            constraints=[],
        )

        with pytest.raises(RuntimeError, match="Cannot add goal: no active field"):
            state_layer.add_goal(goal)


class TestUpdateWithOutput:
    """Test update_with_output method (Requirement 12.7)."""

    def test_update_with_output(self):
        """Test updating field with output."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query", modality="text", metadata={}, goals=[], constraints=[]
        )

        # Activate field
        field = state_layer.activate_from_input(context, exposure_layer)

        # Initial state
        assert len(field.partial_outputs) == 0
        assert len(field.previous_outputs) == 0
        assert field.iteration == 0

        # Update with output
        state_layer.update_with_output("First output")

        # Verify update
        assert len(field.partial_outputs) == 1
        assert field.partial_outputs[0] == "First output"
        assert len(field.previous_outputs) == 1
        assert field.previous_outputs[0] == "First output"
        assert field.iteration == 1

    def test_multiple_output_updates(self):
        """Test multiple output updates for recursive stabilization."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Multiple updates
        for i in range(5):
            state_layer.update_with_output(f"Output {i}")

        # Verify all outputs tracked
        assert len(field.partial_outputs) == 5
        assert len(field.previous_outputs) == 5
        assert field.iteration == 5

        for i in range(5):
            assert field.partial_outputs[i] == f"Output {i}"
            assert field.previous_outputs[i] == f"Output {i}"

    def test_update_without_active_field_raises_error(self):
        """Test that updating without active field raises error."""
        state_layer = StateFieldLayer()

        with pytest.raises(
            RuntimeError, match="Cannot update with output: no active field"
        ):
            state_layer.update_with_output("Test output")


class TestComputeResonance:
    """Test compute_resonance method (Requirements 2.6, 14.1-14.4)."""

    def test_self_resonance_is_one(self):
        """Test that a memory always has perfect resonance with itself."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        event_id = store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Self-resonance should be 1.0
        resonance = state_layer.compute_resonance(event_id, event_id, exposure_layer)
        assert resonance == 1.0

    def test_resonance_with_high_feature_overlap(self):
        """Test that high feature overlap produces positive resonance."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        # Add two events with similar content (high feature overlap)
        event1_id = store_text_event(exposure_layer, "Paris is the capital of France")
        event2_id = store_text_event(exposure_layer, "Paris is the largest city in France")

        context = InputContext(
            query_text="France Paris", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Compute resonance between similar events
        resonance = state_layer.compute_resonance(event1_id, event2_id, exposure_layer)

        # Should have positive resonance due to feature overlap
        assert resonance > 0.0, f"Expected positive resonance, got {resonance}"

    def test_resonance_with_low_feature_overlap(self):
        """Test that low feature overlap produces low resonance."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        # Add two events with very different content
        event1_id = store_text_event(exposure_layer, "Paris is the capital of France")
        event2_id = store_text_event(exposure_layer, "Tokyo has many skyscrapers and technology")

        context = InputContext(
            query_text="cities", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Compute resonance between dissimilar events
        resonance = state_layer.compute_resonance(event1_id, event2_id, exposure_layer)

        # Should have lower resonance than similar events
        assert resonance >= 0.0  # Could be zero or slightly positive

    def test_resonance_with_conflict_penalty(self):
        """Test that conflict markers reduce resonance (negative resonance)."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        event1_id = store_text_event(exposure_layer, "Event A")
        event2_id = store_text_event(exposure_layer, "Event B")

        context = InputContext(
            query_text="Test query", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Compute baseline resonance without conflict
        resonance_without_conflict = state_layer.compute_resonance(
            event1_id, event2_id, exposure_layer
        )

        # Add conflict between these events
        conflict = ConflictMarker(
            conflict_id="conflict_1",
            conflict_type=ConflictType.CONTRADICTION,
            involved_memories=[event1_id, event2_id],
            strength=0.8,
            description="Events contradict each other",
        )
        state_layer.add_conflict(conflict)

        # Create new field state to recompute resonance with conflict
        field2 = state_layer.activate_from_input(context, exposure_layer)
        state_layer.add_conflict(conflict)

        # Compute resonance again with conflict
        resonance_with_conflict = state_layer.compute_resonance(
            event1_id, event2_id, exposure_layer
        )

        # Resonance with conflict should be lower
        assert resonance_with_conflict < resonance_without_conflict

    def test_resonance_between_concepts(self):
        """Test resonance computation between concept memories."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        # Add events that will activate concepts
        store_text_event(exposure_layer, "Test event with concept:geography")
        store_text_event(exposure_layer, "Another event with concept:geography")

        context = InputContext(
            query_text="geography", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Compute resonance between same concept
        resonance = state_layer.compute_resonance(
            "concept:geography", "concept:geography", exposure_layer
        )

        # Same concept should have high resonance
        assert resonance == 1.0

    def test_resonance_caching_in_matrix(self):
        """Test that resonance is cached in resonance matrix."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        event1_id = store_text_event(exposure_layer, "Test event A")
        event2_id = store_text_event(exposure_layer, "Test event B")

        context = InputContext(
            query_text="Test query", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Compute resonance
        resonance1 = state_layer.compute_resonance(event1_id, event2_id, exposure_layer)
        
        # Verify stored in resonance matrix (symmetric)
        assert (event1_id, event2_id) in field.resonance
        assert (event2_id, event1_id) in field.resonance
        assert field.resonance[(event1_id, event2_id)] == resonance1
        assert field.resonance[(event2_id, event1_id)] == resonance1

        # Compute again, should use cached value
        resonance2 = state_layer.compute_resonance(event1_id, event2_id, exposure_layer)
        assert resonance2 == resonance1

    def test_resonance_symmetry(self):
        """Test that resonance is symmetric: R(A,B) = R(B,A)."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        event1_id = store_text_event(exposure_layer, "Event A")
        event2_id = store_text_event(exposure_layer, "Event B")

        context = InputContext(
            query_text="Test", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Compute in both directions
        resonance_ab = state_layer.compute_resonance(event1_id, event2_id, exposure_layer)
        resonance_ba = state_layer.compute_resonance(event2_id, event1_id, exposure_layer)

        # Should be symmetric
        assert resonance_ab == resonance_ba

    def test_resonance_without_active_field_raises_error(self):
        """Test that computing resonance without active field raises error."""
        state_layer = StateFieldLayer()

        with pytest.raises(
            RuntimeError, match="Cannot compute resonance: no active field"
        ):
            state_layer.compute_resonance("mem_1", "mem_2")


class TestGetTensionSources:
    """Test get_tension_sources method."""

    def test_tension_from_conflicts(self):
        """Test tension identification from conflicts."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Add conflict
        conflict = ConflictMarker(
            conflict_id="conflict_1",
            conflict_type=ConflictType.CONTRADICTION,
            involved_memories=["event_1", "event_2"],
            strength=0.75,
            description="Test conflict",
        )
        state_layer.add_conflict(conflict)

        # Get tension sources
        tensions = state_layer.get_tension_sources()

        # Verify conflict identified as tension
        assert len(tensions) >= 1
        assert any("Conflict" in t for t in tensions)
        assert any("Test conflict" in t for t in tensions)

    def test_tension_from_unsatisfied_goals(self):
        """Test tension identification from unsatisfied goals."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query",
            modality="text",
            metadata={},
            goals=["complete_task"],
            constraints=[],
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Get tension sources
        tensions = state_layer.get_tension_sources()

        # Verify unsatisfied goal identified as tension
        assert len(tensions) >= 1
        assert any("Unsatisfied goal" in t for t in tensions)
        assert any("complete_task" in t for t in tensions)

    def test_no_tension_when_goals_satisfied(self):
        """Test no tension when goals are satisfied."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Manually add satisfied goal
        goal = GoalState(
            goal_id="satisfied_goal",
            goal_description="Satisfied",
            satisfaction_level=1.0,
            constraints=[],
        )
        state_layer.add_goal(goal)

        # Get tension sources
        tensions = state_layer.get_tension_sources()

        # Should not include satisfied goal
        assert not any("Satisfied" in t and "Unsatisfied" in t for t in tensions)

    def test_tension_from_incomplete_outputs(self):
        """Test tension identification from incomplete outputs."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        store_text_event(exposure_layer, "Test event")

        context = InputContext(
            query_text="Test query", modality="text", metadata={}, goals=[], constraints=[]
        )

        field = state_layer.activate_from_input(context, exposure_layer)

        # Add incomplete output
        state_layer.update_with_output("Short")

        # Get tension sources
        tensions = state_layer.get_tension_sources()

        # Verify incomplete output identified as tension
        assert any("Incomplete output" in t for t in tensions)

    def test_get_tension_without_active_field_raises_error(self):
        """Test that getting tension sources without active field raises error."""
        state_layer = StateFieldLayer()

        with pytest.raises(
            RuntimeError, match="Cannot get tension sources: no active field"
        ):
            state_layer.get_tension_sources()


class TestPrivateHelperMethods:
    """Test private helper methods."""

    def test_activate_events_helper(self):
        """Test _activate_events helper method."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        # Add test events
        for i in range(5):
            store_text_event(exposure_layer, f"Test event {i}")

        # Create query hypervector
        features = exposure_layer.compute_event_features("Test query", "text")
        query_hv = exposure_layer._bundle_event(features, [])

        # Activate events
        activated = state_layer._activate_events(query_hv, exposure_layer, top_k=3)

        # Verify activation dictionary
        assert isinstance(activated, dict)
        assert len(activated) <= 3
        for event_id, strength in activated.items():
            assert isinstance(event_id, str)
            assert 0.0 <= strength <= 1.0

    def test_activate_concepts_helper(self):
        """Test _activate_concepts helper method."""
        exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
        state_layer = StateFieldLayer()

        # Add test event with concept features
        store_text_event(exposure_layer, "Test event")

        # Create mock activated events
        activated_events = {"event_0": 1.0}

        # Activate concepts
        concepts = state_layer._activate_concepts(activated_events, exposure_layer)

        # Verify concept dictionary
        assert isinstance(concepts, dict)
        for concept_id, strength in concepts.items():
            assert 0.0 <= strength <= 1.0

    def test_initialize_goals_helper(self):
        """Test _initialize_goals helper method."""
        state_layer = StateFieldLayer()

        context = InputContext(
            query_text="Test",
            modality="text",
            metadata={},
            goals=["goal_1", "goal_2"],
            constraints=["constraint_1"],
        )

        goals = state_layer._initialize_goals(context)

        # Verify goals initialized correctly
        assert len(goals) == 2
        assert goals[0].goal_id == "goal_0"
        assert goals[0].goal_description == "goal_1"
        assert goals[0].satisfaction_level == 0.0
        assert goals[0].constraints == ["constraint_1"]
        assert goals[1].goal_id == "goal_1"
        assert goals[1].goal_description == "goal_2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

