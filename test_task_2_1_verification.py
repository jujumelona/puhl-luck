"""
Verification test for Task 2.1: StateFieldLayer implementation
"""

import sys
sys.path.insert(0, r'C:\Users\kkk\Desktop\puhl-luck\packages\puhl_luck')

from puhl_luck._memory_state_field import StateFieldLayer
from puhl_luck._memory_exposure_layer import ExposureEventsLayer
from puhl_luck._memory_field_core import (
    InputContext,
    ConflictMarker,
    ConflictType,
    GoalState,
)


def test_state_field_layer_initialization():
    """Test that StateFieldLayer can be initialized."""
    layer = StateFieldLayer()
    assert layer.current_field is None
    print("✓ StateFieldLayer initialization works")


def test_activate_from_input():
    """Test that activate_from_input creates a field with activations."""
    # Create exposure layer and add some events
    exposure_layer = ExposureEventsLayer(window_size=12, decay=0.72)
    
    # Store a couple of events
    event1_id = exposure_layer.store_event(
        modality="text",
        features=["france", "paris", "capital", "europe"],
        source="test",
        preview="France is a country in Europe. Paris is the capital."
    )
    
    event2_id = exposure_layer.store_event(
        modality="text",
        features=["germany", "berlin", "capital", "europe"],
        source="test",
        preview="Germany is a country in Europe. Berlin is the capital."
    )
    
    # Create state field layer
    state_layer = StateFieldLayer()
    
    # Create input context
    input_ctx = InputContext(
        text="What is the capital of France?",
        modality="text",
        metadata={},
        goals=["answer_question"]
    )
    
    # Activate field from input
    field = state_layer.activate_from_input(input_ctx, exposure_layer)
    
    # Verify field structure
    assert field is not None
    assert len(field.query_features) > 0
    assert field.query_hv is not None
    assert isinstance(field.activated_events, dict)
    assert isinstance(field.activated_concepts, dict)
    assert isinstance(field.activated_operators, dict)
    assert isinstance(field.conflict_markers, list)
    assert isinstance(field.goal_states, list)
    assert len(field.goal_states) == 1  # We added one goal
    assert field.goal_states[0].goal_description == "answer_question"
    assert field.iteration == 0
    
    print("✓ activate_from_input creates proper field structure")
    print(f"  - Activated {len(field.activated_events)} events")
    print(f"  - Extracted {len(field.query_features)} query features")
    print(f"  - Initialized {len(field.goal_states)} goal(s)")


def test_add_conflict():
    """Test that add_conflict adds conflicts to the field."""
    exposure_layer = ExposureEventsLayer()
    state_layer = StateFieldLayer()
    
    # Create a field first
    input_ctx = InputContext(
        text="Test input",
        modality="text",
        metadata={}
    )
    field = state_layer.activate_from_input(input_ctx, exposure_layer)
    
    # Add a conflict
    conflict = ConflictMarker(
        conflict_id="c1",
        conflict_type=ConflictType.CONTRADICTION,
        involved_memories=["mem1", "mem2"],
        strength=0.8,
        description="Test conflict"
    )
    state_layer.add_conflict(conflict)
    
    # Verify conflict was added
    assert len(state_layer.current_field.conflict_markers) == 1
    assert state_layer.current_field.conflict_markers[0].conflict_id == "c1"
    
    print("✓ add_conflict adds conflicts to field")


def test_add_goal():
    """Test that add_goal adds goals to the field."""
    exposure_layer = ExposureEventsLayer()
    state_layer = StateFieldLayer()
    
    # Create a field first
    input_ctx = InputContext(
        text="Test input",
        modality="text",
        metadata={}
    )
    field = state_layer.activate_from_input(input_ctx, exposure_layer)
    
    initial_goals = len(field.goal_states)
    
    # Add a goal
    goal = GoalState(
        goal_id="g1",
        goal_description="test_goal",
        satisfaction_level=0.5,
        constraints=[]
    )
    state_layer.add_goal(goal)
    
    # Verify goal was added
    assert len(state_layer.current_field.goal_states) == initial_goals + 1
    assert state_layer.current_field.goal_states[-1].goal_id == "g1"
    
    print("✓ add_goal adds goals to field")


def test_update_with_output():
    """Test that update_with_output updates field state."""
    exposure_layer = ExposureEventsLayer()
    state_layer = StateFieldLayer()
    
    # Create a field first
    input_ctx = InputContext(
        text="Test input",
        modality="text",
        metadata={}
    )
    field = state_layer.activate_from_input(input_ctx, exposure_layer)
    
    assert field.iteration == 0
    assert len(field.partial_outputs) == 0
    assert len(field.previous_outputs) == 0
    
    # Update with output
    state_layer.update_with_output("First output")
    
    # Verify updates
    assert state_layer.current_field.iteration == 1
    assert len(state_layer.current_field.partial_outputs) == 1
    assert len(state_layer.current_field.previous_outputs) == 1
    assert state_layer.current_field.partial_outputs[0] == "First output"
    
    # Update again
    state_layer.update_with_output("Second output")
    
    assert state_layer.current_field.iteration == 2
    assert len(state_layer.current_field.partial_outputs) == 2
    assert len(state_layer.current_field.previous_outputs) == 2
    
    print("✓ update_with_output updates field state correctly")


def test_get_tension_sources():
    """Test that get_tension_sources identifies tensions."""
    exposure_layer = ExposureEventsLayer()
    state_layer = StateFieldLayer()
    
    # Create a field
    input_ctx = InputContext(
        text="Test input",
        modality="text",
        metadata={},
        goals=["test_goal"]
    )
    field = state_layer.activate_from_input(input_ctx, exposure_layer)
    
    # Should have unsatisfied goal as tension
    tensions = state_layer.get_tension_sources()
    assert len(tensions) > 0
    assert any("goal" in t.lower() for t in tensions)
    
    # Add a conflict
    conflict = ConflictMarker(
        conflict_id="c1",
        conflict_type=ConflictType.CONTRADICTION,
        involved_memories=["mem1", "mem2"],
        strength=0.8,
        description="Test conflict"
    )
    state_layer.add_conflict(conflict)
    
    # Should now include conflict
    tensions = state_layer.get_tension_sources()
    assert any("conflict" in t.lower() for t in tensions)
    
    print("✓ get_tension_sources identifies tensions correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Task 2.1 Verification: StateFieldLayer Implementation")
    print("=" * 60)
    print()
    
    test_state_field_layer_initialization()
    test_activate_from_input()
    test_add_conflict()
    test_add_goal()
    test_update_with_output()
    test_get_tension_sources()
    
    print()
    print("=" * 60)
    print("All Task 2.1 verification tests passed! ✓")
    print("=" * 60)
