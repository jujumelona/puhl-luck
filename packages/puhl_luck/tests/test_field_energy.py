"""
Unit tests for field energy computation (Task 2.4).

Tests the FreeEnergyMinimization class that computes field energy
from cognitive field states.

Requirements tested: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 17.1-17.8
"""

import numpy as np
import pytest

from puhl_luck._memory_field_core import (
    ConflictMarker,
    ConflictType,
    FieldEnergy,
    GoalState,
    StateField,
)
from puhl_luck._memory_field_energy import FreeEnergyMinimization


class TestFreeEnergyMinimization:
    """Test suite for FreeEnergyMinimization class."""

    def test_initialization(self):
        """Test that FreeEnergyMinimization initializes correctly."""
        fem = FreeEnergyMinimization()
        
        # Check default weights exist for all sources
        assert "memory_resonance" in fem.evidence_weights
        assert "goal_satisfaction" in fem.evidence_weights
        assert "contradiction" in fem.conflict_weights
        assert "constraint_violation" in fem.conflict_weights
    
    def test_compute_field_energy_empty_field(self):
        """Test energy computation for an empty field (Requirement 7.1)."""
        fem = FreeEnergyMinimization()
        
        # Create minimal empty field
        field = StateField(
            query_features=["test"],
            query_hv=np.random.randn(10000),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        energy = fem.compute_field_energy(field)
        
        # Verify energy structure
        assert isinstance(energy, FieldEnergy)
        assert isinstance(energy.total, float)
        assert isinstance(energy.evidence, float)
        assert isinstance(energy.conflicts, float)
        assert energy.total == energy.conflicts - energy.evidence
        assert isinstance(energy.evidence_breakdown, dict)
        assert isinstance(energy.conflict_breakdown, dict)
    
    def test_memory_resonance_evidence(self):
        """Test evidence from memory resonance (Requirements 7.2, 17.3)."""
        fem = FreeEnergyMinimization()
        
        # Create field with positive resonance
        field = StateField(
            query_features=["test"],
            query_hv=np.random.randn(10000),
            activated_events={"e1": 0.8, "e2": 0.7},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
            resonance={("e1", "e2"): 0.9, ("e2", "e1"): 0.9},
        )
        
        energy = fem.compute_field_energy(field)
        
        # Positive resonance should contribute to evidence
        assert energy.evidence_breakdown["memory_resonance"] > 0.0
        assert energy.evidence > 0.0
    
    def test_goal_satisfaction_evidence(self):
        """Test evidence from goal satisfaction (Requirements 7.3, 17.4)."""
        fem = FreeEnergyMinimization()
        
        # Create field with satisfied goal
        field = StateField(
            query_features=["test"],
            query_hv=np.random.randn(10000),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="answer_question",
                    satisfaction_level=0.8,
                    constraints=[],
                )
            ],
            partial_outputs=[],
        )
        
        energy = fem.compute_field_energy(field)
        
        # Satisfied goal should contribute to evidence
        assert energy.evidence_breakdown["goal_satisfaction"] > 0.5
    
    def test_memory_support_evidence(self):
        """Test evidence from memory support (Requirement 7.4)."""
        fem = FreeEnergyMinimization()
        
        # Create field with many activated memories
        field = StateField(
            query_features=["test"],
            query_hv=np.random.randn(10000),
            activated_events={"e1": 0.9, "e2": 0.8, "e3": 0.7},
            activated_concepts={"c1": 0.85, "c2": 0.75},
            activated_operators={"op1": 0.9},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        energy = fem.compute_field_energy(field)
        
        # Many strong activations should contribute to memory support
        assert energy.evidence_breakdown["memory_support"] > 0.0
    
    def test_contradiction_conflicts(self):
        """Test conflicts from contradictions (Requirements 7.5, 17.6)."""
        fem = FreeEnergyMinimization()
        
        # Create field with contradiction
        field = StateField(
            query_features=["test"],
            query_hv=np.random.randn(10000),
            activated_events={"e1": 0.8, "e2": 0.7},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[
                ConflictMarker(
                    conflict_id="c1",
                    conflict_type=ConflictType.CONTRADICTION,
                    involved_memories=["e1", "e2"],
                    strength=0.8,
                    description="Events contradict",
                )
            ],
            goal_states=[],
            partial_outputs=[],
        )
        
        energy = fem.compute_field_energy(field)
        
        # Contradiction should contribute to conflicts
        assert energy.conflict_breakdown["contradiction"] > 0.0
        assert energy.conflicts > 0.0
    
    def test_constraint_violation_conflicts(self):
        """Test conflicts from constraint violations (Requirements 7.6, 17.7)."""
        fem = FreeEnergyMinimization()
        
        # Create field with constraint violation
        field = StateField(
            query_features=["test"],
            query_hv=np.random.randn(10000),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[
                ConflictMarker(
                    conflict_id="c1",
                    conflict_type=ConflictType.CONSTRAINT_VIOLATION,
                    involved_memories=[],
                    strength=0.7,
                    description="Constraint violated",
                )
            ],
            goal_states=[],
            partial_outputs=[],
        )
        
        energy = fem.compute_field_energy(field)
        
        # Constraint violation should contribute to conflicts
        assert energy.conflict_breakdown["constraint_violation"] > 0.0
    
    def test_repetition_conflicts(self):
        """Test conflicts from repetition (Requirements 7.7, 17.8)."""
        fem = FreeEnergyMinimization()
        
        # Create field with repetition in outputs
        field = StateField(
            query_features=["test"],
            query_hv=np.random.randn(10000),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
            previous_outputs=["test output", "test output"],  # Identical outputs
        )
        
        energy = fem.compute_field_energy(field)
        
        # Repetition should contribute to conflicts
        assert energy.conflict_breakdown["repetition"] > 0.0
    
    def test_incompleteness_conflicts(self):
        """Test conflicts from incompleteness (implied by Requirements 3.2, 13.1)."""
        fem = FreeEnergyMinimization()
        
        # Create field with unsatisfied goal (indicates incompleteness)
        field = StateField(
            query_features=["test"],
            query_hv=np.random.randn(10000),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="answer_question",
                    satisfaction_level=0.2,  # Low satisfaction = incomplete
                    constraints=[],
                )
            ],
            partial_outputs=["short"],  # Short partial output
        )
        
        energy = fem.compute_field_energy(field)
        
        # Incompleteness should contribute to conflicts
        assert energy.conflict_breakdown["incompleteness"] > 0.0
    
    def test_energy_formula_consistency(self):
        """Test that energy = conflicts - evidence (Requirement 7.1)."""
        fem = FreeEnergyMinimization()
        
        # Create field with both evidence and conflicts
        field = StateField(
            query_features=["test"],
            query_hv=np.random.randn(10000),
            activated_events={"e1": 0.8},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[
                ConflictMarker(
                    conflict_id="c1",
                    conflict_type=ConflictType.CONTRADICTION,
                    involved_memories=["e1"],
                    strength=0.5,
                    description="Test conflict",
                )
            ],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="test",
                    satisfaction_level=0.7,
                    constraints=[],
                )
            ],
            partial_outputs=[],
        )
        
        energy = fem.compute_field_energy(field)
        
        # Verify energy formula
        expected_total = energy.conflicts - energy.evidence
        assert abs(energy.total - expected_total) < 1e-6
    
    def test_predict_energy_after_update(self):
        """Test energy prediction for candidates."""
        fem = FreeEnergyMinimization()
        
        # Create field with unsatisfied goal
        field = StateField(
            query_features=["question", "answer"],
            query_hv=np.random.randn(10000),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="answer question",
                    satisfaction_level=0.3,
                    constraints=[],
                )
            ],
            partial_outputs=[],
        )
        
        current_energy = fem.compute_field_energy(field).total
        
        # Candidate that addresses the goal
        candidate = "Here is the answer to your question."
        predicted_energy = fem.predict_energy_after_update(field, candidate)
        
        # Energy should be predicted to decrease
        assert predicted_energy < current_energy
    
    def test_energy_decomposition(self):
        """Test detailed energy decomposition (Requirement 17.1, 17.2)."""
        fem = FreeEnergyMinimization()
        
        # Create complex field
        field = StateField(
            query_features=["test"],
            query_hv=np.random.randn(10000),
            activated_events={"e1": 0.8, "e2": 0.7},
            activated_concepts={"c1": 0.9},
            activated_operators={"op1": 0.85},
            conflict_markers=[
                ConflictMarker(
                    conflict_id="c1",
                    conflict_type=ConflictType.CONTRADICTION,
                    involved_memories=["e1", "e2"],
                    strength=0.5,
                    description="Test",
                )
            ],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="test",
                    satisfaction_level=0.8,
                    constraints=[],
                )
            ],
            partial_outputs=[],
            resonance={("e1", "e2"): 0.6},
        )
        
        energy = fem.compute_field_energy(field)
        
        # Verify all breakdowns are populated
        assert len(energy.evidence_breakdown) > 0
        assert len(energy.conflict_breakdown) > 0
        assert len(energy.dominant_evidence_sources) > 0
        assert len(energy.dominant_conflict_sources) > 0
        assert 0.0 <= energy.tension_level <= 1.0
    
    def test_custom_weights(self):
        """Test custom evidence and conflict weights."""
        # Emphasize memory resonance evidence
        custom_evidence_weights = {
            "memory_resonance": 2.0,
            "goal_satisfaction": 1.0,
            "constraint_satisfaction": 1.0,
            "memory_support": 1.0,
            "coherence": 1.0,
        }
        
        fem = FreeEnergyMinimization(evidence_weights=custom_evidence_weights)
        
        field = StateField(
            query_features=["test"],
            query_hv=np.random.randn(10000),
            activated_events={"e1": 0.8, "e2": 0.7},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
            resonance={("e1", "e2"): 0.9},
        )
        
        energy = fem.compute_field_energy(field)
        
        # Memory resonance should have higher impact
        assert energy.evidence_breakdown["memory_resonance"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
