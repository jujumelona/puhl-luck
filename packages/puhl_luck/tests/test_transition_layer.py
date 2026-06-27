"""
Unit tests for TransitionMemoryLayer (Layer 4).

Tests the implementation of Task 4.1:
- store_transition method
- find_similar_transitions method using HDC similarity
- get_completion_pattern method
"""

import numpy as np
import pytest

from puhl_luck._memory_field_core import (
    ConflictMarker,
    ConflictType,
    GoalState,
    StateField,
)
from puhl_luck._memory_transition_layer import TransitionMemoryLayer


class TestTransitionMemoryLayer:
    """Test suite for TransitionMemoryLayer implementation."""

    def test_initialization(self):
        """Test that TransitionMemoryLayer initializes correctly."""
        layer = TransitionMemoryLayer(hdc_dimensions=10000)
        
        assert layer.hdc_dimensions == 10000
        assert len(layer.transitions) == 0
        assert len(layer.transition_index) == 0
        assert layer.total_transitions_stored == 0
        assert "conversation" in layer.domain_transitions
        assert "code" in layer.domain_transitions
        assert "text" in layer.modality_transitions

    def test_store_transition_basic(self):
        """Test storing a basic partial-to-complete transition."""
        layer = TransitionMemoryLayer(hdc_dimensions=10000)
        
        # Create a partial state (incomplete)
        partial = StateField(
            query_features=["what", "capital", "france"],
            query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
            activated_events={"event_1": 0.8},
            activated_concepts={"geography": 0.7},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="answer_question",
                    satisfaction_level=0.0,
                    constraints=[]
                )
            ],
            partial_outputs=["What is the capital of France?"],
        )
        
        # Create a complete state
        complete = StateField(
            query_features=["what", "capital", "france", "paris", "answer"],
            query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
            activated_events={"event_1": 0.8, "event_2": 0.9},
            activated_concepts={"geography": 0.7, "capital_cities": 0.9},
            activated_operators={},
            conflict_markers=[],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="answer_question",
                    satisfaction_level=1.0,
                    constraints=[]
                )
            ],
            partial_outputs=["What is the capital of France?", "Paris"],
        )
        
        # Store the transition
        trans_id = layer.store_transition(
            partial=partial,
            complete=complete,
            modality="text",
            domain="conversation"
        )
        
        # Verify storage
        assert trans_id is not None
        assert trans_id.startswith("trans_")
        assert trans_id in layer.transitions
        assert layer.total_transitions_stored == 1
        
        # Verify indexing
        assert trans_id in layer.domain_transitions["conversation"]
        assert trans_id in layer.modality_transitions["text"]
        
        # Verify transition content
        transition = layer.transitions[trans_id]
        assert transition.transition_id == trans_id
        assert transition.partial_state == partial
        assert transition.complete_state == complete
        assert transition.modality == "text"
        assert transition.domain == "conversation"
        assert len(transition.completion_features) > 0
        assert "paris" in transition.completion_features or "answer" in transition.completion_features

    def test_find_similar_transitions(self):
        """Test finding similar transitions using HDC similarity."""
        layer = TransitionMemoryLayer(hdc_dimensions=10000)
        
        # Store a few transitions
        for i in range(3):
            partial = StateField(
                query_features=[f"feature_{i}", "common_feature"],
                query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
                activated_events={f"event_{i}": 0.8},
                activated_concepts={f"concept_{i}": 0.7},
                activated_operators={},
                conflict_markers=[],
                goal_states=[],
                partial_outputs=[],
            )
            
            complete = StateField(
                query_features=[f"feature_{i}", "common_feature", f"answer_{i}"],
                query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
                activated_events={f"event_{i}": 0.8, f"event_{i}_complete": 0.9},
                activated_concepts={f"concept_{i}": 0.7},
                activated_operators={},
                conflict_markers=[],
                goal_states=[],
                partial_outputs=[],
            )
            
            layer.store_transition(
                partial=partial,
                complete=complete,
                modality="text",
                domain="conversation"
            )
        
        # Create a query partial state similar to the first one
        query_partial = StateField(
            query_features=["feature_0", "common_feature", "extra"],
            query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
            activated_events={"event_0": 0.9},
            activated_concepts={"concept_0": 0.8},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        # Find similar transitions
        similar = layer.find_similar_transitions(query_partial, top_k=2)
        
        # Verify results
        assert len(similar) > 0
        assert len(similar) <= 2
        
        # Each result should be (trans_id, similarity_score)
        for trans_id, score in similar:
            assert trans_id in layer.transitions
            assert 0.0 <= score <= 1.0

    def test_get_completion_pattern(self):
        """Test extracting completion patterns from transitions."""
        layer = TransitionMemoryLayer(hdc_dimensions=10000)
        
        # Create and store a transition
        partial = StateField(
            query_features=["incomplete"],
            query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
            activated_events={"event_1": 0.8},
            activated_concepts={"concept_a": 0.7},
            activated_operators={},
            conflict_markers=[
                ConflictMarker(
                    conflict_id="c1",
                    conflict_type=ConflictType.CONTRADICTION,
                    involved_memories=["event_1"],
                    strength=0.8,
                    description="test conflict"
                )
            ],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="complete_task",
                    satisfaction_level=0.2,
                    constraints=[]
                )
            ],
            partial_outputs=["partial output"],
        )
        
        complete = StateField(
            query_features=["incomplete", "complete", "answer"],
            query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
            activated_events={"event_1": 0.8, "event_2": 0.9},
            activated_concepts={"concept_a": 0.7, "concept_b": 0.9},
            activated_operators={},
            conflict_markers=[],  # Conflict resolved
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="complete_task",
                    satisfaction_level=1.0,
                    constraints=[]
                )
            ],
            partial_outputs=["partial output", "complete output"],
        )
        
        trans_id = layer.store_transition(
            partial=partial,
            complete=complete,
            modality="text",
            domain="reasoning"
        )
        
        # Get completion pattern
        pattern = layer.get_completion_pattern(trans_id)
        
        # Verify pattern
        assert pattern is not None
        assert len(pattern.added_features) > 0
        assert "complete" in pattern.added_features or "answer" in pattern.added_features
        assert "concept_b" in pattern.added_concepts
        assert pattern.modality == "text"
        assert pattern.domain == "reasoning"
        assert pattern.completion_type in ["direct", "elaboration", "correction", "explanation"]
        assert 0.0 <= pattern.reliability <= 1.0

    def test_domain_filtering(self):
        """Test filtering transitions by domain."""
        layer = TransitionMemoryLayer(hdc_dimensions=10000)
        
        # Store transitions in different domains
        for domain in ["conversation", "code", "reasoning"]:
            partial = StateField(
                query_features=[f"{domain}_feature"],
                query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
                activated_events={},
                activated_concepts={},
                activated_operators={},
                conflict_markers=[],
                goal_states=[],
                partial_outputs=[],
            )
            
            complete = StateField(
                query_features=[f"{domain}_feature", "complete"],
                query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
                activated_events={},
                activated_concepts={},
                activated_operators={},
                conflict_markers=[],
                goal_states=[],
                partial_outputs=[],
            )
            
            layer.store_transition(
                partial=partial,
                complete=complete,
                modality="text",
                domain=domain
            )
        
        # Get transitions by domain
        code_trans = layer.get_transitions_by_domain("code")
        assert len(code_trans) == 1
        
        conversation_trans = layer.get_transitions_by_domain("conversation")
        assert len(conversation_trans) == 1

    def test_relevance_tracking(self):
        """Test that relevance counts are tracked correctly."""
        layer = TransitionMemoryLayer(hdc_dimensions=10000)
        
        # Store a transition
        partial = StateField(
            query_features=["test"],
            query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        complete = StateField(
            query_features=["test", "complete"],
            query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        trans_id = layer.store_transition(
            partial=partial,
            complete=complete,
            modality="text",
            domain="conversation"
        )
        
        # Initially relevance should be 0
        transition = layer.get_transition(trans_id)
        assert transition.relevance_count == 0
        
        # Update relevance
        layer.update_relevance(trans_id, increment=1)
        assert transition.relevance_count == 1
        
        layer.update_relevance(trans_id, increment=5)
        assert transition.relevance_count == 6

    def test_empty_layer_query(self):
        """Test querying an empty layer returns empty results."""
        layer = TransitionMemoryLayer(hdc_dimensions=10000)
        
        query_partial = StateField(
            query_features=["test"],
            query_hv=np.random.randint(0, 2, 10000, dtype=np.int8),
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
        )
        
        similar = layer.find_similar_transitions(query_partial, top_k=5)
        assert len(similar) == 0
