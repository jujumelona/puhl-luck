"""
Field Formation Process

Creates the cognitive field from input through simultaneous activation of multiple
memory layers. This is the core mechanism that transforms input into an interactive
state field rather than a sequential search process.

The field formation process:
1. Extracts features from input context
2. Activates relevant events from Layer 1 (Exposure Events)
3. Activates relevant concepts from concept memory
4. Activates relevant operators from Layer 3 (Operator Memory)
5. Computes initial resonance patterns between activated memories
6. Creates the StateField representing the living cognitive state

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 14.5, 14.6
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from ._memory_field_core import (
    InputContext,
    StateField,
)

if TYPE_CHECKING:
    from ._memory_exposure_layer import ExposureEventsLayer
    from ._memory_operator_layer import OperatorMemoryLayer


class FieldFormation:
    """
    Field formation process for creating cognitive fields from input.
    
    Implements simultaneous activation across multiple memory layers to create
    an interactive state field. Unlike sequential search-then-rank approaches,
    this creates a living cognitive state where memories interact dynamically.
    
    Key principles:
    - Simultaneous activation: All layers are activated at once, not sequentially
    - Interactive field: Activated memories interact through resonance
    - Context sensitivity: Previous field state influences new activation
    - Multi-layer integration: Events, concepts, and operators all contribute
    
    Requirements:
    - 2.1: Input simultaneously activates multiple memory layers
    - 2.2: Activate relevant events simultaneously
    - 2.3: Activate relevant concepts simultaneously
    - 2.4: Create interactive cognitive field
    - 2.5: Maintain activation strengths
    - 2.6: Compute resonance between activated memories
    - 2.7: Allow memories to interact creating emergent patterns
    """
    
    def __init__(self):
        """Initialize the field formation process."""
        pass
    
    def form_field(
        self,
        input_context: InputContext,
        events_layer: ExposureEventsLayer,
        operators_layer: Optional[OperatorMemoryLayer] = None,
        previous_field: Optional[StateField] = None,
    ) -> StateField:
        """
        Form a cognitive field from input context.
        
        This is the main method that creates the StateField by simultaneously
        activating all memory layers and computing initial interaction patterns.
        
        The process:
        1. Extract features from input text
        2. Create query hypervector for HDC matching
        3. Activate events from Layer 1 using HDC similarity
        4. Activate concepts from activated events and query
        5. Activate operators from Layer 3 (if available) using pattern matching
        6. Initialize goals and constraints from input context
        7. Compute initial resonance patterns between activated memories
        8. Return the complete StateField
        
        If a previous_field is provided (during recursive stabilization), the
        activation process can leverage context from the previous iteration.
        
        Args:
            input_context: Input containing text, modality, goals, constraints
            events_layer: Layer 1 providing access to stored events
            operators_layer: Layer 3 providing access to operators (optional)
            previous_field: Previous field state for recursive updates (optional)
            
        Returns:
            StateField representing the activated cognitive state
            
        Requirements:
        - 2.1: Simultaneously activate multiple memory layers
        - 2.2: Activate relevant events from Exposure Events Layer
        - 2.3: Activate relevant concepts from concept memory
        - 2.4: Create interactive cognitive field, not ranked list
        - 2.5: Maintain activation strengths for all memories
        - 6.2: Recompute activations based on new state during recursion
        """
        # Step 1: Extract features from input text
        features = events_layer.compute_event_features(
            input_context.query_text,
            input_context.modality
        )
        
        # Step 2: Create query hypervector for HDC-based matching
        query_hv = events_layer._bundle_event(features, [])
        
        # Step 3: Activate events from Layer 1
        # This uses HDC similarity to find relevant events
        activated_events = self.activate_events(
            query_features=features,
            query_hv=query_hv,
            events_layer=events_layer,
            previous_field=previous_field,
        )
        
        # Step 4: Activate concepts from activated events
        # Concepts are extracted from event features and aggregated
        activated_concepts = self.activate_concepts(
            query_features=features,
            activated_events=activated_events,
            events_layer=events_layer,
        )
        
        # Step 5: Initialize empty operator activation
        # We need to create a preliminary field first to match operators
        activated_operators: Dict[str, float] = {}
        
        # Step 6: Initialize goals from input context
        from ._memory_field_core import GoalState
        
        goal_states = []
        if input_context.goals:
            for i, goal_desc in enumerate(input_context.goals):
                goal = GoalState(
                    goal_id=f"goal_{i}",
                    goal_description=goal_desc,
                    satisfaction_level=0.0,  # Initially unsatisfied
                    constraints=input_context.constraints or [],
                )
                goal_states.append(goal)
        
        # Create preliminary field for operator matching
        # Note: iteration is incremented in the final field, not preliminary
        preliminary_field = StateField(
            query_features=features,
            query_hv=query_hv,
            activated_events=activated_events,
            activated_concepts=activated_concepts,
            activated_operators={},  # Empty initially
            conflict_markers=[],
            goal_states=goal_states,
            partial_outputs=[],
            resonance={},
            field_energy=None,
            previous_outputs=previous_field.previous_outputs if previous_field else [],
            iteration=previous_field.iteration if previous_field else 0,
        )
        
        # Step 7: Activate operators from Layer 3 (if available)
        if operators_layer is not None:
            activated_operators = self.activate_operators(
                field_state=preliminary_field,
                operators_layer=operators_layer,
            )
        
        # Step 8: Create the final StateField with all activations
        field = StateField(
            query_features=features,
            query_hv=query_hv,
            activated_events=activated_events,
            activated_concepts=activated_concepts,
            activated_operators=activated_operators,
            conflict_markers=[],
            goal_states=goal_states,
            partial_outputs=previous_field.partial_outputs if previous_field else [],
            resonance={},
            field_energy=None,
            previous_outputs=previous_field.previous_outputs if previous_field else [],
            iteration=previous_field.iteration + 1 if previous_field else 0,
        )
        
        # Step 9: Compute initial resonance patterns
        # This creates interactions between activated memories
        self.compute_initial_resonance(field, events_layer)
        
        return field
    
    def activate_events(
        self,
        query_features: list[str],
        query_hv: np.ndarray,
        events_layer: ExposureEventsLayer,
        previous_field: Optional[StateField] = None,
        top_k: int = 20,
    ) -> Dict[str, float]:
        """
        Activate events from Layer 1 based on query similarity.
        
        Uses HDC-based similarity matching to find events that are relevant
        to the current query. Returns normalized activation strengths.
        
        If a previous_field is provided, can optionally boost activations
        for events that were active in the previous iteration (persistence).
        
        Args:
            query_features: List of feature strings from query
            query_hv: Query hypervector for HDC matching
            events_layer: Layer 1 for event retrieval
            previous_field: Previous field for context (optional)
            top_k: Number of top events to activate
            
        Returns:
            Dictionary mapping event_id to activation strength (0.0 to 1.0)
            
        Requirements:
        - 2.1: Activate relevant events simultaneously
        - 2.5: Maintain activation strengths
        """
        # Find similar events using HDC similarity
        similar_events = events_layer.find_similar_events(query_hv, top_k)
        
        if not similar_events:
            return {}
        
        # Normalize activations to [0, 1] range
        max_sim = max(sim for _, sim in similar_events)
        min_sim = min(sim for _, sim in similar_events)
        
        if max_sim == min_sim:
            # All similarities are equal, assign uniform activation
            activated = {event_id: 1.0 for event_id, _ in similar_events}
        else:
            # Normalize to [0, 1] with max_sim → 1.0
            activated = {}
            for event_id, sim in similar_events:
                normalized = (sim - min_sim) / (max_sim - min_sim)
                activated[event_id] = float(normalized)
        
        # Optional: Apply persistence boost from previous field
        # Events active in previous iteration get a small boost
        if previous_field is not None and previous_field.activated_events:
            persistence_boost = 0.1
            for event_id in activated:
                if event_id in previous_field.activated_events:
                    prev_activation = previous_field.activated_events[event_id]
                    # Boost current activation by fraction of previous activation
                    boost = persistence_boost * prev_activation
                    activated[event_id] = min(1.0, activated[event_id] + boost)
        
        return activated
    
    def activate_concepts(
        self,
        query_features: list[str],
        activated_events: Dict[str, float],
        events_layer: ExposureEventsLayer,
    ) -> Dict[str, float]:
        """
        Activate concepts from query features and activated events.
        
        Concepts are abstract patterns that emerge from co-occurring features.
        This method identifies concepts by:
        1. Extracting concept features from query
        2. Extracting concepts from activated events
        3. Aggregating and normalizing concept activations
        
        Args:
            query_features: Features from the input query
            activated_events: Already activated events with strengths
            events_layer: Layer 1 for accessing event data
            
        Returns:
            Dictionary mapping concept_id to activation strength (0.0 to 1.0)
            
        Requirements:
        - 2.2: Activate relevant concepts simultaneously
        - 2.5: Maintain activation strengths
        """
        concept_activation: Dict[str, float] = {}
        
        # Extract concepts from query features
        for feature in query_features:
            if feature.startswith("concept:"):
                # Query directly mentions a concept
                concept_activation[feature] = 1.0
        
        # Extract concepts from activated events
        for event_id, event_strength in activated_events.items():
            event = events_layer.get_event(event_id)
            if event is None:
                continue
            
            # Accumulate activation from this event's concept features
            for feature in event.features:
                if feature.startswith("concept:"):
                    current = concept_activation.get(feature, 0.0)
                    # Accumulate weighted by event activation
                    concept_activation[feature] = current + event_strength
        
        # Normalize concept activations to [0, 1]
        if concept_activation:
            max_activation = max(concept_activation.values())
            if max_activation > 0:
                for concept_id in concept_activation:
                    concept_activation[concept_id] /= max_activation
        
        return concept_activation
    
    def activate_operators(
        self,
        field_state: StateField,
        operators_layer: OperatorMemoryLayer,
        max_operators: int = 10,
        min_confidence: float = 0.3,
    ) -> Dict[str, float]:
        """
        Activate relevant operators from Layer 3 based on field state.
        
        Finds operators whose patterns match the current field state. This
        allows the field to leverage learned transformation patterns for
        candidate generation.
        
        Args:
            field_state: Current field state to match against
            operators_layer: Layer 3 containing stored operators
            max_operators: Maximum number of operators to activate
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary mapping operator_id to activation strength (0.0 to 1.0)
            
        Requirements:
        - 2.3: Activate relevant operators simultaneously
        - 4.7: Apply stored operators to novel inputs
        - 9.1: Identify applicable operators
        """
        # Find applicable operators from Layer 3
        applicable = operators_layer.find_applicable_operators(
            field_state=field_state,
            max_results=max_operators,
            min_confidence=min_confidence,
        )
        
        # Convert to activation dictionary
        # Match scores from operator layer are already normalized [0, 1]
        operator_activation: Dict[str, float] = {}
        for operator_id, match_score in applicable:
            operator_activation[operator_id] = match_score
        
        return operator_activation
    
    def compute_initial_resonance(
        self,
        field: StateField,
        events_layer: Optional[ExposureEventsLayer] = None,
    ) -> None:
        """
        Compute initial resonance patterns between activated memories.
        
        Resonance creates interactions between simultaneously activated memories:
        - Positive resonance: Memories that support each other (coherent patterns)
        - Negative resonance: Memories that conflict with each other (dissonance)
        
        This method computes resonance between:
        - Event-event pairs (based on feature overlap and co-activation history)
        - Concept-concept pairs (based on semantic similarity)
        - Event-concept pairs (based on feature containment)
        - Operator-event pairs (based on pattern matching)
        
        The resonance matrix is stored in field.resonance and used by:
        - Energy computation (high resonance = low energy)
        - Candidate emergence (resonant patterns guide generation)
        - Field stabilization (propagate resonance to amplify coherent patterns)
        
        Args:
            field: StateField to compute resonance for (modified in-place)
            events_layer: Layer 1 for accessing event data (optional)
            
        Requirements:
        - 2.6: Compute resonance between activated memories
        - 2.7: Allow memories to interact creating emergent patterns
        - 14.1: Compute resonance when multiple events activated
        - 14.5: Propagate resonance through field
        - 14.6: Allow resonance to amplify coherent patterns
        """
        # Collect all activated memory IDs
        activated_memories = []
        
        # Add activated events
        activated_memories.extend(field.activated_events.keys())
        
        # Add activated concepts
        activated_memories.extend(field.activated_concepts.keys())
        
        # Add activated operators
        activated_memories.extend(field.activated_operators.keys())
        
        if len(activated_memories) < 2:
            # Need at least 2 memories for resonance
            return
        
        # Compute resonance between all pairs of activated memories
        # We compute a subset to avoid quadratic explosion for large activations
        # Strategy: Compute resonance for top-k most activated memories
        max_resonance_pairs = 100  # Limit computation
        
        # Sort by activation strength (across all types)
        def get_activation(mem_id: str) -> float:
            if mem_id in field.activated_events:
                return field.activated_events[mem_id]
            elif mem_id in field.activated_concepts:
                return field.activated_concepts[mem_id]
            elif mem_id in field.activated_operators:
                return field.activated_operators[mem_id]
            return 0.0
        
        # Sort memories by activation strength
        sorted_memories = sorted(
            activated_memories,
            key=get_activation,
            reverse=True
        )
        
        # Take top memories for resonance computation
        top_k = min(15, len(sorted_memories))  # Limit to top 15 memories
        top_memories = sorted_memories[:top_k]
        
        # Compute resonance between pairs
        from ._memory_state_field import StateFieldLayer
        
        state_layer = StateFieldLayer()
        state_layer.current_field = field
        
        pairs_computed = 0
        for i, mem1 in enumerate(top_memories):
            for mem2 in top_memories[i + 1:]:
                if pairs_computed >= max_resonance_pairs:
                    break
                
                # Compute resonance using StateFieldLayer's method
                resonance = state_layer.compute_resonance(
                    mem1, mem2, events_layer, debug=False
                )
                
                # Store in resonance matrix (symmetric)
                field.resonance[(mem1, mem2)] = resonance
                field.resonance[(mem2, mem1)] = resonance
                
                pairs_computed += 1
            
            if pairs_computed >= max_resonance_pairs:
                break
        
        # Optional: Propagate resonance to amplify coherent patterns
        # This is a simple resonance amplification step
        self._propagate_resonance(field, iterations=1)
    
    def _propagate_resonance(
        self,
        field: StateField,
        iterations: int = 1,
    ) -> None:
        """
        Propagate resonance through the field to amplify coherent patterns.
        
        Uses a simple spreading activation algorithm:
        - Positive resonance spreads to connected memories
        - Negative resonance (dissonance) suppresses conflicting patterns
        
        This creates emergent patterns where coherent memory clusters
        become more strongly activated through mutual reinforcement.
        
        Args:
            field: StateField to propagate resonance in (modified in-place)
            iterations: Number of propagation iterations
            
        Requirements:
        - 14.5: Propagate resonance through field
        - 14.6: Amplify coherent patterns
        """
        if not field.resonance:
            return
        
        for _ in range(iterations):
            # Compute activation adjustments from resonance
            activation_delta: Dict[str, float] = {}
            
            for (mem1, mem2), resonance in field.resonance.items():
                if resonance <= 0:
                    continue  # Only propagate positive resonance
                
                # Get activations
                act1 = self._get_activation(mem1, field)
                act2 = self._get_activation(mem2, field)
                
                # Compute resonance contribution
                # High resonance between two active memories boosts both
                contribution = 0.05 * resonance * act1 * act2
                
                # Accumulate deltas
                activation_delta[mem1] = activation_delta.get(mem1, 0.0) + contribution
                activation_delta[mem2] = activation_delta.get(mem2, 0.0) + contribution
            
            # Apply activation adjustments (clamped to [0, 1])
            for mem_id, delta in activation_delta.items():
                if mem_id in field.activated_events:
                    field.activated_events[mem_id] = min(
                        1.0, field.activated_events[mem_id] + delta
                    )
                elif mem_id in field.activated_concepts:
                    field.activated_concepts[mem_id] = min(
                        1.0, field.activated_concepts[mem_id] + delta
                    )
                elif mem_id in field.activated_operators:
                    field.activated_operators[mem_id] = min(
                        1.0, field.activated_operators[mem_id] + delta
                    )
    
    def _get_activation(self, mem_id: str, field: StateField) -> float:
        """Get activation strength of a memory in the field."""
        if mem_id in field.activated_events:
            return field.activated_events[mem_id]
        elif mem_id in field.activated_concepts:
            return field.activated_concepts[mem_id]
        elif mem_id in field.activated_operators:
            return field.activated_operators[mem_id]
        return 0.0


__all__ = ["FieldFormation"]
