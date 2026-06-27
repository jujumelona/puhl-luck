"""
Layer 2: State Field Layer

Represents the currently activated cognitive field containing simultaneously
activated memories, concepts, operators, conflicts, goals, and partial outputs.

This layer is the core of the field-based memory architecture, where memories
interact through resonance and field dynamics drive output generation.

Requirements: 1.2, 2.4, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from ._memory_field_core import (
    ConflictMarker,
    GoalState,
    InputContext,
    StateField,
)

if TYPE_CHECKING:
    from ._memory_exposure_layer import ExposureEventsLayer
    from ._memory_operator_layer import OperatorMemoryLayer


class StateFieldLayer:
    """
    Layer 2 of the Predictive Field Memory system.
    
    Manages the currently activated cognitive field state, including:
    - Activated events, concepts, and operators with strengths
    - Conflict markers indicating tensions in the field
    - Goal states tracking what the field should achieve
    - Partial outputs representing incomplete generations
    - Resonance patterns between activated memories
    
    The StateField represents a living cognitive state where memories interact
    dynamically through resonance, and field energy drives candidate generation.
    
    Requirements:
    - 1.2: State Field Layer stores currently activated cognitive states
    - 2.4: Represents activations as interactive cognitive field
    - 12.1-12.7: Contains all required field components
    """

    def __init__(self):
        """Initialize an empty State Field Layer."""
        # Current state field (None until activated)
        self.current_field: Optional[StateField] = None

    def activate_from_input(
        self,
        input_context: InputContext,
        events_layer: ExposureEventsLayer,
        operators_layer: Optional[OperatorMemoryLayer] = None,
    ) -> StateField:
        """
        Create a cognitive field by activating memories from input context.
        
        This method forms the initial field state by:
        1. Extracting features from the input context
        2. Creating a query hypervector
        3. Activating relevant events from Layer 1 (Exposure Events)
        4. Activating relevant concepts from concept memory
        5. Activating relevant operators from Layer 3 (if provided)
        6. Initializing goals from input context
        
        The activation is simultaneous across all layers, creating an interactive
        field rather than a sequential search process.
        
        Requirements:
        - 2.1: Input simultaneously activates multiple memory layers
        - 2.4: Creates interactive cognitive field
        - 2.5: Maintains activation strengths
        
        Args:
            input_context: Input containing text, modality, goals, constraints
            events_layer: Layer 1 providing access to stored events
            operators_layer: Layer 3 providing access to operators (optional)
            
        Returns:
            StateField with activated memories and initial field state
        """
        # Extract features from input text
        features = events_layer.compute_event_features(
            input_context.text, input_context.modality
        )
        
        # Create query hypervector
        query_hv = events_layer._bundle_event(features, [])
        
        # Activate events from Layer 1 using HDC similarity
        activated_events = self._activate_events(query_hv, events_layer)
        
        # Activate concepts (for now, extract from activated events)
        activated_concepts = self._activate_concepts(activated_events, events_layer)
        
        # Activate operators from Layer 3 (if operators_layer provided)
        activated_operators: Dict[str, float] = {}
        if operators_layer is not None:
            # We need to create a preliminary field state to match operators against
            preliminary_field = StateField(
                query_features=features,
                query_hv=query_hv,
                activated_events=activated_events,
                activated_concepts=activated_concepts,
                activated_operators={},  # Empty initially
                conflict_markers=[],
                goal_states=self._initialize_goals(input_context),
                partial_outputs=[],
                resonance={},
                field_energy=None,
                previous_outputs=[],
                iteration=0,
            )
            activated_operators = self._activate_operators(preliminary_field, operators_layer)
        
        # Initialize goals from input context
        goal_states = self._initialize_goals(input_context)
        
        # Create the StateField
        field = StateField(
            query_features=features,
            query_hv=query_hv,
            activated_events=activated_events,
            activated_concepts=activated_concepts,
            activated_operators=activated_operators,
            conflict_markers=[],
            goal_states=goal_states,
            partial_outputs=[],
            resonance={},
            field_energy=None,
            previous_outputs=[],
            iteration=0,
        )
        
        # Store as current field
        self.current_field = field
        
        return field

    def add_conflict(self, conflict: ConflictMarker) -> None:
        """
        Add a conflict marker to the current field.
        
        Conflicts increase field energy and represent incompatibilities between
        activated memories, constraint violations, or repetition patterns.
        
        Requirement 12.5: Maintain conflict markers with strengths
        
        Args:
            conflict: ConflictMarker to add to the field
            
        Raises:
            RuntimeError: If no field is currently active
        """
        if self.current_field is None:
            raise RuntimeError("Cannot add conflict: no active field")
        
        self.current_field.conflict_markers.append(conflict)

    def add_goal(self, goal: GoalState) -> None:
        """
        Add a goal state to the current field.
        
        Goals guide candidate generation by defining what the field should achieve.
        Unsatisfied goals create tension that drives output generation.
        
        Requirement 12.6: Maintain goal states with satisfaction levels
        
        Args:
            goal: GoalState to add to the field
            
        Raises:
            RuntimeError: If no field is currently active
        """
        if self.current_field is None:
            raise RuntimeError("Cannot add goal: no active field")
        
        self.current_field.goal_states.append(goal)

    def update_with_output(self, output: str) -> None:
        """
        Update the field state with new output, preparing for next iteration.
        
        This method is called during recursive stabilization after each output
        is generated. It:
        1. Adds the output to partial_outputs
        2. Records it in previous_outputs history
        3. Increments the iteration counter
        
        The field will be reactivated in the next stabilization iteration to
        recompute activations based on the updated state.
        
        Requirement 6.1: Field updates after each output for recursive stabilization
        
        Args:
            output: Generated output text to incorporate into field state
            
        Raises:
            RuntimeError: If no field is currently active
        """
        if self.current_field is None:
            raise RuntimeError("Cannot update with output: no active field")
        
        # Add output to partial outputs
        self.current_field.partial_outputs.append(output)
        
        # Record in history
        self.current_field.previous_outputs.append(output)
        
        # Increment iteration counter
        self.current_field.iteration += 1

    def compute_resonance(
        self, mem1: str, mem2: str, events_layer: Optional[ExposureEventsLayer] = None,
        debug: bool = False,
    ) -> float:
        """
        Compute resonance between two activated memories.
        
        Resonance measures how strongly two memories support or conflict with
        each other. Positive resonance indicates mutual support (memories
        reinforce each other), while negative resonance indicates conflict.
        
        The computation is based on:
        1. Feature overlap: High feature overlap produces positive resonance
        2. Co-activation history: Memories that frequently co-occur resonate positively
        3. Conflict markers: Memories marked as conflicting have negative resonance
        
        Requirements:
        - 2.6: Compute resonance between activated memories
        - 14.1: Compute resonance when multiple events are activated
        - 14.2: Increase activation when memories mutually support
        - 14.3: Decrease activation when memories conflict
        - 14.4: Detect resonance through feature overlap and co-activation history
        
        Args:
            mem1: First memory identifier (event_id, concept_id, or operator_id)
            mem2: Second memory identifier
            events_layer: Optional Layer 1 for accessing event data
            debug: If True, print debug information
            
        Returns:
            Resonance strength (positive = support, negative = conflict, 0 = neutral)
            Range: approximately [-1.0, 1.0] but not strictly bounded
            
        Raises:
            RuntimeError: If no field is currently active
        """
        if self.current_field is None:
            raise RuntimeError("Cannot compute resonance: no active field")
        
        # Check if resonance already computed
        key = (mem1, mem2)
        if key in self.current_field.resonance:
            return self.current_field.resonance[key]
        
        # Self-resonance is always 1.0 (a memory always supports itself)
        if mem1 == mem2:
            self.current_field.resonance[key] = 1.0
            return 1.0
        
        # Initialize resonance components
        feature_overlap_score = 0.0
        coactivation_score = 0.0
        conflict_penalty = 0.0
        
        # Component 1: Feature overlap
        # High feature overlap indicates memories are about similar topics
        feature_overlap_score = self._compute_feature_overlap(mem1, mem2, events_layer)
        
        # Component 2: Co-activation history
        # Memories that co-occur frequently tend to support each other
        if events_layer is not None:
            coactivation_score = self._compute_coactivation_strength(
                mem1, mem2, events_layer
            )
        
        # Component 3: Conflict detection
        # Check if these memories are marked as conflicting
        conflict_penalty = self._compute_conflict_penalty(mem1, mem2)
        
        if debug:
            print(f"  Feature overlap: {feature_overlap_score:.4f}")
            print(f"  Co-activation: {coactivation_score:.4f}")
            print(f"  Conflict penalty: {conflict_penalty:.4f}")
        
        # Combine components:
        # - Feature overlap: weight 0.5 (primary signal)
        # - Co-activation: weight 0.3 (supporting signal)
        # - Conflict: negative contribution (overrides positive signals)
        resonance = (
            0.5 * feature_overlap_score + 0.3 * coactivation_score - conflict_penalty
        )
        
        if debug:
            print(f"  Final resonance: {resonance:.4f}")
        
        # Store in resonance matrix (symmetric)
        self.current_field.resonance[key] = resonance
        self.current_field.resonance[(mem2, mem1)] = resonance
        
        return resonance

    def _compute_feature_overlap(
        self,
        mem1: str,
        mem2: str,
        events_layer: Optional[ExposureEventsLayer],
    ) -> float:
        """
        Compute feature overlap between two memories.
        
        For events, compares their feature sets using Jaccard similarity.
        For concepts, compares the concept identifiers themselves.
        For mixed types, uses a heuristic based on activation patterns.
        
        Args:
            mem1: First memory identifier
            mem2: Second memory identifier
            events_layer: Optional Layer 1 for accessing event data
            
        Returns:
            Overlap score in [0.0, 1.0]
        """
        # Determine memory types
        mem1_type = self._get_memory_type(mem1)
        mem2_type = self._get_memory_type(mem2)
        
        # Both are events: compute Jaccard similarity of features
        if mem1_type == "event" and mem2_type == "event" and events_layer is not None:
            event1 = events_layer.get_event(mem1)
            event2 = events_layer.get_event(mem2)
            
            if event1 is None or event2 is None:
                return 0.0
            
            features1 = set(event1.features)
            features2 = set(event2.features)
            
            if not features1 or not features2:
                return 0.0
            
            intersection = len(features1 & features2)
            union = len(features1 | features2)
            
            return intersection / union if union > 0 else 0.0
        
        # Both are concepts: check if they share the base concept
        elif mem1_type == "concept" and mem2_type == "concept":
            # Extract concept name without "concept:" prefix if present
            concept1 = mem1.replace("concept:", "")
            concept2 = mem2.replace("concept:", "")
            
            # Exact match: high resonance
            if concept1 == concept2:
                return 1.0
            
            # Partial match: check if one is substring of other
            if concept1 in concept2 or concept2 in concept1:
                return 0.5
            
            return 0.0
        
        # Mixed types or unknown: use activation correlation heuristic
        else:
            # If both memories are activated in the current field,
            # their simultaneous activation suggests some relationship
            activation1 = self._get_activation_strength(mem1)
            activation2 = self._get_activation_strength(mem2)
            
            if activation1 > 0 and activation2 > 0:
                # Both activated: minimal positive overlap
                return 0.2
            
            return 0.0

    def _compute_coactivation_strength(
        self, mem1: str, mem2: str, events_layer: ExposureEventsLayer
    ) -> float:
        """
        Compute co-activation strength between two memories from history.
        
        Uses the co-activation edge graph to determine if these memories
        have frequently appeared together in the past.
        
        Args:
            mem1: First memory identifier
            mem2: Second memory identifier
            events_layer: Layer 1 for accessing co-activation data
            
        Returns:
            Co-activation score in [0.0, 1.0]
        """
        # Only applicable to events (concepts/operators don't have co-activation history)
        mem1_type = self._get_memory_type(mem1)
        mem2_type = self._get_memory_type(mem2)
        
        if mem1_type != "event" or mem2_type != "event":
            return 0.0
        
        # Get events
        event1 = events_layer.get_event(mem1)
        event2 = events_layer.get_event(mem2)
        
        if event1 is None or event2 is None:
            return 0.0
        
        # Check co-activation through shared features in the edge graph
        # Get feature IDs for both events
        features1_ids = set()
        features2_ids = set()
        
        for feature in event1.features:
            fid = events_layer.feature_to_id.get(feature)
            if fid is not None:
                features1_ids.add(fid)
        
        for feature in event2.features:
            fid = events_layer.feature_to_id.get(feature)
            if fid is not None:
                features2_ids.add(fid)
        
        if not features1_ids or not features2_ids:
            return 0.0
        
        # Compute co-activation strength from edge weights
        total_edge_weight = 0.0
        edge_count = 0
        
        for fid1 in features1_ids:
            for fid2 in features2_ids:
                # Check both directions in edge graph
                edge_weight = events_layer.edges.get((fid1, fid2), 0.0)
                if edge_weight == 0.0:
                    edge_weight = events_layer.edges.get((fid2, fid1), 0.0)
                
                if edge_weight > 0:
                    total_edge_weight += edge_weight
                    edge_count += 1
        
        if edge_count == 0:
            return 0.0
        
        # Normalize: average edge weight, clipped to [0, 1]
        avg_weight = total_edge_weight / edge_count
        return min(avg_weight, 1.0)

    def _compute_conflict_penalty(self, mem1: str, mem2: str) -> float:
        """
        Compute conflict penalty between two memories.
        
        Checks if the memories are marked as conflicting in the current field's
        conflict markers. Conflicting memories have negative resonance.
        
        Args:
            mem1: First memory identifier
            mem2: Second memory identifier
            
        Returns:
            Conflict penalty (positive value that will be subtracted from resonance)
        """
        if self.current_field is None:
            return 0.0
        
        # Check conflict markers for this pair
        for conflict in self.current_field.conflict_markers:
            involved = set(conflict.involved_memories)
            
            # If both memories are involved in this conflict
            if mem1 in involved and mem2 in involved:
                # Return conflict strength as penalty
                return conflict.strength
        
        return 0.0

    def _get_memory_type(self, mem_id: str) -> str:
        """
        Determine the type of memory identifier.
        
        Args:
            mem_id: Memory identifier
            
        Returns:
            Memory type: "event", "concept", "operator", or "unknown"
        """
        if self.current_field is None:
            return "unknown"
        
        if mem_id in self.current_field.activated_events:
            return "event"
        elif mem_id in self.current_field.activated_concepts:
            return "concept"
        elif mem_id in self.current_field.activated_operators:
            return "operator"
        else:
            return "unknown"

    def _get_activation_strength(self, mem_id: str) -> float:
        """
        Get the activation strength of a memory in the current field.
        
        Args:
            mem_id: Memory identifier
            
        Returns:
            Activation strength in [0.0, 1.0], or 0.0 if not activated
        """
        if self.current_field is None:
            return 0.0
        
        # Check all activation dictionaries
        if mem_id in self.current_field.activated_events:
            return self.current_field.activated_events[mem_id]
        elif mem_id in self.current_field.activated_concepts:
            return self.current_field.activated_concepts[mem_id]
        elif mem_id in self.current_field.activated_operators:
            return self.current_field.activated_operators[mem_id]
        else:
            return 0.0

    def get_tension_sources(self) -> List[str]:
        """
        Identify sources of tension in the current field.
        
        Tension sources indicate high field energy from:
        - Conflicts between memories
        - Unsatisfied goals
        - Incomplete partial outputs
        
        This is a helper method for candidate emergence to identify what
        needs to be addressed in generation.
        
        Returns:
            List of tension source descriptions
            
        Raises:
            RuntimeError: If no field is currently active
        """
        if self.current_field is None:
            raise RuntimeError("Cannot get tension sources: no active field")
        
        sources = []
        
        # Add conflicts as tension sources
        for conflict in self.current_field.conflict_markers:
            sources.append(f"Conflict: {conflict.description}")
        
        # Add unsatisfied goals as tension sources
        for goal in self.current_field.goal_states:
            if goal.satisfaction_level < 1.0:
                sources.append(
                    f"Unsatisfied goal: {goal.goal_description} "
                    f"(satisfaction: {goal.satisfaction_level:.2f})"
                )
        
        # Add incomplete outputs as tension sources
        if self.current_field.partial_outputs:
            last_output = self.current_field.partial_outputs[-1]
            if len(last_output) < 10:  # Simple heuristic for incompleteness
                sources.append(f"Incomplete output: {last_output}")
        
        return sources

    # =========================================================================
    # Private helper methods for activation
    # =========================================================================

    def _activate_events(
        self, query_hv: np.ndarray, events_layer: ExposureEventsLayer, top_k: int = 20
    ) -> Dict[str, float]:
        """
        Activate events from Layer 1 based on HDC similarity to query.
        
        Uses the ExposureEventsLayer's HDC indexing to find similar events
        and returns them with normalized activation strengths.
        
        Args:
            query_hv: Query hypervector
            events_layer: Layer 1 for event retrieval
            top_k: Number of top events to activate
            
        Returns:
            Dictionary mapping event_id to activation strength (0.0 to 1.0)
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
            return {event_id: 1.0 for event_id, _ in similar_events}
        
        # Normalize to [0, 1] with max_sim → 1.0
        activated = {}
        for event_id, sim in similar_events:
            normalized = (sim - min_sim) / (max_sim - min_sim)
            activated[event_id] = float(normalized)
        
        return activated

    def _activate_concepts(
        self, activated_events: Dict[str, float], events_layer: ExposureEventsLayer
    ) -> Dict[str, float]:
        """
        Activate concepts from activated events.
        
        Extracts concept features from activated events and computes their
        activation strengths based on the events that contain them.
        
        Args:
            activated_events: Already activated events with strengths
            events_layer: Layer 1 for accessing event data
            
        Returns:
            Dictionary mapping concept_id to activation strength
        """
        concept_activation: Dict[str, float] = {}
        
        for event_id, event_strength in activated_events.items():
            event = events_layer.get_event(event_id)
            if event is None:
                continue
            
            # Extract concept features from event
            for feature in event.features:
                if feature.startswith("concept:"):
                    # Accumulate activation from this event
                    concept_activation[feature] = (
                        concept_activation.get(feature, 0.0) + event_strength
                    )
        
        # Normalize concept activations
        if concept_activation:
            max_activation = max(concept_activation.values())
            if max_activation > 0:
                for concept_id in concept_activation:
                    concept_activation[concept_id] /= max_activation
        
        return concept_activation

    def _activate_operators(
        self,
        field_state: StateField,
        operators_layer: OperatorMemoryLayer,
    ) -> Dict[str, float]:
        """
        Activate relevant operators from Layer 3 based on current field state.
        
        Finds operators whose patterns match the current field state and returns
        their activation strengths. This allows the field to leverage learned
        transformation patterns.
        
        Requirements:
        - 2.3: Activate relevant operators simultaneously
        - 4.7: Apply stored operators to novel inputs
        - 9.1: Identify applicable operators from operator memory
        
        Args:
            field_state: Current field state to match operators against
            operators_layer: Layer 3 containing stored operators
            
        Returns:
            Dictionary mapping operator_id to activation_strength
        """
        # Find applicable operators
        applicable = operators_layer.find_applicable_operators(
            field_state=field_state,
            max_results=10,
            min_confidence=0.3,
        )
        
        # Convert to activation dictionary
        operator_activation: Dict[str, float] = {}
        for operator_id, match_score in applicable:
            operator_activation[operator_id] = match_score
        
        return operator_activation

    def _initialize_goals(self, input_context: InputContext) -> List[GoalState]:
        """
        Initialize goal states from input context.
        
        Creates GoalState objects from goals specified in the input context.
        Initial satisfaction level is 0.0 (unsatisfied).
        
        Args:
            input_context: Input containing optional goals list
            
        Returns:
            List of GoalState objects
        """
        goal_states = []
        
        if input_context.goals:
            for i, goal_desc in enumerate(input_context.goals):
                goal = GoalState(
                    goal_id=f"goal_{i}",
                    goal_description=goal_desc,
                    satisfaction_level=0.0,
                    constraints=input_context.constraints or [],
                )
                goal_states.append(goal)
        
        return goal_states


__all__ = ["StateFieldLayer"]
