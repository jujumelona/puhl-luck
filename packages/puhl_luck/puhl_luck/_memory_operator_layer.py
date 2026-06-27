"""
Layer 3: Operator Memory Layer

Stores learned state transformation patterns as operators. Operators abstract repeated
transformation patterns across multiple exposures, enabling generalization beyond specific
event examples.

This layer supports:
- Storage of operators by type (completion, repair, explanation, comparison, etc.)
- Pattern matching to find applicable operators for a given field state
- Operator instantiation with context-specific parameters
- Operator indexing for efficient retrieval

Requirements: 1.3, 1.7, 4.1, 16.1-16.7
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ._memory_field_core import (
    OperatorRecord,
    OperatorType,
    StateField,
    StatePattern,
    TransformationRule,
)


class OperatorInstance:
    """
    An instantiated operator with context-specific parameters.
    
    Represents an operator that has been matched to a specific field state
    and is ready to be applied with the appropriate context bindings.
    """
    
    def __init__(
        self,
        operator_id: str,
        operator: OperatorRecord,
        context: StateField,
        match_score: float,
        bindings: Dict[str, any],
    ):
        """
        Initialize operator instance.
        
        Args:
            operator_id: ID of the operator being instantiated
            operator: The operator record
            context: The field state this operator is being applied to
            match_score: How well the operator pattern matches the context
            bindings: Context-specific parameter bindings
        """
        self.operator_id = operator_id
        self.operator = operator
        self.context = context
        self.match_score = match_score
        self.bindings = bindings


class OperatorMemoryLayer:
    """
    Layer 3 of the Predictive Field Memory system.
    
    Stores learned state transformation operators that capture repeated patterns
    of how partial states become complete. Enables generalization by abstracting
    transformation logic from specific examples.
    
    Key capabilities:
    - Store operators with pattern matching structures
    - Find operators applicable to current field state
    - Instantiate operators with context-appropriate parameters
    - Track operator usage and success statistics
    - Support multiple operator types (completion, repair, explanation, etc.)
    
    Requirements:
    - 1.3: Implement Layer 3 as Operator Memory Layer
    - 1.7: Store transformation types including completion, repair, explanation, etc.
    - 4.1: Store patterns as state transformation operators
    - 16.1-16.7: Support operator type taxonomy
    """
    
    def __init__(self):
        """Initialize the Operator Memory Layer."""
        # Core operator storage
        self.operators: Dict[str, OperatorRecord] = {}
        
        # Indexing by operator type for efficient retrieval
        self.operators_by_type: Dict[OperatorType, List[str]] = defaultdict(list)
        
        # Indexing by required features for pattern matching
        self.operators_by_feature: Dict[str, Set[str]] = defaultdict(set)
        
        # Indexing by required concepts
        self.operators_by_concept: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.total_operators = 0
        self.operators_applied = 0
        
    def store_operator(self, operator: OperatorRecord) -> str:
        """
        Store an operator in Layer 3.
        
        Adds the operator to storage and updates all indices for efficient
        pattern matching and retrieval.
        
        Args:
            operator: The operator record to store
            
        Returns:
            operator_id: The stored operator's ID
            
        Requirements:
        - 1.7: Store transformation types
        - 4.1: Store as state transformation operators
        - 16.1-16.7: Support all operator types
        """
        operator_id = operator.operator_id
        
        # Store operator
        self.operators[operator_id] = operator
        
        # Index by type
        self.operators_by_type[operator.operator_type].append(operator_id)
        
        # Index by required features
        for feature in operator.pattern.required_features:
            self.operators_by_feature[feature].add(operator_id)
        
        # Index by required concepts
        for concept in operator.pattern.required_concepts:
            self.operators_by_concept[concept].add(operator_id)
        
        # Update statistics
        self.total_operators += 1
        
        return operator_id
    
    def find_applicable_operators(
        self,
        field_state: StateField,
        max_results: int = 10,
        min_confidence: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """
        Find operators applicable to the current field state.
        
        Matches operator patterns against the field state to identify which
        operators could be applied. Returns operators sorted by match quality.
        
        Args:
            field_state: Current cognitive field state
            max_results: Maximum number of operators to return
            min_confidence: Minimum confidence threshold for operators
            
        Returns:
            List of (operator_id, match_score) tuples, sorted by match score
            
        Requirements:
        - 4.7: Apply stored operators to novel inputs
        - 9.1: Identify applicable operators from operator memory
        """
        if not self.operators:
            return []
        
        # Collect candidate operators through indexed lookup
        candidates: Counter[str] = Counter()
        
        # Match by features
        query_features = set(field_state.query_features)
        for feature in query_features:
            for op_id in self.operators_by_feature.get(feature, []):
                candidates[op_id] += 1
        
        # Match by activated concepts
        for concept_id in field_state.activated_concepts.keys():
            for op_id in self.operators_by_concept.get(concept_id, []):
                candidates[op_id] += 1
        
        # If no candidates from indexing, consider all operators
        if not candidates:
            candidates = Counter({op_id: 0 for op_id in self.operators.keys()})
        
        # Compute detailed match scores for candidates
        results = []
        for operator_id in candidates:
            operator = self.operators.get(operator_id)
            if operator is None or operator.confidence < min_confidence:
                continue
            
            match_score = self._compute_match_score(operator, field_state)
            
            if match_score > 0:
                results.append((operator_id, match_score))
        
        # Sort by match score (descending) and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def instantiate_operator(
        self,
        operator_id: str,
        context: StateField,
    ) -> Optional[OperatorInstance]:
        """
        Instantiate an operator with context-specific parameters.
        
        Takes an operator and binds it to a specific field state, creating
        an executable operator instance with all parameters resolved from
        the context.
        
        Args:
            operator_id: ID of operator to instantiate
            context: Field state providing the context
            
        Returns:
            OperatorInstance or None if instantiation fails
            
        Requirements:
        - 4.7: Instantiate operators with context-appropriate parameters
        - 9.2: Apply operators to current field state
        """
        operator = self.operators.get(operator_id)
        if operator is None:
            return None
        
        # Compute match score
        match_score = self._compute_match_score(operator, context)
        if match_score <= 0:
            return None
        
        # Extract bindings from context
        bindings = self._extract_bindings(operator, context)
        
        # Create operator instance
        instance = OperatorInstance(
            operator_id=operator_id,
            operator=operator,
            context=context,
            match_score=match_score,
            bindings=bindings,
        )
        
        # Update usage statistics
        operator.usage_count += 1
        self.operators_applied += 1
        
        return instance
    
    def get_operator(self, operator_id: str) -> Optional[OperatorRecord]:
        """
        Retrieve an operator by ID.
        
        Args:
            operator_id: Operator identifier
            
        Returns:
            OperatorRecord or None if not found
        """
        return self.operators.get(operator_id)
    
    def get_operators_by_type(self, operator_type: OperatorType) -> List[OperatorRecord]:
        """
        Retrieve all operators of a specific type.
        
        Args:
            operator_type: Type of operators to retrieve
            
        Returns:
            List of operator records
        """
        operator_ids = self.operators_by_type.get(operator_type, [])
        return [self.operators[op_id] for op_id in operator_ids if op_id in self.operators]
    
    def update_operator_stats(
        self,
        operator_id: str,
        success: bool,
    ) -> None:
        """
        Update operator statistics after application.
        
        Args:
            operator_id: ID of operator that was applied
            success: Whether the operator application was successful
        """
        operator = self.operators.get(operator_id)
        if operator is None:
            return
        
        # Update success rate using exponential moving average
        alpha = 0.1  # Learning rate
        target = 1.0 if success else 0.0
        operator.success_rate = (1 - alpha) * operator.success_rate + alpha * target
        
        # Update confidence based on usage and success
        # More usage with high success increases confidence
        usage_factor = min(1.0, operator.usage_count / 10.0)
        operator.confidence = usage_factor * operator.success_rate
    
    # =========================================================================
    # Internal helper methods (prefixed with _)
    # =========================================================================
    
    def _compute_match_score(
        self,
        operator: OperatorRecord,
        field_state: StateField,
    ) -> float:
        """
        Compute how well an operator pattern matches a field state.
        
        Args:
            operator: Operator to match
            field_state: Field state to match against
            
        Returns:
            Match score (0.0 to 1.0), higher is better
        """
        pattern = operator.pattern
        
        # Check preconditions (hard constraints)
        for precondition in operator.preconditions:
            if not self._check_precondition(precondition, field_state):
                return 0.0
        
        # Compute feature match score
        query_features = set(field_state.query_features)
        required_features = pattern.required_features
        
        if required_features:
            feature_overlap = len(query_features & required_features)
            feature_score = feature_overlap / len(required_features)
        else:
            feature_score = 1.0
        
        # Compute concept match score
        activated_concepts = set(field_state.activated_concepts.keys())
        required_concepts = pattern.required_concepts
        
        if required_concepts:
            concept_overlap = len(activated_concepts & required_concepts)
            concept_score = concept_overlap / len(required_concepts)
        else:
            concept_score = 1.0
        
        # Compute incompleteness match score
        # Check if field state has incompleteness markers that match operator patterns
        incompleteness_score = 0.0
        if pattern.incompleteness_markers:
            # Count how many incompleteness patterns are present
            matches = 0
            for marker in pattern.incompleteness_markers:
                if self._has_incompleteness(marker, field_state):
                    matches += 1
            incompleteness_score = matches / len(pattern.incompleteness_markers)
        else:
            incompleteness_score = 1.0
        
        # Compute goal match score
        active_goals = [g.goal_description for g in field_state.goal_states]
        goal_score = 0.0
        if pattern.goal_patterns:
            matches = 0
            for goal_pattern in pattern.goal_patterns:
                if any(goal_pattern in goal for goal in active_goals):
                    matches += 1
            goal_score = matches / len(pattern.goal_patterns) if pattern.goal_patterns else 1.0
        else:
            goal_score = 1.0
        
        # Combine scores with weights
        # Feature and concept matches are most important
        # Incompleteness and goal matches provide context
        match_score = (
            0.35 * feature_score +
            0.25 * concept_score +
            0.25 * incompleteness_score +
            0.15 * goal_score
        )
        
        # Weight by operator confidence
        match_score *= operator.confidence
        
        return match_score
    
    def _check_precondition(
        self,
        precondition: str,
        field_state: StateField,
    ) -> bool:
        """
        Check if a precondition is satisfied by the field state.
        
        Args:
            precondition: Precondition string to check
            field_state: Field state to check against
            
        Returns:
            True if precondition is satisfied, False otherwise
        """
        # Parse precondition format: "type:value"
        if ":" not in precondition:
            return False
        
        precond_type, precond_value = precondition.split(":", 1)
        
        if precond_type == "has_feature":
            return precond_value in field_state.query_features
        
        elif precond_type == "has_concept":
            return precond_value in field_state.activated_concepts
        
        elif precond_type == "has_goal":
            return any(precond_value in g.goal_description for g in field_state.goal_states)
        
        elif precond_type == "has_conflict":
            return any(precond_value in c.description for c in field_state.conflict_markers)
        
        elif precond_type == "min_activated_events":
            try:
                threshold = int(precond_value)
                return len(field_state.activated_events) >= threshold
            except ValueError:
                return False
        
        elif precond_type == "has_partial_output":
            return len(field_state.partial_outputs) > 0
        
        else:
            # Unknown precondition type
            return False
    
    def _has_incompleteness(
        self,
        marker: str,
        field_state: StateField,
    ) -> bool:
        """
        Check if field state has a specific incompleteness marker.
        
        Args:
            marker: Incompleteness marker to check for
            field_state: Field state to check
            
        Returns:
            True if incompleteness is present, False otherwise
        """
        # Check for various incompleteness indicators
        
        if marker == "unsatisfied_goal":
            return any(g.satisfaction_level < 0.5 for g in field_state.goal_states)
        
        elif marker == "has_conflict":
            return len(field_state.conflict_markers) > 0
        
        elif marker == "low_activation":
            # Check if overall activation is low
            total_activation = sum(field_state.activated_events.values())
            return total_activation < 1.0
        
        elif marker == "partial_output":
            return len(field_state.partial_outputs) > 0
        
        elif marker == "high_tension":
            # Check if field energy indicates high tension
            if field_state.field_energy is not None:
                return field_state.field_energy.tension_level > 0.6
            return False
        
        elif marker == "missing_evidence":
            # Check if field has low evidence
            if field_state.field_energy is not None:
                return field_state.field_energy.evidence < 0.3
            return False
        
        else:
            # Check if marker appears in field features
            return marker in field_state.query_features
    
    def _extract_bindings(
        self,
        operator: OperatorRecord,
        context: StateField,
    ) -> Dict[str, any]:
        """
        Extract context-specific parameter bindings for operator instantiation.
        
        Args:
            operator: Operator to extract bindings for
            context: Field state providing the context
            
        Returns:
            Dictionary of parameter bindings
        """
        bindings = {}
        
        # Extract basic context information
        bindings["query_features"] = context.query_features
        bindings["activated_events"] = list(context.activated_events.keys())
        bindings["activated_concepts"] = list(context.activated_concepts.keys())
        
        # Extract goals
        bindings["goals"] = [g.goal_description for g in context.goal_states]
        bindings["unsatisfied_goals"] = [
            g.goal_description for g in context.goal_states if g.satisfaction_level < 0.5
        ]
        
        # Extract conflicts
        bindings["conflicts"] = [c.description for c in context.conflict_markers]
        
        # Extract partial outputs
        bindings["partial_outputs"] = context.partial_outputs
        
        # Extract transformation rule parameters
        rule = operator.transformation
        for param_name, param_value in rule.parameters.items():
            # If parameter is a reference to context, resolve it
            if isinstance(param_value, str) and param_value.startswith("$"):
                # Parameter is a reference like "$query_features" or "$goals"
                ref = param_value[1:]  # Remove $ prefix
                if ref in bindings:
                    bindings[param_name] = bindings[ref]
                else:
                    bindings[param_name] = param_value
            else:
                bindings[param_name] = param_value
        
        # Extract high-activation events for context
        if context.activated_events:
            top_events = sorted(
                context.activated_events.items(),
                key=lambda x: x[1],
                reverse=True
            )
            bindings["top_event"] = top_events[0][0] if top_events else None
            bindings["top_events"] = [eid for eid, _ in top_events[:5]]
        
        # Extract high-activation concepts
        if context.activated_concepts:
            top_concepts = sorted(
                context.activated_concepts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            bindings["top_concept"] = top_concepts[0][0] if top_concepts else None
            bindings["top_concepts"] = [cid for cid, _ in top_concepts[:3]]
        
        return bindings


__all__ = ["OperatorMemoryLayer", "OperatorInstance"]
