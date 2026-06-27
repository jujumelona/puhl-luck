"""
Operator Induction for Predictive Field Memory

Learns repeated state transformation patterns from state transitions and abstracts
them into reusable operators. This enables generalization beyond specific examples
by identifying common transformation structures across multiple exposures.

Supports inducing operators for:
- Completion: incomplete → complete
- Repair: error → correction
- Explanation: question → answer
- Comparison: entity → comparison
- Transformation: problem → solution
- Composition: part → whole

Requirements:
- 4.2: Identify repeated patterns across multiple exposures
- 4.3: Create explanation_operator from question→answer patterns
- 4.4: Create repair_operator from error→correction patterns
- 4.5: Create completion_operator from incomplete→complete patterns
- 4.6: Create transformation_operator from problem→solution patterns
- 4.7: Apply stored operators to novel inputs
- 4.8: Enable generalization beyond specific event examples
- 16.1-16.7: Support all operator types
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ._brain_hdc import hv_similarity
from ._memory_field_core import (
    OperatorRecord,
    OperatorType,
    StatePattern,
    StateTransition,
    TransformationRule,
)


class PatternCluster:
    """
    A cluster of similar state transitions that share transformation structure.
    
    Used during operator induction to group transitions that exhibit the same
    kind of transformation, which can then be abstracted into an operator.
    """
    
    def __init__(self, cluster_id: str):
        """
        Initialize pattern cluster.
        
        Args:
            cluster_id: Unique identifier for this cluster
        """
        self.cluster_id = cluster_id
        self.transitions: List[StateTransition] = []
        self.centroid_vector: Optional[np.ndarray] = None
        self.operator_type: Optional[OperatorType] = None
        
    def add_transition(self, transition: StateTransition) -> None:
        """Add a transition to this cluster."""
        self.transitions.append(transition)
        
    def compute_centroid(self) -> None:
        """Compute the centroid of completion vectors in this cluster."""
        if not self.transitions:
            return
        
        # Average the completion vectors
        vectors = [t.completion_vector for t in self.transitions if t.completion_vector.size > 0]
        if not vectors:
            return
        
        # Ensure all vectors have the same size
        min_size = min(v.size for v in vectors)
        truncated_vectors = [v[:min_size] for v in vectors]
        
        # For binary HDC vectors, use majority voting per dimension
        stacked = np.stack(truncated_vectors)
        self.centroid_vector = (np.mean(stacked, axis=0) > 0.5).astype(np.int8)


class OperatorInduction:
    """
    Learns operators from repeated state transformation patterns.
    
    Analyzes transition history to identify repeated patterns, abstracts them into
    operators, and stores them in the Operator Memory Layer. This enables the system
    to generalize learned transformations to novel situations.
    
    Key capabilities:
    - Identify repeated transformation patterns across exposures
    - Cluster similar transitions by transformation structure
    - Abstract patterns into state patterns
    - Generalize transformations into transformation rules
    - Classify operator types (completion, repair, explanation, etc.)
    - Create operators with appropriate confidence levels
    
    Requirements:
    - 4.2: Identify repeated patterns across multiple exposures
    - 4.3: Create explanation_operator from question→answer patterns
    - 4.4: Create repair_operator from error→correction patterns
    - 4.5: Create completion_operator from incomplete→complete patterns
    - 4.6: Create transformation_operator from problem→solution patterns
    - 4.7: Apply stored operators to novel inputs
    - 4.8: Enable generalization beyond specific event examples
    """
    
    def __init__(self, hdc_dimensions: int = 10000):
        """
        Initialize operator induction.
        
        Args:
            hdc_dimensions: Dimensionality for HDC vectors (in bits)
        """
        self.hdc_dimensions = hdc_dimensions
        
        # Statistics
        self.total_operators_induced = 0
        self.induction_runs = 0
    
    def induce_from_history(
        self,
        transitions: List[StateTransition],
        min_pattern_count: int = 3,
        similarity_threshold: float = 0.6,
    ) -> List[OperatorRecord]:
        """
        Induce operators from transition history.
        
        Main entry point for operator induction. Analyzes transitions to identify
        repeated patterns and creates operators that capture transformation structure.
        
        Args:
            transitions: List of state transitions to analyze
            min_pattern_count: Minimum number of transitions required to induce an operator
            similarity_threshold: Minimum similarity for transitions to be clustered together
            
        Returns:
            List of induced operator records
            
        Requirements:
        - 4.2: Identify repeated patterns across multiple exposures
        - 4.5: Create completion_operator from incomplete→complete patterns
        - 4.8: Enable generalization beyond specific event examples
        """
        if len(transitions) < 2:
            return []
        
        self.induction_runs += 1
        
        # Step 1: Cluster similar transitions
        clusters = self.identify_repeated_patterns(
            transitions,
            similarity_threshold=similarity_threshold
        )
        
        # Step 2: Filter clusters by minimum size
        valid_clusters = [c for c in clusters if len(c.transitions) >= min_pattern_count]
        
        if not valid_clusters:
            return []
        
        # Step 3: Induce operators from valid clusters
        operators = []
        for cluster in valid_clusters:
            operator = self._induce_operator_from_cluster(cluster)
            if operator is not None:
                operators.append(operator)
        
        # Update statistics
        self.total_operators_induced += len(operators)
        
        return operators
    
    def identify_repeated_patterns(
        self,
        transitions: List[StateTransition],
        similarity_threshold: float = 0.6,
    ) -> List[PatternCluster]:
        """
        Identify repeated patterns across transitions using clustering.
        
        Groups transitions with similar transformation structure into clusters.
        Uses HDC vector similarity on completion vectors to identify patterns.
        
        Args:
            transitions: List of transitions to cluster
            similarity_threshold: Minimum similarity for clustering
            
        Returns:
            List of pattern clusters
            
        Requirement 4.2: Identify repeated patterns across multiple exposures
        """
        if not transitions:
            return []
        
        # **FAST PATH**: Use Rust for greedy clustering
        # Build items: (transition_id, completion_features)
        items = []
        trans_by_id = {}
        for trans in transitions:
            if trans.completion_features:
                items.append((trans.transition_id, trans.completion_features))
                trans_by_id[trans.transition_id] = trans
        
        if not items:
            return []
        
        # Try Rust fast path
        try:
            from puhl_luck.puhl_luck_core import greedy_cluster_transitions_rust
            cluster_lists = greedy_cluster_transitions_rust(items, similarity_threshold)
            # cluster_lists is list of lists: [[trans_id, ...], ...]
            
            # Convert to PatternCluster objects
            clusters = []
            for i, member_ids in enumerate(cluster_lists):
                cluster = PatternCluster(f"cluster_{i}")
                for tid in member_ids:
                    if tid in trans_by_id:
                        cluster.add_transition(trans_by_id[tid])
                cluster.compute_centroid()
                clusters.append(cluster)
            
            return clusters
        except (ImportError, Exception):
            # Fall back to Python if Rust unavailable
            pass
        
        # **FALLBACK**: Python implementation
        # Initialize clusters
        clusters: List[PatternCluster] = []
        
        # Simple greedy clustering algorithm
        for transition in transitions:
            if transition.completion_vector.size == 0:
                continue
            
            # Find best matching cluster
            best_cluster = None
            best_similarity = similarity_threshold
            
            for cluster in clusters:
                if cluster.centroid_vector is None:
                    continue
                
                # Compute similarity between transition and cluster centroid
                min_size = min(
                    transition.completion_vector.size,
                    cluster.centroid_vector.size
                )
                
                sim = hv_similarity(
                    transition.completion_vector[:min_size],
                    cluster.centroid_vector[:min_size],
                    min_size
                )
                
                if sim > best_similarity:
                    best_similarity = sim
                    best_cluster = cluster
            
            # Add to best cluster or create new cluster
            if best_cluster is not None:
                best_cluster.add_transition(transition)
                best_cluster.compute_centroid()  # Update centroid
            else:
                # Create new cluster
                cluster_id = f"cluster_{len(clusters)}"
                new_cluster = PatternCluster(cluster_id)
                new_cluster.add_transition(transition)
                new_cluster.centroid_vector = transition.completion_vector.copy()
                clusters.append(new_cluster)
        
        return clusters
    
    def abstract_pattern(
        self,
        transitions: List[StateTransition]
    ) -> StatePattern:
        """
        Abstract a state pattern from a set of similar transitions.
        
        Identifies common features, concepts, incompleteness markers, and goal patterns
        across transitions to create an abstract pattern that matches field states.
        
        Args:
            transitions: List of transitions with similar structure
            
        Returns:
            StatePattern representing the abstract pattern
            
        Requirement 4.2: Abstract patterns from transitions
        """
        if not transitions:
            # Return empty pattern
            return StatePattern(
                required_features=set(),
                required_concepts=set(),
                incompleteness_markers=[],
                goal_patterns=[],
            )
        
        # Extract features from partial states
        all_features: List[Set[str]] = []
        all_concepts: List[Set[str]] = []
        all_goals: List[str] = []
        
        for trans in transitions:
            partial = trans.partial_state
            all_features.append(set(partial.query_features))
            all_concepts.append(set(partial.activated_concepts.keys()))
            all_goals.extend([g.goal_description for g in partial.goal_states])
        
        # Find common features (appear in at least 50% of transitions)
        threshold = len(transitions) * 0.5
        feature_counts: Counter[str] = Counter()
        for features in all_features:
            feature_counts.update(features)
        
        required_features = {
            feat for feat, count in feature_counts.items()
            if count >= threshold
        }
        
        # Find common concepts (appear in at least 40% of transitions)
        concept_threshold = len(transitions) * 0.4
        concept_counts: Counter[str] = Counter()
        for concepts in all_concepts:
            concept_counts.update(concepts)
        
        required_concepts = {
            concept for concept, count in concept_counts.items()
            if count >= concept_threshold
        }
        
        # Identify incompleteness markers from partial states
        incompleteness_markers = self._extract_incompleteness_markers(transitions)
        
        # Identify goal patterns (common goals across transitions)
        goal_counts: Counter[str] = Counter(all_goals)
        goal_patterns = [
            goal for goal, count in goal_counts.items()
            if count >= len(transitions) * 0.3
        ]
        
        return StatePattern(
            required_features=required_features,
            required_concepts=required_concepts,
            incompleteness_markers=incompleteness_markers,
            goal_patterns=goal_patterns,
        )
    
    def generalize_transformation(
        self,
        transitions: List[StateTransition]
    ) -> TransformationRule:
        """
        Generalize a transformation rule from similar transitions.
        
        Extracts the common transformation logic across transitions, creating
        a rule that can be applied to novel inputs with similar structure.
        
        Args:
            transitions: List of transitions with similar structure
            
        Returns:
            TransformationRule representing the generalized transformation
            
        Requirement 4.2: Generalize transformations from transitions
        """
        if not transitions:
            # Return default rule
            return TransformationRule(
                rule_type="learned_transition",
                parameters={},
                confidence_threshold=0.5,
            )
        
        # Analyze completion characteristics
        avg_completion_size = np.mean([
            len(t.completion_features) for t in transitions
        ])
        
        # Determine rule type based on characteristics
        if avg_completion_size < 5:
            rule_type = "template"
        else:
            rule_type = "learned_transition"
        
        # Extract common completion features
        all_completion_features: List[str] = []
        for trans in transitions:
            all_completion_features.extend(trans.completion_features)
        
        feature_counts = Counter(all_completion_features)
        common_features = [
            feat for feat, count in feature_counts.most_common(10)
            if count >= len(transitions) * 0.3
        ]
        
        # Build parameters
        parameters = {
            "common_features": common_features,
            "avg_completion_size": float(avg_completion_size),
            "num_examples": len(transitions),
        }
        
        # Compute confidence based on cluster coherence
        # Higher coherence = transitions are more similar = higher confidence
        confidence_threshold = self._compute_cluster_confidence(transitions)
        
        return TransformationRule(
            rule_type=rule_type,
            parameters=parameters,
            confidence_threshold=confidence_threshold,
        )
    
    # =========================================================================
    # Internal helper methods
    # =========================================================================
    
    def _induce_operator_from_cluster(
        self,
        cluster: PatternCluster
    ) -> Optional[OperatorRecord]:
        """
        Induce an operator from a pattern cluster.
        
        Args:
            cluster: Pattern cluster containing similar transitions
            
        Returns:
            OperatorRecord or None if induction fails
        """
        if not cluster.transitions:
            return None
        
        # Determine operator type based on transition characteristics
        operator_type = self._classify_operator_type(cluster.transitions)
        
        # Abstract the state pattern
        pattern = self.abstract_pattern(cluster.transitions)
        
        # Generalize the transformation rule
        transformation = self.generalize_transformation(cluster.transitions)
        
        # Extract preconditions
        preconditions = self._extract_preconditions(cluster.transitions)
        
        # Create completion template
        completion_template = self._create_completion_template(cluster.transitions)
        
        # Compute operator confidence
        confidence = self._compute_operator_confidence(cluster.transitions)
        
        # Generate operator ID
        operator_id = self._generate_operator_id(operator_type, pattern)
        
        # Create operator record
        operator = OperatorRecord(
            operator_id=operator_id,
            operator_type=operator_type,
            pattern=pattern,
            preconditions=preconditions,
            transformation=transformation,
            completion_template=completion_template,
            confidence=confidence,
            usage_count=0,
            success_rate=0.0,
            generalization_level=self._compute_generalization_level(pattern),
            induced_from=[t.transition_id for t in cluster.transitions],
            timestamp=time.time(),
        )
        
        return operator
    
    def _classify_operator_type(
        self,
        transitions: List[StateTransition]
    ) -> OperatorType:
        """
        Classify the type of operator based on transition characteristics.
        
        Requirements:
        - 4.3: Create explanation_operator from question→answer patterns
        - 4.4: Create repair_operator from error→correction patterns
        - 4.5: Create completion_operator from incomplete→complete patterns
        - 4.6: Create transformation_operator from problem→solution patterns
        """
        # Analyze features to determine operator type
        
        # Count feature occurrences in partial states
        feature_counts: Counter[str] = Counter()
        for trans in transitions:
            feature_counts.update(trans.partial_state.query_features)
        
        # Check for question patterns (explanation operator)
        question_keywords = {"question", "what", "why", "how", "when", "where", "who"}
        if any(kw in feature_counts for kw in question_keywords):
            return OperatorType.EXPLANATION
        
        # Check for error/repair patterns (repair operator)
        error_keywords = {"error", "bug", "fix", "incorrect", "wrong", "issue"}
        if any(kw in feature_counts for kw in error_keywords):
            return OperatorType.REPAIR
        
        # Check for comparison patterns (comparison operator)
        comparison_keywords = {"compare", "difference", "similar", "versus", "vs"}
        if any(kw in feature_counts for kw in comparison_keywords):
            return OperatorType.COMPARISON
        
        # Check for transformation/problem-solving patterns
        transform_keywords = {"solve", "transform", "convert", "translate", "problem"}
        if any(kw in feature_counts for kw in transform_keywords):
            return OperatorType.TRANSFORMATION
        
        # Check for composition patterns
        composition_keywords = {"combine", "merge", "integrate", "compose"}
        if any(kw in feature_counts for kw in composition_keywords):
            return OperatorType.COMPOSITION
        
        # Default: completion operator
        return OperatorType.COMPLETION
    
    def _extract_incompleteness_markers(
        self,
        transitions: List[StateTransition]
    ) -> List[str]:
        """Extract incompleteness markers from partial states."""
        markers: Set[str] = set()
        
        for trans in transitions:
            partial = trans.partial_state
            
            # Check for unsatisfied goals
            if any(g.satisfaction_level < 0.5 for g in partial.goal_states):
                markers.add("unsatisfied_goal")
            
            # Check for conflicts
            if partial.conflict_markers:
                markers.add("has_conflict")
            
            # Check for partial outputs
            if partial.partial_outputs:
                markers.add("partial_output")
            
            # Check for low activation
            total_activation = sum(partial.activated_events.values())
            if total_activation < 1.0:
                markers.add("low_activation")
        
        return list(markers)
    
    def _extract_preconditions(
        self,
        transitions: List[StateTransition]
    ) -> List[str]:
        """Extract preconditions from transitions."""
        preconditions: List[str] = []
        
        # Check for common features required
        feature_counts: Counter[str] = Counter()
        for trans in transitions:
            feature_counts.update(trans.partial_state.query_features)
        
        # Features appearing in >70% of transitions become preconditions
        threshold = len(transitions) * 0.7
        for feature, count in feature_counts.items():
            if count >= threshold:
                preconditions.append(f"has_feature:{feature}")
        
        # Check for common goals
        goal_counts: Counter[str] = Counter()
        for trans in transitions:
            for goal in trans.partial_state.goal_states:
                goal_counts[goal.goal_description] += 1
        
        for goal, count in goal_counts.items():
            if count >= threshold:
                preconditions.append(f"has_goal:{goal}")
        
        return preconditions
    
    def _create_completion_template(
        self,
        transitions: List[StateTransition]
    ) -> str:
        """Create a completion template from transitions."""
        # Extract most common completion features
        all_completion_features: List[str] = []
        for trans in transitions:
            all_completion_features.extend(trans.completion_features)
        
        feature_counts = Counter(all_completion_features)
        top_features = [feat for feat, _ in feature_counts.most_common(5)]
        
        # Create simple template
        if top_features:
            template = " ".join(top_features)
        else:
            template = "{context}"
        
        return template
    
    def _compute_operator_confidence(
        self,
        transitions: List[StateTransition]
    ) -> float:
        """
        Compute confidence score for an operator based on transition coherence.
        
        Higher coherence (more similar transitions) = higher confidence
        """
        if len(transitions) <= 1:
            return 0.5
        
        # Compute pairwise similarities of completion vectors
        similarities: List[float] = []
        
        for i in range(len(transitions)):
            for j in range(i + 1, len(transitions)):
                vec1 = transitions[i].completion_vector
                vec2 = transitions[j].completion_vector
                
                if vec1.size > 0 and vec2.size > 0:
                    min_size = min(vec1.size, vec2.size)
                    sim = hv_similarity(vec1[:min_size], vec2[:min_size], min_size)
                    similarities.append(sim)
        
        if not similarities:
            return 0.5
        
        # Average similarity as confidence
        avg_similarity = np.mean(similarities)
        
        # Boost confidence for larger clusters
        size_boost = min(0.2, len(transitions) * 0.02)
        
        confidence = min(0.95, avg_similarity + size_boost)
        
        return float(confidence)
    
    def _compute_cluster_confidence(
        self,
        transitions: List[StateTransition]
    ) -> float:
        """Compute confidence threshold for transformation rule."""
        # Similar to operator confidence but scaled for rule application
        operator_conf = self._compute_operator_confidence(transitions)
        
        # Rule confidence threshold is slightly lower
        rule_confidence = max(0.3, operator_conf - 0.2)
        
        return float(rule_confidence)
    
    def _compute_generalization_level(
        self,
        pattern: StatePattern
    ) -> int:
        """
        Compute how abstract/generalized the operator is.
        
        Lower required features = higher generalization level
        """
        num_required = len(pattern.required_features) + len(pattern.required_concepts)
        
        if num_required <= 2:
            return 3  # Highly generalized
        elif num_required <= 5:
            return 2  # Moderately generalized
        elif num_required <= 10:
            return 1  # Slightly generalized
        else:
            return 0  # Very specific
    
    def _generate_operator_id(
        self,
        operator_type: OperatorType,
        pattern: StatePattern
    ) -> str:
        """Generate a stable operator ID from type and pattern."""
        # Create ID from operator type and pattern characteristics
        features_str = ",".join(sorted(list(pattern.required_features)[:5]))
        concepts_str = ",".join(sorted(list(pattern.required_concepts)[:3]))
        
        id_string = f"{operator_type.value}:{features_str}:{concepts_str}"
        hash_obj = hashlib.sha256(id_string.encode())
        
        return f"op_{operator_type.value}_{hash_obj.hexdigest()[:12]}"


__all__ = ["OperatorInduction", "PatternCluster"]
