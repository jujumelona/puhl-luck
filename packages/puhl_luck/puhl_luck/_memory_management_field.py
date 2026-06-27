"""
Memory Management for Field-Based Architecture.

Implements intelligent pruning of events, operators, and transitions
to maintain memory health while preserving important patterns and diversity.
Uses Rust implementation for high performance with parallel processing.
"""

from typing import Dict, List, Set, Optional
import time

# Check for Rust availability
RUST_AVAILABLE = False
try:
    import sys
    import os
    # Add package directory to path if needed
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'puhl_luck')
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    
    import puhl_luck_core
    # Check if memory_management submodule exists
    if hasattr(puhl_luck_core, 'memory_management'):
        RUST_AVAILABLE = True
except (ImportError, AttributeError):
    pass


class MemoryManager:
    """
    Manages memory pruning and optimization for cognitive field.
    
    Implements targeted forgetting that:
    - Keeps high-value memories (novel, recent, frequently accessed)
    - Maintains diversity in stored patterns
    - Removes low-confidence operators and transitions
    - Preserves protected/important items
    - Triggers automatic pruning when thresholds exceeded
    """
    
    def __init__(
        self,
        event_capacity: int = 10000,
        operator_capacity: int = 1000,
        transition_capacity: int = 5000,
        min_operator_confidence: float = 0.3,
        min_operator_uses: int = 2,
        auto_prune: bool = True,
    ):
        """
        Initialize memory manager.
        
        Args:
            event_capacity: Maximum number of events to keep
            operator_capacity: Maximum number of operators to keep
            transition_capacity: Maximum number of transitions to keep
            min_operator_confidence: Minimum confidence for operator retention
            min_operator_uses: Minimum use count for operator retention
            auto_prune: Enable automatic pruning when capacity exceeded
        """
        self.event_capacity = event_capacity
        self.operator_capacity = operator_capacity
        self.transition_capacity = transition_capacity
        self.min_operator_confidence = min_operator_confidence
        self.min_operator_uses = min_operator_uses
        self.auto_prune = auto_prune
        
        # Statistics
        self.total_events_pruned = 0
        self.total_operators_pruned = 0
        self.total_transitions_pruned = 0
        self.last_prune_time = 0
        
    def prune_events(
        self,
        event_ids: List[str],
        event_novelty: Dict[str, float],
        event_last_accessed: Dict[str, int],
        event_activation: Dict[str, float],
        protected_ids: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Prune events based on novelty, recency, and activation.
        
        Uses Rust implementation if available for parallel processing.
        
        Args:
            event_ids: List of all event IDs
            event_novelty: Novelty scores (0-1)
            event_last_accessed: Last access timestamps
            event_activation: Current activation levels (0-1)
            protected_ids: Event IDs to never prune
            
        Returns:
            List of event IDs to remove
        """
        if protected_ids is None:
            protected_ids = set()
        
        if len(event_ids) <= self.event_capacity:
            return []
        
        current_time = int(time.time())
        
        if RUST_AVAILABLE:
            # Use Rust implementation (parallel)
            import puhl_luck_core
            to_prune = puhl_luck_core.memory_management.prune_events_rust(
                event_ids,
                event_novelty,
                event_last_accessed,
                event_activation,
                protected_ids,
                self.event_capacity,
                current_time,
            )
        else:
            # Python fallback
            to_prune = self._prune_events_python(
                event_ids,
                event_novelty,
                event_last_accessed,
                event_activation,
                protected_ids,
                current_time,
            )
        
        self.total_events_pruned += len(to_prune)
        self.last_prune_time = current_time
        
        return to_prune
    
    def _prune_events_python(
        self,
        event_ids: List[str],
        event_novelty: Dict[str, float],
        event_last_accessed: Dict[str, int],
        event_activation: Dict[str, float],
        protected_ids: Set[str],
        current_time: int,
    ) -> List[str]:
        """Python fallback for event pruning."""
        num_to_prune = len(event_ids) - self.event_capacity
        
        # Compute retention scores
        scored_events = []
        for event_id in event_ids:
            if event_id in protected_ids:
                continue
            
            novelty = event_novelty.get(event_id, 0.0)
            last_accessed = event_last_accessed.get(event_id, 0)
            activation = event_activation.get(event_id, 0.0)
            
            # Recency score (days since access)
            days_since_access = max(1, (current_time - last_accessed) // 86400)
            recency_score = 1.0 / days_since_access
            
            # Weighted retention score
            retention_score = (
                0.3 * novelty +
                0.3 * recency_score +
                0.4 * activation
            )
            
            scored_events.append((event_id, retention_score))
        
        # Sort by score (lowest first)
        scored_events.sort(key=lambda x: x[1])
        
        # Take lowest scoring
        to_prune = [event_id for event_id, _ in scored_events[:num_to_prune]]
        
        return to_prune
    
    def prune_operators(
        self,
        operator_ids: List[str],
        operator_confidence: Dict[str, float],
        operator_use_count: Dict[str, int],
        operator_success_count: Dict[str, int],
        operator_last_used: Dict[str, int],
    ) -> List[str]:
        """
        Prune low-confidence or unused operators.
        
        Uses Rust implementation if available for parallel processing.
        
        Args:
            operator_ids: List of all operator IDs
            operator_confidence: Confidence scores (0-1)
            operator_use_count: Number of times used
            operator_success_count: Number of successful applications
            operator_last_used: Last use timestamps
            
        Returns:
            List of operator IDs to remove
        """
        current_time = int(time.time())
        
        if RUST_AVAILABLE:
            # Use Rust implementation (parallel)
            import puhl_luck_core
            to_prune = puhl_luck_core.memory_management.prune_operators_rust(
                operator_ids,
                operator_confidence,
                operator_use_count,
                operator_success_count,
                operator_last_used,
                self.min_operator_confidence,
                self.min_operator_uses,
                current_time,
            )
        else:
            # Python fallback
            to_prune = self._prune_operators_python(
                operator_ids,
                operator_confidence,
                operator_use_count,
                operator_success_count,
                operator_last_used,
                current_time,
            )
        
        self.total_operators_pruned += len(to_prune)
        
        return to_prune
    
    def _prune_operators_python(
        self,
        operator_ids: List[str],
        operator_confidence: Dict[str, float],
        operator_use_count: Dict[str, int],
        operator_success_count: Dict[str, int],
        operator_last_used: Dict[str, int],
        current_time: int,
    ) -> List[str]:
        """Python fallback for operator pruning."""
        to_prune = []
        
        for op_id in operator_ids:
            confidence = operator_confidence.get(op_id, 0.0)
            use_count = operator_use_count.get(op_id, 0)
            success_count = operator_success_count.get(op_id, 0)
            last_used = operator_last_used.get(op_id, 0)
            
            # Success rate
            success_rate = success_count / use_count if use_count > 0 else 0.0
            
            # Days since last use
            days_since_use = max(1, (current_time - last_used) // 86400)
            
            # Prune conditions
            should_prune = (
                (confidence < self.min_operator_confidence and use_count < self.min_operator_uses) or
                (days_since_use > 180 and success_rate < 0.3) or
                (use_count == 0 and confidence < self.min_operator_confidence * 1.5)
            )
            
            if should_prune:
                to_prune.append(op_id)
        
        return to_prune
    
    def prune_transitions(
        self,
        transition_ids: List[str],
        transition_relevance: Dict[str, float],
        transition_match_count: Dict[str, int],
        transition_last_matched: Dict[str, int],
    ) -> List[str]:
        """
        Prune low-relevance or unused transitions.
        
        Uses Rust implementation if available for parallel processing.
        
        Args:
            transition_ids: List of all transition IDs
            transition_relevance: Relevance scores (0-1)
            transition_match_count: Number of times matched
            transition_last_matched: Last match timestamps
            
        Returns:
            List of transition IDs to remove
        """
        if len(transition_ids) <= self.transition_capacity:
            return []
        
        current_time = int(time.time())
        
        if RUST_AVAILABLE:
            # Use Rust implementation (parallel)
            import puhl_luck_core
            to_prune = puhl_luck_core.memory_management.prune_transitions_rust(
                transition_ids,
                transition_relevance,
                transition_match_count,
                transition_last_matched,
                self.transition_capacity,
                current_time,
            )
        else:
            # Python fallback
            to_prune = self._prune_transitions_python(
                transition_ids,
                transition_relevance,
                transition_match_count,
                transition_last_matched,
                current_time,
            )
        
        self.total_transitions_pruned += len(to_prune)
        
        return to_prune
    
    def _prune_transitions_python(
        self,
        transition_ids: List[str],
        transition_relevance: Dict[str, float],
        transition_match_count: Dict[str, int],
        transition_last_matched: Dict[str, int],
        current_time: int,
    ) -> List[str]:
        """Python fallback for transition pruning."""
        import math
        
        num_to_prune = len(transition_ids) - self.transition_capacity
        
        # Compute retention scores
        scored_transitions = []
        for trans_id in transition_ids:
            relevance = transition_relevance.get(trans_id, 0.0)
            match_count = transition_match_count.get(trans_id, 0)
            last_matched = transition_last_matched.get(trans_id, 0)
            
            # Recency
            days_since_match = max(1, (current_time - last_matched) // 86400)
            recency_score = 1.0 / math.sqrt(days_since_match)
            
            # Utility
            utility_score = math.log(match_count + 1.0)
            
            # Retention score
            retention_score = (
                0.4 * relevance +
                0.3 * recency_score +
                0.3 * utility_score
            )
            
            scored_transitions.append((trans_id, retention_score))
        
        # Sort by score (lowest first)
        scored_transitions.sort(key=lambda x: x[1])
        
        # Take lowest scoring
        to_prune = [trans_id for trans_id, _ in scored_transitions[:num_to_prune]]
        
        return to_prune
    
    def compute_health_metrics(
        self,
        event_count: int,
        operator_count: int,
        transition_count: int,
        event_novelty_values: List[float],
        operator_confidence_values: List[float],
        transition_relevance_values: List[float],
    ) -> Dict:
        """
        Compute memory health metrics.
        
        Uses Rust implementation if available for parallel statistics.
        
        Args:
            event_count: Number of events
            operator_count: Number of operators
            transition_count: Number of transitions
            event_novelty_values: All novelty scores
            operator_confidence_values: All confidence scores
            transition_relevance_values: All relevance scores
            
        Returns:
            Dictionary with health metrics
        """
        if RUST_AVAILABLE:
            # Use Rust implementation (parallel)
            import puhl_luck_core
            metrics = puhl_luck_core.memory_management.compute_memory_health_rust(
                event_count,
                operator_count,
                transition_count,
                event_novelty_values,
                operator_confidence_values,
                transition_relevance_values,
            )
        else:
            # Python fallback
            metrics = self._compute_health_python(
                event_count,
                operator_count,
                transition_count,
                event_novelty_values,
                operator_confidence_values,
                transition_relevance_values,
            )
        
        # Add pruning statistics
        metrics['total_events_pruned'] = self.total_events_pruned
        metrics['total_operators_pruned'] = self.total_operators_pruned
        metrics['total_transitions_pruned'] = self.total_transitions_pruned
        metrics['last_prune_time'] = self.last_prune_time
        
        return metrics
    
    def _compute_health_python(
        self,
        event_count: int,
        operator_count: int,
        transition_count: int,
        event_novelty_values: List[float],
        operator_confidence_values: List[float],
        transition_relevance_values: List[float],
    ) -> Dict:
        """Python fallback for health computation."""
        metrics = {
            'event_count': event_count,
            'operator_count': operator_count,
            'transition_count': transition_count,
            'total_memory_items': event_count + operator_count + transition_count,
        }
        
        # Event stats
        if event_novelty_values:
            metrics['avg_event_novelty'] = sum(event_novelty_values) / len(event_novelty_values)
            metrics['max_event_novelty'] = max(event_novelty_values)
            metrics['min_event_novelty'] = min(event_novelty_values)
        
        # Operator stats
        if operator_confidence_values:
            metrics['avg_operator_confidence'] = sum(operator_confidence_values) / len(operator_confidence_values)
            metrics['max_operator_confidence'] = max(operator_confidence_values)
            metrics['min_operator_confidence'] = min(operator_confidence_values)
        
        # Transition stats
        if transition_relevance_values:
            metrics['avg_transition_relevance'] = sum(transition_relevance_values) / len(transition_relevance_values)
            metrics['max_transition_relevance'] = max(transition_relevance_values)
            metrics['min_transition_relevance'] = min(transition_relevance_values)
        
        # Health score
        memory_health_score = (
            (1.0 if event_count < self.event_capacity else self.event_capacity / event_count) * 0.4 +
            (metrics.get('avg_operator_confidence', 0.5)) * 0.3 +
            (metrics.get('avg_transition_relevance', 0.5)) * 0.3
        )
        
        metrics['memory_health_score'] = memory_health_score
        metrics['needs_pruning'] = (
            event_count > self.event_capacity or
            operator_count > self.operator_capacity or
            transition_count > self.transition_capacity
        )
        
        return metrics
    
    def should_auto_prune(
        self,
        event_count: int,
        operator_count: int,
        transition_count: int,
    ) -> bool:
        """
        Check if automatic pruning should be triggered.
        
        Args:
            event_count: Current number of events
            operator_count: Current number of operators
            transition_count: Current number of transitions
            
        Returns:
            True if pruning needed
        """
        if not self.auto_prune:
            return False
        
        return (
            event_count > self.event_capacity or
            operator_count > self.operator_capacity or
            transition_count > self.transition_capacity
        )


__all__ = ["MemoryManager", "RUST_AVAILABLE"]
