"""
Layer 4: Transition Memory Layer

Stores partial-to-complete state transitions (S_partial → S_complete), enabling
the system to learn completion dynamics from experience. Unlike before/after pairs,
transitions specifically capture incomplete→complete transformations across different
domains (conversation, code, documents, reasoning).

Requirements:
- 1.4: Layer 4 implementation as Transition Memory Layer
- 1.8: Store S_partial → S_complete pairs, not S_before → S_after
- 5.1: Transitions stored as S_partial → S_complete
- 5.2: Conversation turns as incomplete→completed context
- 5.3: Code generation as incomplete→completed code
- 5.4: Reasoning steps as incomplete→completed reasoning
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._brain_hdc import hdc_bands, hv_similarity
from ._memory_field_core import CompletionPattern, StateField, StateTransition


class TransitionMemoryLayer:
    """
    Layer 4 of the Predictive Field Memory system.
    
    Stores partial-to-complete state transitions that capture how incomplete states
    are completed across different domains. These transitions serve as training data
    for the state completion mechanism and guide candidate generation.
    
    Unlike traditional sequence memory (S_before → S_after), this layer specifically
    stores incomplete→complete pairs, focusing on the restoration/completion operation
    rather than general temporal succession.
    """

    def __init__(self, hdc_dimensions: int = 10000):
        """
        Initialize the Transition Memory Layer.
        
        Args:
            hdc_dimensions: Dimensionality for HDC vectors (in bits)
        """
        self.hdc_dimensions = hdc_dimensions
        
        # Transition storage
        self.transitions: Dict[str, StateTransition] = {}
        
        # HDC-based indexing for fast similarity search
        # Maps band tuples to sets of transition IDs
        self.transition_index: Dict[Tuple[int, int], set[str]] = {}
        
        # Domain-specific indexing for targeted retrieval
        self.domain_transitions: Dict[str, List[str]] = {
            "conversation": [],
            "code": [],
            "document": [],
            "reasoning": [],
        }
        
        # Modality-specific indexing
        self.modality_transitions: Dict[str, List[str]] = {
            "text": [],
            "code": [],
            "image": [],
            "audio": [],
            "bytes": [],
        }
        
        # Statistics
        self.total_transitions_stored = 0

    def store_transition(
        self, 
        partial: StateField, 
        complete: StateField,
        modality: str = "text",
        domain: str = "conversation",
    ) -> str:
        """
        Store a partial-to-complete state transition.
        
        This method stores an (incomplete state, complete state) pair, representing
        how the system learned to complete a particular type of incomplete state.
        The completion vector captures what was added to move from partial to complete.
        
        Args:
            partial: The incomplete state field
            complete: The complete state field
            modality: Content modality (text, code, image, audio, bytes)
            domain: Domain of the transition (conversation, code, document, reasoning)
            
        Returns:
            transition_id: Unique identifier for the stored transition
            
        Requirements:
        - 1.8: Store S_partial → S_complete pairs
        - 5.1: Transitions stored as S_partial → S_complete, not S_before → S_after
        - 5.2: Conversation turns as incomplete→completed context
        - 5.3: Code generation as incomplete→completed code
        - 5.4: Reasoning steps as incomplete→completed reasoning
        """
        # Generate stable transition ID from the state characteristics
        transition_id = self._generate_transition_id(partial, complete, modality, domain)
        
        # Compute completion vector: what was added to move from partial to complete
        completion_vector = self._compute_completion_vector(partial, complete)
        
        # Extract completion features: new features that appeared in complete state
        completion_features = self._extract_completion_features(partial, complete)
        
        # Create transition record
        transition = StateTransition(
            transition_id=transition_id,
            partial_state=partial,
            complete_state=complete,
            completion_vector=completion_vector,
            completion_features=completion_features,
            modality=modality,
            domain=domain,
            timestamp=time.time(),
            relevance_count=0,  # Initially unused
        )
        
        # Store transition
        self.transitions[transition_id] = transition
        
        # Index by HDC bands for fast similarity search
        self._index_transition(transition_id, completion_vector)
        
        # Index by domain and modality for targeted retrieval
        if domain in self.domain_transitions:
            self.domain_transitions[domain].append(transition_id)
        
        if modality in self.modality_transitions:
            self.modality_transitions[modality].append(transition_id)
        
        # Update statistics
        self.total_transitions_stored += 1
        
        return transition_id

    def find_similar_transitions(
        self, 
        current_partial: StateField, 
        top_k: int = 10,
        domain_filter: Optional[str] = None,
        modality_filter: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find transitions with similar partial states using HDC similarity.
        
        Given a current incomplete state, finds previously stored transitions
        that started from similar incomplete states. This allows the system
        to reuse completion patterns learned from similar situations.
        
        Args:
            current_partial: The current incomplete state to match
            top_k: Number of top similar transitions to return
            domain_filter: Optional domain to restrict search (conversation, code, etc.)
            modality_filter: Optional modality to restrict search (text, code, etc.)
            
        Returns:
            List of (transition_id, similarity_score) tuples, sorted by similarity
            
        Requirement 1.4: Retrieve relevant transitions for current incomplete state
        """
        if not self.transitions:
            return []
        
        # Apply filters to get candidate transitions
        candidate_ids = self._get_filtered_candidates(domain_filter, modality_filter)
        
        if not candidate_ids:
            return []
        
        # **FAST PATH**: Use Rust for feature-based similarity
        query_features = current_partial.query_features
        
        # Build stored transitions data: (id, partial_features, complete_features, relevance_count)
        stored_data = []
        for trans_id in candidate_ids:
            trans = self.transitions.get(trans_id)
            if trans is not None:
                stored_data.append((
                    trans_id,
                    trans.partial_state.query_features,
                    trans.complete_state.query_features,
                    trans.relevance_count,
                ))
        
        if not stored_data:
            return []
        
        # Try Rust fast path
        try:
            from puhl_luck.puhl_luck_core import find_transitions_by_features_rust
            results = find_transitions_by_features_rust(
                query_features,
                stored_data,
                top_k,
                0.0,  # min_sim
            )
            # results is list of (trans_id, similarity)
            if results:
                return results
        except (ImportError, Exception):
            # Fall back to Python if Rust unavailable
            pass
        
        # **FALLBACK**: Python implementation with HDC
        # Use HDC-based band indexing for initial filtering
        query_hv = current_partial.query_hv
        if query_hv.size == 0:
            # Fallback: if no query vector, use all candidates
            band_candidates = set(candidate_ids)
        else:
            # Get candidate transitions from band index
            bands = hdc_bands(query_hv, max(1, len(self.transitions)))
            band_candidates = set()
            for band in bands:
                for trans_id in self.transition_index.get(band, []):
                    if trans_id in candidate_ids:
                        band_candidates.add(trans_id)
            
            # If band index returns too few, use filtered candidates
            if len(band_candidates) < top_k:
                band_candidates = set(candidate_ids)
        
        # Compute actual similarities
        results = []
        for trans_id in band_candidates:
            transition = self.transitions.get(trans_id)
            if transition is None:
                continue
            
            # Compute similarity between current partial state and stored partial state
            similarity = self._compute_state_similarity(
                current_partial, 
                transition.partial_state
            )
            results.append((trans_id, similarity))
        
        # Sort by similarity (descending) and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_completion_pattern(self, transition_id: str) -> Optional[CompletionPattern]:
        """
        Extract the completion pattern from a stored transition.
        
        Returns an abstract representation of how the incomplete state was completed,
        which can be applied to similar partial states.
        
        Args:
            transition_id: ID of the transition to extract pattern from
            
        Returns:
            CompletionPattern or None if transition not found
            
        Requirement 1.4: Support learning of completion dynamics
        """
        transition = self.transitions.get(transition_id)
        if transition is None:
            return None
        
        # Extract what was added during completion
        added_features = transition.completion_features
        
        # Extract concepts that became activated during completion
        partial_concepts = set(transition.partial_state.activated_concepts.keys())
        complete_concepts = set(transition.complete_state.activated_concepts.keys())
        added_concepts = complete_concepts - partial_concepts
        
        # Determine completion type based on characteristics
        completion_type = self._infer_completion_type(transition)
        
        # Compute reliability based on how often this transition was useful
        reliability = self._compute_transition_reliability(transition)
        
        return CompletionPattern(
            added_features=added_features,
            added_concepts=added_concepts,
            completion_type=completion_type,
            modality=transition.modality,
            domain=transition.domain,
            completion_vector=transition.completion_vector,
            reliability=reliability,
        )

    def get_transition(self, transition_id: str) -> Optional[StateTransition]:
        """
        Retrieve a transition by its ID.
        
        Args:
            transition_id: Transition identifier
            
        Returns:
            StateTransition or None if not found
        """
        return self.transitions.get(transition_id)

    def update_relevance(self, transition_id: str, increment: int = 1) -> None:
        """
        Update the relevance count for a transition.
        
        Called when a transition is successfully used to guide completion,
        tracking which transitions are most useful.
        
        Args:
            transition_id: ID of the transition to update
            increment: Amount to increment relevance count (default: 1)
        """
        transition = self.transitions.get(transition_id)
        if transition is not None:
            transition.relevance_count += increment

    def get_transitions_by_domain(self, domain: str, limit: int = 100) -> List[str]:
        """
        Get transition IDs for a specific domain.
        
        Args:
            domain: Domain to filter by (conversation, code, document, reasoning)
            limit: Maximum number of transitions to return
            
        Returns:
            List of transition IDs
        """
        domain_trans = self.domain_transitions.get(domain, [])
        return domain_trans[:limit]

    def get_transitions_by_modality(self, modality: str, limit: int = 100) -> List[str]:
        """
        Get transition IDs for a specific modality.
        
        Args:
            modality: Modality to filter by (text, code, image, audio, bytes)
            limit: Maximum number of transitions to return
            
        Returns:
            List of transition IDs
        """
        modality_trans = self.modality_transitions.get(modality, [])
        return modality_trans[:limit]

    # =========================================================================
    # Internal helper methods
    # =========================================================================

    def _generate_transition_id(
        self, 
        partial: StateField, 
        complete: StateField,
        modality: str,
        domain: str,
    ) -> str:
        """Generate a stable ID for a transition."""
        import hashlib
        
        # Create ID from key characteristics
        partial_features = ",".join(sorted(partial.query_features[:10]))
        complete_features = ",".join(sorted(complete.query_features[:10]))
        partial_goals = ",".join(sorted(g.goal_id for g in partial.goal_states))
        
        id_string = f"{modality}:{domain}:{partial_features}:{complete_features}:{partial_goals}"
        hash_obj = hashlib.sha256(id_string.encode())
        return f"trans_{hash_obj.hexdigest()[:16]}"

    def _compute_completion_vector(
        self, 
        partial: StateField, 
        complete: StateField
    ) -> np.ndarray:
        """
        Compute HDC vector representing what was added during completion.
        
        This is the difference between complete and partial states in HDC space.
        """
        # Simple approach: subtract partial from complete
        # If vectors are different sizes, use the smaller size
        partial_hv = partial.query_hv
        complete_hv = complete.query_hv
        
        if partial_hv.size == 0 or complete_hv.size == 0:
            # Return zero vector if either is empty
            return np.zeros(self.hdc_dimensions, dtype=np.int8)
        
        min_size = min(partial_hv.size, complete_hv.size)
        
        # XOR operation for binary HDC vectors (difference)
        completion = np.zeros(min_size, dtype=np.int8)
        completion = complete_hv[:min_size] ^ partial_hv[:min_size]
        
        # Pad to target dimensions if needed
        if completion.size < self.hdc_dimensions:
            padded = np.zeros(self.hdc_dimensions, dtype=np.int8)
            padded[:completion.size] = completion
            return padded
        
        return completion[:self.hdc_dimensions]

    def _extract_completion_features(
        self, 
        partial: StateField, 
        complete: StateField
    ) -> List[str]:
        """
        Extract new features that appeared in complete state but not in partial.
        """
        partial_features = set(partial.query_features)
        complete_features = set(complete.query_features)
        
        # New features added during completion
        added_features = complete_features - partial_features
        
        return list(added_features)

    def _index_transition(self, transition_id: str, completion_vector: np.ndarray) -> None:
        """
        Add transition to HDC band index for fast similarity search.
        """
        if completion_vector.size == 0:
            return
        
        # Generate bands from completion vector
        bands = hdc_bands(completion_vector, max(1, len(self.transitions)))
        
        # Add to index
        for band in bands:
            if band not in self.transition_index:
                self.transition_index[band] = set()
            self.transition_index[band].add(transition_id)

    def _get_filtered_candidates(
        self,
        domain_filter: Optional[str],
        modality_filter: Optional[str],
    ) -> List[str]:
        """
        Get candidate transition IDs after applying domain/modality filters.
        """
        candidates = set(self.transitions.keys())
        
        # Apply domain filter
        if domain_filter is not None and domain_filter in self.domain_transitions:
            domain_set = set(self.domain_transitions[domain_filter])
            candidates = candidates.intersection(domain_set)
        
        # Apply modality filter
        if modality_filter is not None and modality_filter in self.modality_transitions:
            modality_set = set(self.modality_transitions[modality_filter])
            candidates = candidates.intersection(modality_set)
        
        return list(candidates)

    def _compute_state_similarity(
        self, 
        state1: StateField, 
        state2: StateField
    ) -> float:
        """
        Compute similarity between two state fields.
        
        Uses multiple signals:
        - HDC vector similarity (primary)
        - Feature overlap
        - Goal similarity
        """
        # Primary: HDC vector similarity
        if state1.query_hv.size > 0 and state2.query_hv.size > 0:
            min_size = min(state1.query_hv.size, state2.query_hv.size)
            hdc_sim = hv_similarity(
                state1.query_hv[:min_size], 
                state2.query_hv[:min_size],
                min_size
            )
        else:
            hdc_sim = 0.0
        
        # Secondary: Feature overlap
        features1 = set(state1.query_features)
        features2 = set(state2.query_features)
        if features1 or features2:
            feature_overlap = len(features1.intersection(features2)) / max(
                len(features1.union(features2)), 1
            )
        else:
            feature_overlap = 0.0
        
        # Tertiary: Goal similarity
        goals1 = {g.goal_description for g in state1.goal_states}
        goals2 = {g.goal_description for g in state2.goal_states}
        if goals1 or goals2:
            goal_overlap = len(goals1.intersection(goals2)) / max(
                len(goals1.union(goals2)), 1
            )
        else:
            goal_overlap = 0.0
        
        # Weighted combination
        similarity = (
            0.6 * hdc_sim + 
            0.3 * feature_overlap + 
            0.1 * goal_overlap
        )
        
        return float(similarity)

    def _infer_completion_type(self, transition: StateTransition) -> str:
        """
        Infer the type of completion from transition characteristics.
        
        Types: "direct", "elaboration", "correction", "explanation"
        """
        partial = transition.partial_state
        complete = transition.complete_state
        
        # Check if there were conflicts in partial that were resolved
        if len(partial.conflict_markers) > len(complete.conflict_markers):
            return "correction"
        
        # Check if goals were unsatisfied and became satisfied
        partial_satisfaction = sum(g.satisfaction_level for g in partial.goal_states) / max(
            len(partial.goal_states), 1
        )
        complete_satisfaction = sum(g.satisfaction_level for g in complete.goal_states) / max(
            len(complete.goal_states), 1
        )
        
        if complete_satisfaction > partial_satisfaction + 0.5:
            # High satisfaction increase suggests direct completion
            return "direct"
        
        # Check if many new features were added (elaboration)
        if len(transition.completion_features) > len(partial.query_features) * 0.5:
            return "elaboration"
        
        # Default to explanation
        return "explanation"

    def _compute_transition_reliability(self, transition: StateTransition) -> float:
        """
        Compute reliability score for a transition based on usage statistics.
        
        Higher relevance_count indicates the transition was frequently useful.
        """
        # Base reliability on relevance count, normalized
        # Use log scaling to prevent excessive influence of very high counts
        import math
        
        if transition.relevance_count == 0:
            return 0.5  # Neutral reliability for unused transitions
        
        # Log-scaled reliability: ranges from ~0.5 to ~1.0
        reliability = 0.5 + 0.5 * min(1.0, math.log1p(transition.relevance_count) / 5.0)
        
        return reliability

    # =========================================================================
    # Token-level transition storage and retrieval
    # =========================================================================

    def store_token_transition(
        self,
        context_tokens: List[str],
        next_token: str,
        modality: str = "code",
        domain: str = "code",
    ) -> None:
        """
        Store a token-level transition: context → next_token.
        
        This is the core of LLM-style token-by-token generation, but using
        pattern counting instead of neural weights.
        
        Storage format:
        - Key: HDC bundle of context_tokens[-8:] (last 8 tokens)
        - Value: Counter of possible next tokens with frequencies
        
        Args:
            context_tokens: Previous token sequence (use last 8 for context)
            next_token: The token that followed this context
            modality: Content type (default: "code")
            domain: Domain (default: "code")
        """
        if not hasattr(self, "_token_transitions"):
            # Initialize token transition storage
            # Format: {context_key: Counter({token: count})}
            self._token_transitions: Dict[str, Dict[str, int]] = {}
        
        # Use last 8 tokens as context window
        context_window = context_tokens[-8:] if len(context_tokens) > 8 else context_tokens
        
        # Create stable key from context
        context_key = "|".join(context_window)
        
        # Store transition
        if context_key not in self._token_transitions:
            self._token_transitions[context_key] = {}
        
        if next_token not in self._token_transitions[context_key]:
            self._token_transitions[context_key][next_token] = 0
        
        self._token_transitions[context_key][next_token] += 1

    def find_next_token(
        self,
        context_tokens: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find most likely next tokens given context.
        
        Returns patterns ranked by frequency (not probability - we don't store weights!).
        
        Args:
            context_tokens: Current token sequence
            top_k: Number of top candidates to return
            
        Returns:
            List of (token, score) tuples, sorted by score descending
        """
        if not hasattr(self, "_token_transitions"):
            return []
        
        # Try exact context match first
        context_window = context_tokens[-8:] if len(context_tokens) > 8 else context_tokens
        context_key = "|".join(context_window)
        
        if context_key in self._token_transitions:
            # Exact match found
            token_counts = self._token_transitions[context_key]
            
            # Rank by count (frequency)
            sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Normalize counts to scores (0-1 range, but NOT probabilities)
            max_count = max(token_counts.values()) if token_counts else 1
            results = [(tok, count / max_count) for tok, count in sorted_tokens[:top_k]]
            
            return results
        
        # Fallback: try shorter contexts (backoff)
        for context_len in range(len(context_window) - 1, 0, -1):
            shorter_context = context_window[-context_len:]
            shorter_key = "|".join(shorter_context)
            
            if shorter_key in self._token_transitions:
                token_counts = self._token_transitions[shorter_key]
                sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
                max_count = max(token_counts.values()) if token_counts else 1
                results = [(tok, count / max_count * 0.5) for tok, count in sorted_tokens[:top_k]]
                return results
        
        return []

    def get_token_transition_stats(self) -> Dict[str, int]:
        """
        Get statistics about stored token transitions.
        
        Returns:
            Dictionary with counts of contexts, total transitions, etc.
        """
        if not hasattr(self, "_token_transitions"):
            return {
                "total_contexts": 0,
                "total_transitions": 0,
                "avg_fanout": 0.0,
            }
        
        total_contexts = len(self._token_transitions)
        total_transitions = sum(sum(counts.values()) for counts in self._token_transitions.values())
        avg_fanout = total_transitions / max(1, total_contexts)
        
        return {
            "total_contexts": total_contexts,
            "total_transitions": total_transitions,
            "avg_fanout": round(avg_fanout, 2),
        }


__all__ = ["TransitionMemoryLayer"]
