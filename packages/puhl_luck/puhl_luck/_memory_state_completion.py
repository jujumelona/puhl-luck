"""
StateCompletion: Universal state completion across domains.

This module implements domain-agnostic state completion using the
field-based memory architecture. The same completion algorithm works
for conversation, code, documents, and reasoning.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from ._memory_field_core import StateField, Candidate, CandidateSource, InputContext
from ._brain_hdc import feature_hv, hv_similarity

# Try to import Rust implementation
try:
    from puhl_luck_core import (
        complete_state_rust,
        merge_completions_rust,
        rank_completions_rust
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


@dataclass
class CompletionConfig:
    """Configuration for state completion."""
    max_candidates: int = 5
    min_confidence: float = 0.1
    merge_strategy: str = "weighted"  # "union", "intersection", "weighted"
    quality_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.quality_metrics is None:
            self.quality_metrics = {
                "coherence": 0.4,
                "completeness": 0.3,
                "relevance": 0.3
            }


class StateCompletion:
    """
    Universal state completion system.
    
    Completes incomplete states using learned patterns from memory.
    Works across domains: conversation, code, documents, reasoning.
    """
    
    def __init__(self, config: Optional[CompletionConfig] = None):
        """Initialize StateCompletion.
        
        Args:
            config: Configuration for completion behavior
        """
        self.config = config or CompletionConfig()
        self._completion_patterns: List[Tuple[List[str], List[str], float]] = []
    
    def add_completion_pattern(
        self,
        pattern_features: List[str],
        completion_features: List[str],
        confidence: float
    ):
        """Add a learned completion pattern.
        
        Args:
            pattern_features: Features that trigger this pattern
            completion_features: Features that complete the pattern
            confidence: Confidence in this pattern (0-1)
        """
        self._completion_patterns.append((
            pattern_features,
            completion_features,
            confidence
        ))
    
    def complete_state(
        self,
        incomplete_field: StateField,
        context: Optional[InputContext] = None
    ) -> List[Candidate]:
        """
        Complete an incomplete state field.
        
        This is the universal completion method that works across all domains.
        The only difference between domains is the surface representation.
        
        Args:
            incomplete_field: The incomplete state to complete
            context: Optional context for completion
        
        Returns:
            List of completion candidates sorted by confidence
        """
        # Extract features from incomplete field
        incomplete_features = self._extract_field_features(incomplete_field)
        
        # Extract context features
        context_features = []
        if context and context.modality:
            context_features.append(f"type:{context.modality}")
        if context and context.domain:
            context_features.append(f"domain:{context.domain}")
        
        # Add activated memory features from all activation dictionaries
        for mem_id, activation in incomplete_field.activated_events.items():
            if activation > 0.3:
                context_features.append(f"mem:{mem_id}")
        for concept_id, activation in incomplete_field.activated_concepts.items():
            if activation > 0.3:
                context_features.append(f"concept:{concept_id}")
        for op_id, activation in incomplete_field.activated_operators.items():
            if activation > 0.3:
                context_features.append(f"operator:{op_id}")
        
        # Use Rust implementation if available
        if RUST_AVAILABLE:
            try:
                results = complete_state_rust(
                    incomplete_features,
                    context_features,
                    self._completion_patterns,
                    self.config.max_candidates
                )
                
                # Convert to Candidate objects
                candidates = []
                for completion_features, confidence in results:
                    if confidence >= self.config.min_confidence:
                        candidate = self._create_candidate(
                            completion_features,
                            confidence,
                            CandidateSource.TRANSITION_BASED
                        )
                        candidates.append(candidate)
                
                # Only return Rust results if non-empty; otherwise fall through to Python
                if candidates:
                    return candidates
                
            except Exception:
                pass
        
        # Python fallback implementation
        return self._complete_state_python(
            incomplete_features,
            context_features,
            incomplete_field
        )
    
    def _complete_state_python(
        self,
        incomplete_features: List[str],
        context_features: List[str],
        field: StateField
    ) -> List[Candidate]:
        """Python fallback for state completion."""
        # Extract simple tokens from features for matching
        incomplete_tokens = set()
        for feat in incomplete_features:
            # Extract tokens after ":" if present
            if ":" in feat:
                incomplete_tokens.add(feat.split(":", 1)[1])
            else:
                incomplete_tokens.add(feat)
        
        context_tokens = set()
        for feat in context_features:
            if ":" in feat:
                context_tokens.add(feat.split(":", 1)[1])
            else:
                context_tokens.add(feat)
        
        scored_completions = []
        
        for pattern, completion, base_confidence in self._completion_patterns:
            pattern_set = set(pattern)
            
            if len(pattern_set) == 0:
                continue
            
            # Match against incomplete tokens
            state_match = len(incomplete_tokens & pattern_set) / len(pattern_set)
            
            # Context relevance
            context_relevance = len(context_tokens & pattern_set) / max(len(pattern_set), 1)
            
            # Combined score
            score = base_confidence * (0.7 * state_match + 0.3 * context_relevance)
            
            if score > self.config.min_confidence:
                scored_completions.append((completion, score))
        
        # Sort by score
        scored_completions.sort(key=lambda x: x[1], reverse=True)
        scored_completions = scored_completions[:self.config.max_candidates]
        
        # Convert to Candidate objects
        candidates = []
        for completion_features, confidence in scored_completions:
            candidate = self._create_candidate(
                completion_features,
                confidence,
                CandidateSource.TRANSITION_BASED
            )
            candidates.append(candidate)
        
        return candidates
    
    def merge_completions(
        self,
        candidates: List[Candidate],
        strategy: Optional[str] = None
    ) -> Candidate:
        """
        Merge multiple completion candidates.
        
        Args:
            candidates: List of candidates to merge
            strategy: Merge strategy ("union", "intersection", "weighted")
        
        Returns:
            Merged candidate
        """
        if not candidates:
            raise ValueError("Cannot merge empty candidate list")
        
        if len(candidates) == 1:
            return candidates[0]
        
        strategy = strategy or self.config.merge_strategy
        
        # Extract features and confidences
        candidate_data = [
            (self._candidate_to_features(c), c.confidence)
            for c in candidates
        ]
        
        # Use Rust implementation if available
        if RUST_AVAILABLE:
            try:
                merged_features = merge_completions_rust(
                    candidate_data,
                    strategy
                )
                
                # Calculate merged confidence
                avg_confidence = sum(c.confidence for c in candidates) / len(candidates)
                
                return self._create_candidate(
                    merged_features,
                    avg_confidence,
                    CandidateSource.TRANSITION_BASED
                )
                
            except Exception:
                pass
        
        # Python fallback
        return self._merge_completions_python(candidates, strategy)
    
    def _merge_completions_python(
        self,
        candidates: List[Candidate],
        strategy: str
    ) -> Candidate:
        """Python fallback for merging completions."""
        if strategy == "union":
            # Union of all features
            all_features = set()
            for c in candidates:
                all_features.update(self._candidate_to_features(c))
            merged_features = list(all_features)
            
        elif strategy == "intersection":
            # Intersection of all features
            feature_sets = [set(self._candidate_to_features(c)) for c in candidates]
            merged_features = list(set.intersection(*feature_sets))
            
        else:  # weighted
            # Weighted by confidence
            feature_scores = {}
            total_confidence = sum(c.confidence for c in candidates)
            
            for candidate in candidates:
                normalized_conf = candidate.confidence / total_confidence
                for feature in self._candidate_to_features(candidate):
                    feature_scores[feature] = feature_scores.get(feature, 0.0) + normalized_conf
            
            # Take features with score > threshold
            merged_features = [
                f for f, score in feature_scores.items()
                if score > 0.3
            ]
        
        # Calculate merged confidence
        avg_confidence = sum(c.confidence for c in candidates) / len(candidates)
        
        return self._create_candidate(
            merged_features,
            avg_confidence,
            CandidateSource.TRANSITION_BASED
        )
    
    def rank_completions(
        self,
        completions: List[Candidate],
        context: Optional[InputContext] = None
    ) -> List[Tuple[Candidate, float]]:
        """
        Rank completions by quality.
        
        Args:
            completions: List of completion candidates
            context: Optional context for ranking
        
        Returns:
            List of (candidate, quality_score) tuples sorted by quality
        """
        context_features = []
        if context and context.text:
            context_features = context.text.split()
        
        completion_features = [
            self._candidate_to_features(c) for c in completions
        ]
        
        # Use Rust implementation if available
        if RUST_AVAILABLE:
            try:
                ranked = rank_completions_rust(
                    completion_features,
                    context_features,
                    self.config.quality_metrics
                )
                
                # Pair with original candidates
                result = []
                for i, (_, quality) in enumerate(ranked):
                    if i < len(completions):
                        result.append((completions[i], quality))
                
                return result
                
            except Exception:
                pass
        
        # Python fallback
        return self._rank_completions_python(completions, context_features)
    
    def _rank_completions_python(
        self,
        completions: List[Candidate],
        context_features: List[str]
    ) -> List[Tuple[Candidate, float]]:
        """Python fallback for ranking completions."""
        context_set = set(context_features)
        
        scored = []
        for candidate in completions:
            features = self._candidate_to_features(candidate)
            feature_set = set(features)
            
            # Coherence: internal consistency
            coherence = 1.0 if len(features) == 0 else \
                1.0 - (len(features) - len(feature_set)) / len(features)
            
            # Completeness: sufficient information
            completeness = min(len(features) / 10.0, 1.0)
            
            # Relevance: match with context
            relevance = 0.0
            if len(context_set) > 0:
                relevance = len(context_set & feature_set) / len(context_set)
            
            # Combined quality
            metrics = self.config.quality_metrics
            quality = (
                metrics["coherence"] * coherence +
                metrics["completeness"] * completeness +
                metrics["relevance"] * relevance
            )
            
            scored.append((candidate, quality))
        
        # Sort by quality
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _extract_field_features(self, field: StateField) -> List[str]:
        """Extract feature strings from a state field."""
        features = []
        
        # Add activation features from all three activation dictionaries
        for mem_id, activation in field.activated_events.items():
            if activation > 0.3:
                features.append(f"active:{mem_id}")
        for concept_id, activation in field.activated_concepts.items():
            if activation > 0.3:
                features.append(f"concept:{concept_id}")
        for op_id, activation in field.activated_operators.items():
            if activation > 0.3:
                features.append(f"operator:{op_id}")
        
        # Add conflict markers
        for conflict in field.conflict_markers:
            features.append(f"conflict:{conflict.conflict_type.value}")
        
        # Add goal features
        for goal in field.goal_states:
            features.append(f"goal:{goal.goal_description}")
        
        return features
    
    def _candidate_to_features(self, candidate: Candidate) -> List[str]:
        """Convert a candidate to feature list."""
        features = []
        
        # Add candidate content as features
        if candidate.content:
            features.append(f"content:{candidate.content}")
        
        # Add tokens
        for token in candidate.tokens:
            features.append(f"token:{token}")
        
        # Add source information
        for op in candidate.source_operators:
            features.append(f"operator:{op}")
        for trans in candidate.source_transitions:
            features.append(f"transition:{trans}")
        
        return features
    
    def _create_candidate(
        self,
        features: List[str],
        confidence: float,
        source: CandidateSource
    ) -> Candidate:
        """Create a Candidate object from features."""
        # Extract content: prefer "content:" prefixed features, fall back to raw tokens
        content_features = [f.split(":", 1)[1] for f in features if f.startswith("content:")]
        if content_features:
            content = " ".join(content_features)
        else:
            # features may be plain tokens (e.g. ["response", "answer"]) — use them directly
            plain = [f for f in features if ":" not in f]
            content = " ".join(plain) if plain else " ".join(
                f.split(":", 1)[1] for f in features if ":" in f
            )
        
        # Tokenize content
        tokens = content.split() if content else []
        
        return Candidate(
            content=content,
            tokens=tokens,
            energy_reduction=0.0,  # Will be computed later
            predicted_energy_after=0.0,  # Will be computed later
            source=source,
            source_operators=[],
            source_transitions=[],
            tensions_addressed=[],
            tensions_resolved_count=0,
            confidence=confidence
        )

