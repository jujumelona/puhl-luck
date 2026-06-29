"""
Tests for StateCompletion: Universal state completion.
"""

import pytest
import numpy as np
from puhl_luck._memory_state_completion import StateCompletion, CompletionConfig
from puhl_luck._memory_field_core import (
    StateField, InputContext, Candidate, CandidateSource,
    GoalState, ConflictMarker, ConflictType
)


class TestStateCompletion:
    """Test universal state completion."""
    
    def test_initialization(self):
        """Test StateCompletion initialization."""
        completion = StateCompletion()
        assert completion.config is not None
        assert completion.config.max_candidates == 5
        assert completion.config.min_confidence == 0.1
    
    def test_custom_config(self):
        """Test StateCompletion with custom config."""
        config = CompletionConfig(
            max_candidates=10,
            min_confidence=0.2,
            merge_strategy="union"
        )
        completion = StateCompletion(config)
        assert completion.config.max_candidates == 10
        assert completion.config.min_confidence == 0.2
        assert completion.config.merge_strategy == "union"
    
    def test_add_completion_pattern(self):
        """Test adding completion patterns."""
        completion = StateCompletion()
        
        completion.add_completion_pattern(
            pattern_features=["user", "question"],
            completion_features=["response", "answer"],
            confidence=0.9
        )
        
        assert len(completion._completion_patterns) == 1
        pattern, comp, conf = completion._completion_patterns[0]
        assert "user" in pattern
        assert "response" in comp
        assert conf == 0.9
    
    def test_complete_state_basic(self):
        """Test basic state completion."""
        completion = StateCompletion()
        
        # Add patterns
        completion.add_completion_pattern(
            pattern_features=["user", "question"],
            completion_features=["response"],
            confidence=0.9
        )
        completion.add_completion_pattern(
            pattern_features=["query"],
            completion_features=["answer"],
            confidence=0.8
        )
        
        # Create incomplete field
        field = StateField(
            query_features=["user", "question"],
            query_hv=np.array([], dtype=np.uint64),
            activated_events={"user_q": 0.8, "query": 0.5},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[]
        )
        
        # Complete state
        candidates = completion.complete_state(field)
        
        assert len(candidates) > 0
        assert all(isinstance(c, Candidate) for c in candidates)
        assert all(c.confidence >= completion.config.min_confidence for c in candidates)
    
    def test_complete_state_with_context(self):
        """Test state completion with context."""
        completion = StateCompletion()
        
        completion.add_completion_pattern(
            pattern_features=["text", "chat"],
            completion_features=["response"],
            confidence=0.9
        )
        
        field = StateField(
            query_features=["conversation"],
            query_hv=np.array([], dtype=np.uint64),
            activated_events={"mem1": 0.7},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[]
        )
        
        context = InputContext(
            query_text="Hello",
            modality="text",
            domain="chat"
        )
        
        candidates = completion.complete_state(field, context)
        
        # Should match context-aware pattern
        assert len(candidates) > 0
    
    def test_complete_state_filters_low_confidence(self):
        """Test that low confidence completions are filtered."""
        config = CompletionConfig(min_confidence=0.7)
        completion = StateCompletion(config)
        
        # Add low confidence pattern
        completion.add_completion_pattern(
            pattern_features=["rare"],
            completion_features=["unlikely"],
            confidence=0.3
        )
        
        field = StateField(
            query_features=["rare"],
            query_hv=np.array([], dtype=np.uint64),
            activated_events={"rare": 0.9},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[]
        )
        
        candidates = completion.complete_state(field)
        
        # Should filter out low confidence
        assert all(c.confidence >= 0.7 for c in candidates)
    
    def test_complete_state_limits_candidates(self):
        """Test max candidates limit."""
        config = CompletionConfig(max_candidates=3)
        completion = StateCompletion(config)
        
        # Add many patterns
        for i in range(10):
            completion.add_completion_pattern(
                pattern_features=[f"pattern{i}"],
                completion_features=[f"completion{i}"],
                confidence=0.5 + i * 0.05
            )
        
        field = StateField(
            query_features=[f"pattern{i}" for i in range(10)],
            query_hv=np.array([], dtype=np.uint64),
            activated_events={f"pattern{i}": 0.9 for i in range(10)},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[]
        )
        
        candidates = completion.complete_state(field)
        
        assert len(candidates) <= 3
    
    def test_merge_completions_union(self):
        """Test merging completions with union strategy."""
        completion = StateCompletion()
        
        candidates = [
            Candidate(
                content="a b",
                confidence=0.9,
                energy_reduction=1.0,
                source=CandidateSource.TRANSITION_BASED,
                tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
            ),
            Candidate(
                content="b c",
                confidence=0.8,
                energy_reduction=1.0,
                source=CandidateSource.TRANSITION_BASED,
                tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
            )
        ]
        
        merged = completion.merge_completions(candidates, strategy="union")
        
        assert merged is not None
        assert isinstance(merged, Candidate)
        assert 0.8 <= merged.confidence <= 0.9
    
    def test_merge_completions_intersection(self):
        """Test merging completions with intersection strategy."""
        completion = StateCompletion()
        
        candidates = [
            Candidate(
                content="a b",
                confidence=0.9,
                energy_reduction=1.0,
                source=CandidateSource.TRANSITION_BASED,
                tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
            ),
            Candidate(
                content="b c",
                confidence=0.8,
                energy_reduction=1.0,
                source=CandidateSource.TRANSITION_BASED,
                tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
            )
        ]
        
        merged = completion.merge_completions(candidates, strategy="intersection")
        
        assert merged is not None
        assert isinstance(merged, Candidate)
    
    def test_merge_completions_weighted(self):
        """Test merging completions with weighted strategy."""
        completion = StateCompletion()
        
        candidates = [
            Candidate(
                content="high",
                confidence=0.9,
                energy_reduction=1.0,
                source=CandidateSource.TRANSITION_BASED,
                tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
            ),
            Candidate(
                content="low",
                confidence=0.3,
                energy_reduction=0.5,
                source=CandidateSource.TRANSITION_BASED,
                tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
            )
        ]
        
        merged = completion.merge_completions(candidates, strategy="weighted")
        
        assert merged is not None
        # Should favor higher confidence
        assert merged.confidence > 0.5
    
    def test_merge_single_candidate(self):
        """Test merging single candidate returns itself."""
        completion = StateCompletion()
        
        candidate = Candidate(
            content="single",
            confidence=0.9,
            energy_reduction=1.0,
            source=CandidateSource.TRANSITION_BASED,
            tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
        )
        
        merged = completion.merge_completions([candidate])
        
        assert merged is candidate
    
    def test_merge_empty_raises_error(self):
        """Test merging empty list raises error."""
        completion = StateCompletion()
        
        with pytest.raises(ValueError, match="Cannot merge empty"):
            completion.merge_completions([])
    
    def test_rank_completions(self):
        """Test ranking completions by quality."""
        completion = StateCompletion()
        
        candidates = [
            Candidate(
                content="short",
                confidence=0.9,
                energy_reduction=1.0,
                source=CandidateSource.TRANSITION_BASED,
                tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
            ),
            Candidate(
                content="longer completion with more content",
                confidence=0.8,
                energy_reduction=1.0,
                source=CandidateSource.TRANSITION_BASED,
                tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
            ),
            Candidate(
                content="duplicate duplicate duplicate",
                confidence=0.7,
                energy_reduction=1.0,
                source=CandidateSource.TRANSITION_BASED,
                tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
            )
        ]
        
        ranked = completion.rank_completions(candidates)
        
        assert len(ranked) == 3
        assert all(isinstance(r[0], Candidate) for r in ranked)
        assert all(isinstance(r[1], float) for r in ranked)
        # Should be sorted by quality
        assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]
    
    def test_rank_with_context(self):
        """Test ranking with context relevance."""
        completion = StateCompletion()
        
        candidates = [
            Candidate(
                content="relevant context match",
                confidence=0.8,
                energy_reduction=1.0,
                source=CandidateSource.TRANSITION_BASED,
                tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
            ),
            Candidate(
                content="unrelated different topic",
                confidence=0.8,
                energy_reduction=1.0,
                source=CandidateSource.TRANSITION_BASED,
                tokens=[], source_operators=[], source_transitions=[], tensions_addressed=[], tensions_resolved_count=0, predicted_energy_after=0.0
            )
        ]
        
        context = InputContext(
            query_text="relevant context",
            modality="text"
        )
        
        ranked = completion.rank_completions(candidates, context)
        
        # First should have similar or higher quality due to context match
        # (May be equal if both have same relevance score)
        assert ranked[0][1] >= ranked[1][1]
    
    def test_universal_algorithm_conversation(self):
        """Test universal completion for conversation domain."""
        completion = StateCompletion()
        
        completion.add_completion_pattern(
            pattern_features=["user", "greeting"],
            completion_features=["response", "greeting"],
            confidence=0.9
        )
        
        field = StateField(
            query_features=["user", "greeting"],
            query_hv=np.array([], dtype=np.uint64),
            activated_events={"user": 0.9, "greeting": 0.8},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[]
        )
        
        context = InputContext(
            query_text="Hello",
            modality="text",
            domain="chat",
            metadata={"context_type": "conversation"}
        )
        
        candidates = completion.complete_state(field, context)
        
        assert len(candidates) > 0
        assert all(c.source == CandidateSource.TRANSITION_BASED for c in candidates)
    
    def test_universal_algorithm_code(self):
        """Test universal completion for code domain."""
        completion = StateCompletion()
        
        completion.add_completion_pattern(
            pattern_features=["function", "incomplete"],
            completion_features=["return", "statement"],
            confidence=0.9
        )
        
        field = StateField(
            query_features=["function", "incomplete"],
            query_hv=np.array([], dtype=np.uint64),
            activated_events={"function": 0.9, "incomplete": 0.8},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[]
        )
        
        context = InputContext(
            query_text="def func():",
            modality="code",
            domain="python"
        )
        
        candidates = completion.complete_state(field, context)
        
        # Same algorithm, different domain
        assert len(candidates) > 0
        assert all(c.source == CandidateSource.TRANSITION_BASED for c in candidates)
    
    def test_extract_field_features(self):
        """Test extracting features from state field."""
        completion = StateCompletion()
        
        field = StateField(
            query_features=["test"],
            query_hv=np.array([], dtype=np.uint64),
            activated_events={"mem1": 0.8, "mem2": 0.2},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[
                ConflictMarker(
                    conflict_id="c1",
                    conflict_type=ConflictType.CONTRADICTION,
                    involved_memories=["mem1", "mem2"],
                    strength=0.7,
                    description="test conflict"
                )
            ],
            goal_states=[
                GoalState(
                    goal_id="g1",
                    goal_description="test goal",
                    satisfaction_level=0.5,
                    constraints=[]
                )
            ],
            partial_outputs=[]
        )
        
        features = completion._extract_field_features(field)
        
        # Should include high activation, conflict, and goal
        assert any("active:mem1" in f for f in features)
        assert any("conflict:" in f for f in features)
        assert any("goal:" in f for f in features)
        # Should not include low activation
        assert not any("mem2" in f for f in features)
    
    def test_candidate_to_features(self):
        """Test converting candidate to features."""
        completion = StateCompletion()
        
        candidate = Candidate(
            content="test content",
            tokens=["test", "content"],
            confidence=0.9,
            energy_reduction=1.0,
            predicted_energy_after=0.0,
            source=CandidateSource.OPERATOR_BASED,
            source_operators=["op1"],
            source_transitions=["trans1"],
            tensions_addressed=[],
            tensions_resolved_count=0
        )
        
        features = completion._candidate_to_features(candidate)
        
        assert "content:test content" in features
        assert any("token:" in f for f in features)
        assert any("operator:op1" in f for f in features)
    
    def test_create_candidate_from_features(self):
        """Test creating candidate from features."""
        completion = StateCompletion()
        
        features = [
            "content:hello world"
        ]
        
        candidate = completion._create_candidate(
            features,
            confidence=0.8,
            source=CandidateSource.TRANSITION_BASED
        )
        
        assert candidate.content == "hello world"
        assert candidate.confidence == 0.8
        assert candidate.source == CandidateSource.TRANSITION_BASED
        assert candidate.tokens == ["hello", "world"]


class TestCompletionConfig:
    """Test CompletionConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CompletionConfig()
        
        assert config.max_candidates == 5
        assert config.min_confidence == 0.1
        assert config.merge_strategy == "weighted"
        assert config.quality_metrics is not None
        assert "coherence" in config.quality_metrics
        assert "completeness" in config.quality_metrics
        assert "relevance" in config.quality_metrics
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CompletionConfig(
            max_candidates=10,
            min_confidence=0.5,
            merge_strategy="union",
            quality_metrics={"custom": 1.0}
        )
        
        assert config.max_candidates == 10
        assert config.min_confidence == 0.5
        assert config.merge_strategy == "union"
        assert config.quality_metrics == {"custom": 1.0}
    
    def test_config_post_init(self):
        """Test post-init sets default quality metrics."""
        config = CompletionConfig(max_candidates=7)
        
        # Should have default quality metrics
        assert config.quality_metrics is not None
        assert sum(config.quality_metrics.values()) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


