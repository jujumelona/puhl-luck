"""
Universal Surface Sequence Storage

THE ONLY place where raw target text is preserved.

Key principle:
- expose_pair(input, target) → store target as SurfaceSequence
- generate(query) → retrieve relevant SurfaceSequences → return raw text
- NO mixing with features, transitions, or operators in output

Storage is task-agnostic:
- Code: "def add(a,b): return a+b"
- Classification: "IT과학", "entailment", "positive"
- QA: "정답 텍스트", "answer span"
- Generation: "전체 답변 문장"

All are just surface sequences.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


def stable_hash(text: str) -> str:
    """Generate stable 16-char hash."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


@dataclass
class SurfaceSequence:
    """
    A raw surface sequence (target output text).
    
    This is what the system should output verbatim when queried
    with a matching input.
    """
    
    sequence_id: str  # stable hash
    raw_text: str  # NEVER MODIFIED - the actual output
    tokens: List[str]  # tokenized target (for token transitions)
    source_input_hash: str  # hash of input that produced this
    input_features: List[str]  # features from input (for retrieval)
    input_tokens: List[str]  # tokens from input (for retrieval)
    retrieval_count: int = 0
    successful_output: bool = True  # marked False if user corrects it


class SurfaceSequenceStorage:
    """
    Universal storage for all target outputs.
    
    Stores raw text indexed by INPUT characteristics for retrieval:
    - input_feature_to_sequences: query features → target sequences
    - input_token_to_sequences: query tokens → target sequences  
    - target_token_to_sequences: target tokens → sequences (auxiliary)
    """
    
    def __init__(self):
        # Core storage
        self.sequences: Dict[str, SurfaceSequence] = {}
        
        # PRIMARY: Index by INPUT characteristics (for query→target retrieval)
        self.input_to_sequences: Dict[str, List[str]] = {}  # input_hash → seq_ids
        self.input_feature_to_sequences: Dict[str, Set[str]] = {}  # input_feature → seq_ids
        self.input_token_to_sequences: Dict[str, Set[str]] = {}  # input_token → seq_ids
        
        # AUXILIARY: Index by TARGET tokens (for similarity/fallback)
        self.target_token_to_sequences: Dict[str, Set[str]] = {}  # target_token → seq_ids
        
        # Stats
        self.total_stored = 0
        self.retrieval_stats = {
            "exact_hash": 0,
            "input_feature": 0,
            "input_token": 0,
            "target_token": 0,
            "fallback": 0,
            "empty": 0,
        }
        
    def store_sequence(
        self,
        raw_text: str,
        tokens: List[str],
        source_input: str,
        input_features: Optional[List[str]] = None,
        input_tokens: Optional[List[str]] = None,
    ) -> str:
        """
        Store a surface sequence with INPUT-based indexing.
        
        Args:
            raw_text: The target output text (stored as-is)
            tokens: Tokenized target (for token transitions)
            source_input: The input text that produced this output
            input_features: Features from input (for feature-based retrieval)
            input_tokens: Tokens from input (for token-based retrieval)
            
        Returns:
            sequence_id: Unique ID for this sequence
        """
        # Generate IDs
        sequence_id = stable_hash(raw_text)
        input_hash = stable_hash(source_input)
        
        # Check if already stored
        if sequence_id in self.sequences:
            existing = self.sequences[sequence_id]
            existing.retrieval_count += 1
            return sequence_id
        
        # Create sequence
        seq = SurfaceSequence(
            sequence_id=sequence_id,
            raw_text=raw_text,
            tokens=tokens,
            source_input_hash=input_hash,
            input_features=input_features or [],
            input_tokens=input_tokens or [],
        )
        
        # Store
        self.sequences[sequence_id] = seq
        
        # Index by exact input hash
        if input_hash not in self.input_to_sequences:
            self.input_to_sequences[input_hash] = []
        self.input_to_sequences[input_hash].append(sequence_id)
        
        # INDEX 1: INPUT features → target sequence
        if input_features:
            for feature in input_features[:20]:  # Use top 20 input features
                if feature not in self.input_feature_to_sequences:
                    self.input_feature_to_sequences[feature] = set()
                self.input_feature_to_sequences[feature].add(sequence_id)
        
        # INDEX 2: INPUT tokens → target sequence
        if input_tokens:
            for token in input_tokens[:15]:  # Use top 15 input tokens
                if token not in self.input_token_to_sequences:
                    self.input_token_to_sequences[token] = set()
                self.input_token_to_sequences[token].add(sequence_id)
        
        # INDEX 3 (auxiliary): TARGET tokens → sequence (for similarity)
        for token in tokens[:10]:  # Use first 10 target tokens
            if token not in self.target_token_to_sequences:
                self.target_token_to_sequences[token] = set()
            self.target_token_to_sequences[token].add(sequence_id)
        
        self.total_stored += 1
        
        return sequence_id
    
    def retrieve_by_input(self, input_text: str, top_k: int = 10) -> List[SurfaceSequence]:
        """
        Retrieve sequences by exact input hash match.
        
        Args:
            input_text: Query input
            top_k: Max sequences to return
            
        Returns:
            List of matching SurfaceSequence objects
        """
        input_hash = stable_hash(input_text)
        
        # Exact match
        seq_ids = self.input_to_sequences.get(input_hash, [])
        
        if seq_ids:
            self.retrieval_stats["exact_hash"] += 1
        
        results = []
        for seq_id in seq_ids[:top_k]:
            seq = self.sequences.get(seq_id)
            if seq:
                results.append(seq)
        
        return results
    
    def retrieve_by_input_features(
        self, 
        query_features: List[str], 
        top_k: int = 10,
        min_overlap: int = 2,
    ) -> List[Tuple[SurfaceSequence, float]]:
        """
        Retrieve sequences by INPUT feature overlap.
        
        PRIMARY retrieval method: matches query features to input features.
        
        Args:
            query_features: Features from query
            top_k: Max to return
            min_overlap: Minimum feature overlap required
            
        Returns:
            List of (sequence, overlap_score) tuples
        """
        # Find sequences with input feature overlap
        candidate_ids: Dict[str, int] = {}
        
        for feature in query_features[:30]:  # Use top 30 query features
            seq_ids = self.input_feature_to_sequences.get(feature, set())
            for seq_id in seq_ids:
                candidate_ids[seq_id] = candidate_ids.get(seq_id, 0) + 1
        
        # Filter by minimum overlap
        candidate_ids = {k: v for k, v in candidate_ids.items() if v >= min_overlap}
        
        if candidate_ids:
            self.retrieval_stats["input_feature"] += 1
        
        # Score by overlap ratio
        results = []
        for seq_id, overlap_count in candidate_ids.items():
            seq = self.sequences.get(seq_id)
            if seq:
                # Score: overlap / min(query_features, stored_input_features)
                denom = min(len(query_features), len(seq.input_features), 30)
                score = overlap_count / max(denom, 1)
                results.append((seq, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def retrieve_by_input_tokens(
        self, 
        query_tokens: List[str], 
        top_k: int = 10,
        min_overlap: int = 2,
    ) -> List[Tuple[SurfaceSequence, float]]:
        """
        Retrieve sequences by INPUT token overlap.
        
        SECONDARY retrieval method: matches query tokens to input tokens.
        
        Args:
            query_tokens: Tokens from query
            top_k: Max to return
            min_overlap: Minimum token overlap required
            
        Returns:
            List of (sequence, overlap_score) tuples
        """
        # Find sequences with input token overlap
        candidate_ids: Dict[str, int] = {}
        
        for token in query_tokens[:20]:  # Use top 20 query tokens
            seq_ids = self.input_token_to_sequences.get(token, set())
            for seq_id in seq_ids:
                candidate_ids[seq_id] = candidate_ids.get(seq_id, 0) + 1
        
        # Filter by minimum overlap
        candidate_ids = {k: v for k, v in candidate_ids.items() if v >= min_overlap}
        
        if candidate_ids:
            self.retrieval_stats["input_token"] += 1
        
        # Score by overlap
        results = []
        for seq_id, overlap_count in candidate_ids.items():
            seq = self.sequences.get(seq_id)
            if seq:
                # Score: overlap / min(query_len, stored_len)
                denom = min(len(query_tokens), len(seq.input_tokens), 20)
                score = overlap_count / max(denom, 1)
                results.append((seq, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def retrieve_by_target_tokens(
        self, 
        query_tokens: List[str], 
        top_k: int = 10,
    ) -> List[Tuple[SurfaceSequence, float]]:
        """
        Retrieve sequences by TARGET token overlap (auxiliary method).
        
        Used only when input-based retrieval fails.
        
        Args:
            query_tokens: Tokens from query
            top_k: Max to return
            
        Returns:
            List of (sequence, overlap_score) tuples
        """
        # Find sequences with target token overlap
        candidate_ids: Dict[str, int] = {}
        
        for token in query_tokens[:10]:  # Use first 10 tokens
            seq_ids = self.target_token_to_sequences.get(token, set())
            for seq_id in seq_ids:
                candidate_ids[seq_id] = candidate_ids.get(seq_id, 0) + 1
        
        if candidate_ids:
            self.retrieval_stats["target_token"] += 1
        
        # Score by overlap
        results = []
        for seq_id, overlap_count in candidate_ids.items():
            seq = self.sequences.get(seq_id)
            if seq:
                score = overlap_count / max(len(query_tokens), len(seq.tokens), 1)
                results.append((seq, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def retrieve_all(self, top_k: int = 100) -> List[SurfaceSequence]:
        """Retrieve most frequently used sequences."""
        seqs = list(self.sequences.values())
        seqs.sort(key=lambda s: s.retrieval_count, reverse=True)
        return seqs[:top_k]
    
    def mark_retrieved(self, sequence_id: str) -> None:
        """Mark sequence as retrieved (for statistics)."""
        seq = self.sequences.get(sequence_id)
        if seq:
            seq.retrieval_count += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get storage statistics."""
        return {
            "total_sequences": len(self.sequences),
            "unique_inputs": len(self.input_to_sequences),
            "indexed_input_features": len(self.input_feature_to_sequences),
            "indexed_input_tokens": len(self.input_token_to_sequences),
            "indexed_target_tokens": len(self.target_token_to_sequences),
            **self.retrieval_stats,
        }


__all__ = ["SurfaceSequence", "SurfaceSequenceStorage"]
