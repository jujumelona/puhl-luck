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
    tokens: List[str]  # tokenized (for token transitions)
    source_input_hash: str  # hash of input that produced this
    retrieval_count: int = 0
    successful_output: bool = True  # marked False if user corrects it


class SurfaceSequenceStorage:
    """
    Universal storage for all target outputs.
    
    Stores raw text + tokens, indexed by input hash for fast retrieval.
    """
    
    def __init__(self):
        # Core storage
        self.sequences: Dict[str, SurfaceSequence] = {}
        
        # Indexing: input_hash → list of sequence_ids
        self.input_to_sequences: Dict[str, List[str]] = {}
        
        # Token-based index for partial matching
        self.token_to_sequences: Dict[str, Set[str]] = {}
        
        # Stats
        self.total_stored = 0
        
    def store_sequence(
        self,
        raw_text: str,
        tokens: List[str],
        source_input: str,
    ) -> str:
        """
        Store a surface sequence.
        
        Args:
            raw_text: The target output text (stored as-is)
            tokens: Tokenized form
            source_input: The input text that produced this output
            
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
        )
        
        # Store
        self.sequences[sequence_id] = seq
        
        # Index by input
        if input_hash not in self.input_to_sequences:
            self.input_to_sequences[input_hash] = []
        self.input_to_sequences[input_hash].append(sequence_id)
        
        # Index by tokens (first 5 tokens for quick lookup)
        for token in tokens[:5]:
            if token not in self.token_to_sequences:
                self.token_to_sequences[token] = set()
            self.token_to_sequences[token].add(sequence_id)
        
        self.total_stored += 1
        
        return sequence_id
    
    def retrieve_by_input(self, input_text: str, top_k: int = 10) -> List[SurfaceSequence]:
        """
        Retrieve sequences that were learned from this input.
        
        Args:
            input_text: Query input
            top_k: Max sequences to return
            
        Returns:
            List of matching SurfaceSequence objects
        """
        input_hash = stable_hash(input_text)
        
        # Exact match
        seq_ids = self.input_to_sequences.get(input_hash, [])
        
        results = []
        for seq_id in seq_ids[:top_k]:
            seq = self.sequences.get(seq_id)
            if seq:
                results.append(seq)
        
        return results
    
    def retrieve_by_tokens(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[SurfaceSequence, float]]:
        """
        Retrieve sequences with overlapping tokens.
        
        Args:
            query_tokens: Tokens from query
            top_k: Max to return
            
        Returns:
            List of (sequence, overlap_score) tuples
        """
        # Find sequences with token overlap
        candidate_ids: Dict[str, int] = {}
        
        for token in query_tokens[:10]:  # Use first 10 tokens
            seq_ids = self.token_to_sequences.get(token, set())
            for seq_id in seq_ids:
                candidate_ids[seq_id] = candidate_ids.get(seq_id, 0) + 1
        
        # Score by overlap
        results = []
        for seq_id, overlap_count in candidate_ids.items():
            seq = self.sequences.get(seq_id)
            if seq:
                # Score: overlap / max(query_len, seq_len)
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
            "indexed_tokens": len(self.token_to_sequences),
        }


__all__ = ["SurfaceSequence", "SurfaceSequenceStorage"]
