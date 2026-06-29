"""
Operator Memory Storage

Stores operator graphs and their relationships.
NO raw text storage - only operator patterns and transitions.

Replaces _memory_surface_storage.py for operator-based generation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ._memory_operator_graph import Operator, OperatorGraph


@dataclass
class StoredGraph:
    """Stored operator graph with metadata."""
    graph: OperatorGraph
    token_sequence: List[str]
    field_signature: str
    exposure_count: int = 0
    last_activated: float = 0.0


class OperatorMemoryStorage:
    """
    Store operator graphs and their relationships.
    
    Storage structure:
    - graphs: signature → OperatorGraph
    - field_to_graph: field_signature → [graph_signatures]
    - operator_transitions: (op1, op2) → count
    - graph_to_tokens: graph_signature → [token_sequences]
    
    NO raw text storage.
    """
    
    def __init__(self):
        # Operator graphs indexed by signature
        self.graphs: Dict[str, StoredGraph] = {}
        
        # Field signature → operator graph signatures
        self.field_to_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Operator → operator transitions (edges)
        self.operator_transitions: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Operator graph → token sequence patterns
        self.graph_to_tokens: Dict[str, List[List[str]]] = defaultdict(list)
        
        # Operator type → activation count (for scoring)
        self.operator_counts: Dict[str, int] = defaultdict(int)
        
        # Statistics
        self.total_graphs_stored: int = 0
        self.total_transitions_learned: int = 0
    
    def store_graph(
        self,
        graph: OperatorGraph,
        field_signature: str,
        token_sequence: List[str],
    ) -> str:
        """
        Store operator graph and its connections.
        
        Args:
            graph: Operator graph extracted from target
            field_signature: Compact field signature from input
            token_sequence: Tokenized target sequence
            
        Returns:
            Graph signature (hash)
        """
        graph_sig = graph.serialize()
        
        # Store graph if new
        if graph_sig not in self.graphs:
            self.graphs[graph_sig] = StoredGraph(
                graph=graph,
                token_sequence=token_sequence,
                field_signature=field_signature,
                exposure_count=1,
            )
            self.total_graphs_stored += 1
        else:
            # Update exposure count
            self.graphs[graph_sig].exposure_count += 1
        
        # Link field → graph
        if graph_sig not in self.field_to_graph[field_signature]:
            self.field_to_graph[field_signature].append(graph_sig)
        
        # Store operator transitions (edges)
        for from_idx, to_idx in graph.edges:
            from_op = graph.nodes[from_idx].op_type
            to_op = graph.nodes[to_idx].op_type
            self.operator_transitions[(from_op, to_op)] += 1
            self.total_transitions_learned += 1
        
        # Update operator counts
        for op in graph.nodes:
            self.operator_counts[op.op_type] += 1
        
        # Link graph → token sequence
        self.graph_to_tokens[graph_sig].append(token_sequence)
        
        return graph_sig
    
    def retrieve_graphs_by_field(
        self,
        field_signature: str,
        max_results: int = 10,
        min_overlap: float = 0.3,
    ) -> List[Tuple[OperatorGraph, float]]:
        """
        Retrieve operator graphs activated by field signature.
        
        Uses PARTIAL MATCHING, not exact match.
        Activates graphs based on feature overlap.
        
        Args:
            field_signature: Field signature from query
            max_results: Maximum number of graphs to return
            min_overlap: Minimum overlap ratio to consider
            
        Returns:
            List of (graph, score) tuples sorted by relevance
        """
        # Parse query features from signature
        query_features = set(field_signature.split("|"))
        
        # Score all graphs by feature overlap
        scored_graphs = []
        
        for stored_field_sig, graph_sigs in self.field_to_graph.items():
            # Parse stored features
            stored_features = set(stored_field_sig.split("|"))
            
            # Compute overlap
            if not stored_features:
                continue
            
            intersection = len(query_features & stored_features)
            union = len(query_features | stored_features)
            
            if union == 0:
                continue
            
            overlap_ratio = intersection / union
            
            # Skip if below threshold
            if overlap_ratio < min_overlap:
                continue
            
            # Add graphs with overlap score
            for graph_sig in graph_sigs:
                stored = self.graphs.get(graph_sig)
                if stored:
                    # Score = overlap * frequency
                    freq_score = stored.exposure_count / (1.0 + self.total_graphs_stored)
                    final_score = overlap_ratio * (1.0 + freq_score)
                    scored_graphs.append((stored.graph, final_score))
        
        # Sort by score descending
        scored_graphs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_graphs[:max_results]
    
    def get_transition_score(self, from_op: str, to_op: str) -> float:
        """
        Get transition probability from operator to operator.
        
        Args:
            from_op: Source operator type
            to_op: Target operator type
            
        Returns:
            Transition score (0.0 to 1.0)
        """
        count = self.operator_transitions.get((from_op, to_op), 0)
        total = self.total_transitions_learned
        return count / total if total > 0 else 0.0
    
    def get_operator_frequency(self, op_type: str) -> float:
        """
        Get operator frequency score.
        
        Args:
            op_type: Operator type
            
        Returns:
            Frequency score (0.0 to 1.0)
        """
        count = self.operator_counts.get(op_type, 0)
        total = sum(self.operator_counts.values())
        return count / total if total > 0 else 0.0
    
    def get_token_patterns(
        self,
        graph_signature: str,
    ) -> List[List[str]]:
        """
        Get token sequences associated with operator graph.
        
        Args:
            graph_signature: Graph signature
            
        Returns:
            List of token sequences
        """
        return self.graph_to_tokens.get(graph_signature, [])
    
    def get_all_operators(self) -> Set[str]:
        """Get all unique operator types stored."""
        return set(self.operator_counts.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_graphs": self.total_graphs_stored,
            "total_transitions": self.total_transitions_learned,
            "unique_operators": len(self.operator_counts),
            "unique_fields": len(self.field_to_graph),
            "most_common_operators": sorted(
                self.operator_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "most_common_transitions": sorted(
                self.operator_transitions.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }
    
    def clear(self) -> None:
        """Clear all stored data."""
        self.graphs.clear()
        self.field_to_graph.clear()
        self.operator_transitions.clear()
        self.graph_to_tokens.clear()
        self.operator_counts.clear()
        self.total_graphs_stored = 0
        self.total_transitions_learned = 0


__all__ = ["OperatorMemoryStorage", "StoredGraph"]
