"""
Operator Graph Representation

Core data structures for operator-based generation:
- Operator: Single operation node (e.g., FILTER, COUNT, RETURN)
- OperatorGraph: DAG of operators with execution order

NO raw text - only operator patterns and transitions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Operator:
    """
    Single operator node in computation graph.
    
    Examples:
    - Code: Operator("FILTER", params={"condition": "even"})
    - Code: Operator("FUNCTION_DEF", params={"name": "count_even"})
    - NLP:  Operator("KEYWORD_MATCH", params={"keywords": ["IT", "기술"]})
    """
    
    op_type: str  # e.g., "FILTER", "COUNT", "RETURN", "CLASSIFY"
    params: Dict[str, Any] = field(default_factory=dict)
    activation: float = 0.0  # Activation score during generation
    
    def __hash__(self):
        # Make hashable for sets/dicts
        param_str = "|".join(f"{k}:{v}" for k, v in sorted(self.params.items()))
        return hash(f"{self.op_type}:{param_str}")
    
    def __eq__(self, other):
        if not isinstance(other, Operator):
            return False
        return self.op_type == other.op_type and self.params == other.params
    
    def signature(self) -> str:
        """Compact signature for this operator."""
        if not self.params:
            return self.op_type
        param_str = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.op_type}({param_str})"


@dataclass
class OperatorGraph:
    """
    Directed acyclic graph (DAG) of operators.
    
    Represents computation structure, NOT raw code.
    Example:
        nodes = [FILTER, EVEN, COUNT, RETURN]
        edges = [(0,1), (1,2), (2,3)]  # FILTER→EVEN→COUNT→RETURN
    """
    
    nodes: List[Operator]
    edges: List[Tuple[int, int]]  # (from_idx, to_idx)
    
    def serialize(self) -> str:
        """
        Serialize to compact signature.
        
        Format: "OP1|OP2|OP3;0-1,1-2,2-3"
        """
        node_sigs = "|".join(op.signature() for op in self.nodes)
        edge_sigs = ",".join(f"{i}-{j}" for i, j in self.edges)
        return f"{node_sigs};{edge_sigs}"
    
    @staticmethod
    def deserialize(signature: str) -> OperatorGraph:
        """Deserialize from compact signature."""
        node_part, edge_part = signature.split(";")
        
        # Parse nodes
        nodes = []
        for node_sig in node_part.split("|"):
            if "(" in node_sig:
                op_type, params_str = node_sig.split("(", 1)
                params_str = params_str.rstrip(")")
                params = {}
                for param in params_str.split(","):
                    k, v = param.split("=", 1)
                    params[k] = v
                nodes.append(Operator(op_type, params))
            else:
                nodes.append(Operator(node_sig))
        
        # Parse edges
        edges = []
        if edge_part:
            for edge_sig in edge_part.split(","):
                i, j = edge_sig.split("-")
                edges.append((int(i), int(j)))
        
        return OperatorGraph(nodes, edges)
    
    def topological_order(self) -> List[Operator]:
        """
        Return operators in topological order (Kahn's algorithm).
        
        Ensures operators execute in valid dependency order.
        """
        # Build in-degree map
        in_degree = {i: 0 for i in range(len(self.nodes))}
        adj_list = {i: [] for i in range(len(self.nodes))}
        
        for from_idx, to_idx in self.edges:
            adj_list[from_idx].append(to_idx)
            in_degree[to_idx] += 1
        
        # Start with nodes that have no dependencies
        queue = [i for i, deg in in_degree.items() if deg == 0]
        ordered = []
        
        while queue:
            idx = queue.pop(0)
            ordered.append(self.nodes[idx])
            
            # Reduce in-degree for neighbors
            for neighbor in adj_list[idx]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(ordered) != len(self.nodes):
            raise ValueError("Graph has cycles - not a valid DAG")
        
        return ordered
    
    def has_cycle(self) -> bool:
        """Check if graph has cycles."""
        try:
            self.topological_order()
            return False
        except ValueError:
            return True
    
    def add_edge(self, from_idx: int, to_idx: int) -> None:
        """Add edge to graph."""
        if (from_idx, to_idx) not in self.edges:
            self.edges.append((from_idx, to_idx))
    
    def remove_edge(self, from_idx: int, to_idx: int) -> None:
        """Remove edge from graph."""
        if (from_idx, to_idx) in self.edges:
            self.edges.remove((from_idx, to_idx))
    
    def get_roots(self) -> List[int]:
        """Get root nodes (no incoming edges)."""
        has_incoming = set(to_idx for _, to_idx in self.edges)
        return [i for i in range(len(self.nodes)) if i not in has_incoming]
    
    def get_leaves(self) -> List[int]:
        """Get leaf nodes (no outgoing edges)."""
        has_outgoing = set(from_idx for from_idx, _ in self.edges)
        return [i for i in range(len(self.nodes)) if i not in has_outgoing]
    
    def hash_signature(self) -> str:
        """Generate hash of graph signature."""
        sig = self.serialize()
        return hashlib.sha256(sig.encode('utf-8')).hexdigest()[:16]


__all__ = ["Operator", "OperatorGraph"]
