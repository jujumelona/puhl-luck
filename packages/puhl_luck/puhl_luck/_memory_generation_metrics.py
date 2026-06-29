"""
Generation Metrics Tracker

Tracks whether outputs are exact copies, novel compositions, or reused patterns.

Key metrics:
- was_exact_copy: True if output matches stored training example exactly
- nearest_train_similarity: Overlap ratio with most similar training example
- operator_reuse_rate: Fraction of operators seen during training
- novel_composition: True if operator graph structure is new
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class GenerationMetrics:
    """Metrics for a single generation."""
    
    # Copy detection
    was_exact_copy: bool = False
    exact_copy_source: Optional[str] = None
    
    # Similarity to training data
    nearest_train_similarity: float = 0.0
    nearest_train_id: Optional[str] = None
    
    # Operator reuse
    operator_reuse_rate: float = 0.0
    operators_used: List[str] = None
    operators_novel: List[str] = None
    
    # Composition novelty
    novel_composition: bool = False
    graph_signature: Optional[str] = None
    
    # Generation method
    generation_method: str = "unknown"  # "operator", "sparse_autoregressive", "retrieval", "token_fallback"
    
    # Sparse autoregressive metrics (NEW)
    tokens_generated: int = 0
    backoff_levels: Dict[int, int] = None  # {level: count}
    copy_gate_activations: int = 0
    empty_output: bool = False
    
    def __post_init__(self):
        if self.operators_used is None:
            self.operators_used = []
        if self.operators_novel is None:
            self.operators_novel = []
        if self.backoff_levels is None:
            self.backoff_levels = {}


class MetricsTracker:
    """
    Track generation metrics across all outputs.
    
    Compares generated outputs against stored training data to detect:
    - Exact copies (memorization)
    - Novel operator compositions (generalization)
    - Operator reuse patterns
    """
    
    def __init__(self):
        # Training data storage
        self.training_outputs: Set[str] = set()
        self.training_graph_signatures: Set[str] = set()
        self.training_operators: Set[str] = set()
        
        # Generation history
        self.generation_count: int = 0
        self.exact_copy_count: int = 0
        self.novel_composition_count: int = 0
        
        # Metrics aggregation
        self.total_similarity: float = 0.0
        self.total_operator_reuse: float = 0.0
    
    def record_training_example(
        self,
        output: str,
        graph_signature: Optional[str] = None,
        operators: Optional[List[str]] = None,
    ) -> None:
        """
        Record a training example for later comparison.
        
        Args:
            output: The target output text
            graph_signature: Operator graph signature
            operators: List of operator types in the graph
        """
        self.training_outputs.add(output.strip())
        
        if graph_signature:
            self.training_graph_signatures.add(graph_signature)
        
        if operators:
            self.training_operators.update(operators)
    
    def compute_metrics(
        self,
        generated_output: str,
        graph_signature: Optional[str] = None,
        operators_used: Optional[List[str]] = None,
        generation_method: str = "unknown",
    ) -> GenerationMetrics:
        """
        Compute metrics for a generated output.
        
        Args:
            generated_output: The text that was generated
            graph_signature: Operator graph signature (if using operators)
            operators_used: List of operators used in generation
            generation_method: How it was generated ("operator", "retrieval", etc.)
            
        Returns:
            GenerationMetrics with all computed metrics
        """
        metrics = GenerationMetrics(generation_method=generation_method)
        
        # 1. Exact copy detection
        output_clean = generated_output.strip()
        metrics.was_exact_copy = output_clean in self.training_outputs
        
        if metrics.was_exact_copy:
            metrics.exact_copy_source = output_clean
            self.exact_copy_count += 1
        
        # 2. Nearest training similarity
        if self.training_outputs:
            max_sim = 0.0
            nearest_id = None
            
            for train_output in self.training_outputs:
                sim = self._compute_token_overlap(output_clean, train_output)
                if sim > max_sim:
                    max_sim = sim
                    nearest_id = train_output[:50]  # Store prefix as ID
            
            metrics.nearest_train_similarity = max_sim
            metrics.nearest_train_id = nearest_id
            self.total_similarity += max_sim
        
        # 3. Operator reuse rate
        if operators_used and self.training_operators:
            seen_ops = [op for op in operators_used if op in self.training_operators]
            novel_ops = [op for op in operators_used if op not in self.training_operators]
            
            metrics.operators_used = operators_used
            metrics.operators_novel = novel_ops
            metrics.operator_reuse_rate = len(seen_ops) / len(operators_used) if operators_used else 0.0
            
            self.total_operator_reuse += metrics.operator_reuse_rate
        
        # 4. Novel composition detection
        if graph_signature:
            metrics.graph_signature = graph_signature
            metrics.novel_composition = graph_signature not in self.training_graph_signatures
            
            if metrics.novel_composition:
                self.novel_composition_count += 1
        
        self.generation_count += 1
        
        return metrics
    
    def _compute_token_overlap(self, text1: str, text2: str) -> float:
        """
        Compute token-level overlap between two texts.
        
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all generations.
        
        Returns:
            Dictionary with aggregate metrics
        """
        if self.generation_count == 0:
            return {
                "total_generations": 0,
                "copy_rate": 0.0,
                "novel_composition_rate": 0.0,
                "avg_similarity": 0.0,
                "avg_operator_reuse": 0.0,
            }
        
        return {
            "total_generations": self.generation_count,
            "copy_rate": self.exact_copy_count / self.generation_count,
            "novel_composition_rate": self.novel_composition_count / self.generation_count,
            "avg_similarity": self.total_similarity / self.generation_count,
            "avg_operator_reuse": self.total_operator_reuse / self.generation_count,
        }
    
    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        self.generation_count = 0
        self.exact_copy_count = 0
        self.novel_composition_count = 0
        self.total_similarity = 0.0
        self.total_operator_reuse = 0.0


__all__ = ["GenerationMetrics", "MetricsTracker"]
