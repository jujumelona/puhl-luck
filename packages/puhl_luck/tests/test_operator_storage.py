"""
Test operator storage functionality.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from puhl_luck._memory_operator_graph import Operator, OperatorGraph
from puhl_luck._memory_operator_storage import OperatorMemoryStorage


class TestOperatorMemoryStorage:
    """Test operator memory storage."""
    
    def test_store_and_retrieve_graph(self):
        """Test storing and retrieving operator graphs."""
        storage = OperatorMemoryStorage()
        
        # Create a simple graph
        ops = [
            Operator("FILTER"),
            Operator("COUNT"),
            Operator("RETURN"),
        ]
        edges = [(0, 1), (1, 2)]
        graph = OperatorGraph(ops, edges)
        
        # Store graph
        field_sig = "feature1|feature2|feature3"
        tokens = ["def", "count", "(", "x", ")"]
        graph_sig = storage.store_graph(graph, field_sig, tokens)
        
        assert graph_sig is not None
        assert storage.total_graphs_stored == 1
        
        # Retrieve by field
        results = storage.retrieve_graphs_by_field(field_sig)
        assert len(results) > 0
        
        retrieved_graph, score = results[0]
        assert len(retrieved_graph.nodes) == 3
        assert retrieved_graph.nodes[0].op_type == "FILTER"
    
    def test_operator_transitions(self):
        """Test operator transition tracking."""
        storage = OperatorMemoryStorage()
        
        # Create graph with transitions
        ops = [
            Operator("FILTER"),
            Operator("EVEN"),
            Operator("COUNT"),
        ]
        edges = [(0, 1), (1, 2)]  # FILTER → EVEN → COUNT
        graph = OperatorGraph(ops, edges)
        
        storage.store_graph(graph, "field1", [])
        
        # Check transitions stored
        score1 = storage.get_transition_score("FILTER", "EVEN")
        score2 = storage.get_transition_score("EVEN", "COUNT")
        
        assert score1 > 0
        assert score2 > 0
        
        # Non-existent transition
        score3 = storage.get_transition_score("FILTER", "COUNT")
        assert score3 == 0 or score3 < score1
    
    def test_operator_frequency(self):
        """Test operator frequency tracking."""
        storage = OperatorMemoryStorage()
        
        # Store multiple graphs with overlapping operators
        graph1 = OperatorGraph([Operator("FILTER"), Operator("COUNT")], [(0, 1)])
        graph2 = OperatorGraph([Operator("FILTER"), Operator("SUM")], [(0, 1)])
        
        storage.store_graph(graph1, "field1", [])
        storage.store_graph(graph2, "field2", [])
        
        # FILTER appears twice
        freq_filter = storage.get_operator_frequency("FILTER")
        freq_count = storage.get_operator_frequency("COUNT")
        freq_sum = storage.get_operator_frequency("SUM")
        
        assert freq_filter > freq_count
        assert freq_filter > freq_sum
        assert freq_count == freq_sum  # Both appear once
    
    def test_token_patterns(self):
        """Test storing and retrieving token patterns."""
        storage = OperatorMemoryStorage()
        
        # Create graph
        graph = OperatorGraph([Operator("COUNT")], [])
        tokens1 = ["len", "(", "x", ")"]
        tokens2 = ["count", "(", "x", ")"]
        
        graph_sig = storage.store_graph(graph, "field1", tokens1)
        storage.store_graph(graph, "field1", tokens2)
        
        # Retrieve token patterns
        patterns = storage.get_token_patterns(graph_sig)
        assert len(patterns) == 2
        assert tokens1 in patterns
        assert tokens2 in patterns
    
    def test_multiple_fields_same_graph(self):
        """Test same graph activated by different fields."""
        storage = OperatorMemoryStorage()
        
        # Same graph, different fields
        graph = OperatorGraph([Operator("COUNT")], [])
        
        storage.store_graph(graph, "field1", [])
        storage.store_graph(graph, "field2", [])
        
        # Should be retrievable from both fields
        results1 = storage.retrieve_graphs_by_field("field1")
        results2 = storage.retrieve_graphs_by_field("field2")
        
        assert len(results1) > 0
        assert len(results2) > 0
    
    def test_exposure_count(self):
        """Test exposure count increases on repeated storage."""
        storage = OperatorMemoryStorage()
        
        graph = OperatorGraph([Operator("COUNT")], [])
        field = "field1"
        
        # Store same graph multiple times
        graph_sig = storage.store_graph(graph, field, [])
        storage.store_graph(graph, field, [])
        storage.store_graph(graph, field, [])
        
        # Check exposure count
        stored = storage.graphs[graph_sig]
        assert stored.exposure_count == 3
    
    def test_statistics(self):
        """Test statistics reporting."""
        storage = OperatorMemoryStorage()
        
        # Store some graphs
        graph1 = OperatorGraph([Operator("FILTER"), Operator("COUNT")], [(0, 1)])
        graph2 = OperatorGraph([Operator("SUM"), Operator("RETURN")], [(0, 1)])
        
        storage.store_graph(graph1, "field1", [])
        storage.store_graph(graph2, "field2", [])
        
        stats = storage.get_statistics()
        
        assert stats["total_graphs"] == 2
        assert stats["total_transitions"] == 2
        assert stats["unique_operators"] == 4
        assert stats["unique_fields"] == 2
        assert len(stats["most_common_operators"]) > 0
        assert len(stats["most_common_transitions"]) > 0
    
    def test_get_all_operators(self):
        """Test getting all unique operators."""
        storage = OperatorMemoryStorage()
        
        graph1 = OperatorGraph([Operator("FILTER"), Operator("COUNT")], [])
        graph2 = OperatorGraph([Operator("SUM"), Operator("COUNT")], [])
        
        storage.store_graph(graph1, "field1", [])
        storage.store_graph(graph2, "field2", [])
        
        all_ops = storage.get_all_operators()
        
        assert "FILTER" in all_ops
        assert "COUNT" in all_ops
        assert "SUM" in all_ops
        assert len(all_ops) == 3
    
    def test_clear(self):
        """Test clearing storage."""
        storage = OperatorMemoryStorage()
        
        # Store some data
        graph = OperatorGraph([Operator("COUNT")], [])
        storage.store_graph(graph, "field1", [])
        
        assert storage.total_graphs_stored > 0
        
        # Clear
        storage.clear()
        
        assert storage.total_graphs_stored == 0
        assert len(storage.graphs) == 0
        assert len(storage.field_to_graph) == 0


class TestOperatorStorageIntegration:
    """Test integration with extraction."""
    
    def test_store_extracted_code_graph(self):
        """Test storing graph extracted from code."""
        from puhl_luck._memory_operator_extraction import CodeOperatorExtractor
        
        storage = OperatorMemoryStorage()
        extractor = CodeOperatorExtractor()
        
        # Extract graph from code
        code = """
def count_even(nums):
    return len([x for x in nums if x % 2 == 0])
"""
        graph = extractor.extract(code)
        
        # Store graph
        field_sig = "count|even|numbers"
        tokens = code.split()
        graph_sig = storage.store_graph(graph, field_sig, tokens)
        
        assert graph_sig is not None
        assert storage.total_graphs_stored == 1
        
        # Should have learned transitions
        assert storage.total_transitions_learned > 0
        
        # Should be retrievable
        results = storage.retrieve_graphs_by_field(field_sig)
        assert len(results) > 0
    
    def test_store_extracted_nlp_graph(self):
        """Test storing graph extracted from NLP."""
        from puhl_luck._memory_operator_extraction import NLPOperatorExtractor
        
        storage = OperatorMemoryStorage()
        extractor = NLPOperatorExtractor()
        
        # Extract graph from text
        text = "IT 기술 혁신"
        label = "IT과학"
        graph = extractor.extract(text, task="classification", target=label)
        
        # Store graph
        field_sig = "IT|기술|혁신"
        tokens = text.split()
        graph_sig = storage.store_graph(graph, field_sig, tokens)
        
        assert graph_sig is not None
        assert storage.total_graphs_stored == 1
        
        # Retrieve
        results = storage.retrieve_graphs_by_field(field_sig)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
