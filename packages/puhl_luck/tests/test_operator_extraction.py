"""
Test operator extraction from code and text.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from puhl_luck._memory_operator_graph import Operator, OperatorGraph
from puhl_luck._memory_operator_extraction import (
    CodeOperatorExtractor,
    NLPOperatorExtractor,
)


class TestOperatorGraph:
    """Test operator graph data structure."""
    
    def test_operator_creation(self):
        """Test creating operators."""
        op = Operator("FILTER", params={"condition": "even"})
        assert op.op_type == "FILTER"
        assert op.params["condition"] == "even"
        assert op.signature() == "FILTER(condition=even)"
    
    def test_operator_without_params(self):
        """Test operator without parameters."""
        op = Operator("RETURN")
        assert op.signature() == "RETURN"
    
    def test_graph_serialization(self):
        """Test graph serialization and deserialization."""
        ops = [
            Operator("FILTER"),
            Operator("COUNT"),
            Operator("RETURN"),
        ]
        edges = [(0, 1), (1, 2)]
        graph = OperatorGraph(ops, edges)
        
        # Serialize
        sig = graph.serialize()
        assert "FILTER" in sig
        assert "COUNT" in sig
        assert "0-1" in sig
        
        # Deserialize
        graph2 = OperatorGraph.deserialize(sig)
        assert len(graph2.nodes) == 3
        assert len(graph2.edges) == 2
        assert graph2.nodes[0].op_type == "FILTER"
    
    def test_topological_order(self):
        """Test topological ordering."""
        ops = [
            Operator("A"),
            Operator("B"),
            Operator("C"),
        ]
        edges = [(0, 1), (1, 2)]  # A → B → C
        graph = OperatorGraph(ops, edges)
        
        ordered = graph.topological_order()
        assert len(ordered) == 3
        assert ordered[0].op_type == "A"
        assert ordered[1].op_type == "B"
        assert ordered[2].op_type == "C"
    
    def test_cycle_detection(self):
        """Test cycle detection."""
        ops = [Operator("A"), Operator("B")]
        edges = [(0, 1), (1, 0)]  # A → B → A (cycle!)
        graph = OperatorGraph(ops, edges)
        
        assert graph.has_cycle()
        
        with pytest.raises(ValueError):
            graph.topological_order()
    
    def test_roots_and_leaves(self):
        """Test finding root and leaf nodes."""
        ops = [Operator("A"), Operator("B"), Operator("C")]
        edges = [(0, 1), (1, 2)]  # A → B → C
        graph = OperatorGraph(ops, edges)
        
        roots = graph.get_roots()
        assert roots == [0]  # Only A has no incoming
        
        leaves = graph.get_leaves()
        assert leaves == [2]  # Only C has no outgoing


class TestCodeOperatorExtractor:
    """Test code operator extraction."""
    
    def test_simple_function(self):
        """Test extracting from simple function."""
        code = """
def add(a, b):
    return a + b
"""
        extractor = CodeOperatorExtractor()
        graph = extractor.extract(code)
        
        op_types = [op.op_type for op in graph.nodes]
        assert "FUNCTION_DEF" in op_types
        assert "ARG_LIST" in op_types
        assert "RETURN" in op_types
        assert "ADD" in op_types
    
    def test_list_comprehension(self):
        """Test extracting list comprehension."""
        code = """
def count_even(nums):
    return len([x for x in nums if x % 2 == 0])
"""
        extractor = CodeOperatorExtractor()
        graph = extractor.extract(code)
        
        op_types = [op.op_type for op in graph.nodes]
        
        # Check expected operators
        assert "FUNCTION_DEF" in op_types
        assert "LIST_COMPREHENSION" in op_types
        assert "ITERATE" in op_types
        assert "FILTER" in op_types
        assert "MODULO" in op_types
        assert "COMPARE_EQ" in op_types
        assert "LEN" in op_types
        assert "RETURN" in op_types
        
        print(f"Extracted {len(graph.nodes)} operators:")
        for op in graph.nodes:
            print(f"  {op.signature()}")
    
    def test_if_statement(self):
        """Test extracting if statement."""
        code = """
def check(x):
    if x > 0:
        return True
    else:
        return False
"""
        extractor = CodeOperatorExtractor()
        graph = extractor.extract(code)
        
        op_types = [op.op_type for op in graph.nodes]
        assert "IF" in op_types
        assert "ELSE" in op_types
        assert "COMPARE_GT" in op_types
    
    def test_for_loop(self):
        """Test extracting for loop."""
        code = """
def sum_list(nums):
    total = 0
    for x in nums:
        total += x
    return total
"""
        extractor = CodeOperatorExtractor()
        graph = extractor.extract(code)
        
        op_types = [op.op_type for op in graph.nodes]
        assert "FOR" in op_types
        assert "RETURN" in op_types
    
    def test_nested_comprehension(self):
        """Test nested list comprehension."""
        code = """
def matrix_even(matrix):
    return [[x for x in row if x % 2 == 0] for row in matrix]
"""
        extractor = CodeOperatorExtractor()
        graph = extractor.extract(code)
        
        op_types = [op.op_type for op in graph.nodes]
        # Should have multiple LIST_COMPREHENSION
        list_comp_count = op_types.count("LIST_COMPREHENSION")
        assert list_comp_count >= 2
    
    def test_invalid_code(self):
        """Test handling invalid code."""
        code = "def invalid syntax"
        extractor = CodeOperatorExtractor()
        graph = extractor.extract(code)
        
        # Should return empty graph for invalid code
        assert len(graph.nodes) == 0
    
    def test_function_calls(self):
        """Test extracting function calls."""
        code = """
def process(nums):
    return max(sorted(nums))
"""
        extractor = CodeOperatorExtractor()
        graph = extractor.extract(code)
        
        op_types = [op.op_type for op in graph.nodes]
        assert "MAX" in op_types
        assert "SORTED" in op_types
    
    def test_boolean_operations(self):
        """Test extracting boolean operations."""
        code = """
def check(a, b, c):
    return a > 0 and b < 10 or not c
"""
        extractor = CodeOperatorExtractor()
        graph = extractor.extract(code)
        
        op_types = [op.op_type for op in graph.nodes]
        assert "AND" in op_types or "OR" in op_types
        assert "NOT" in op_types


class TestNLPOperatorExtractor:
    """Test NLP operator extraction."""
    
    def test_classification_extraction(self):
        """Test extracting from classification task."""
        text = "IT 기술 혁신 인공지능 뉴스"
        label = "IT과학"
        
        extractor = NLPOperatorExtractor()
        graph = extractor.extract(text, task="classification", target=label)
        
        op_types = [op.op_type for op in graph.nodes]
        assert "KEYWORD_MATCH" in op_types or "DOMAIN_MATCH" in op_types
        assert "CLASSIFY" in op_types
        assert "RETURN_LABEL" in op_types
        
        # Check label is stored
        label_ops = [op for op in graph.nodes if op.op_type == "RETURN_LABEL"]
        assert len(label_ops) > 0
        assert label_ops[0].params.get("label") == "IT과학"
    
    def test_nli_extraction(self):
        """Test extracting from NLI task."""
        text = "All cats are animals. Some animals are cats."
        label = "entailment"
        
        extractor = NLPOperatorExtractor()
        graph = extractor.extract(text, task="nli", target=label)
        
        op_types = [op.op_type for op in graph.nodes]
        # Should detect quantifiers or logical reasoning
        assert any("QUANTIFIER" in op for op in op_types) or "LOGICAL_REASONING" in op_types
        assert "RETURN_LABEL" in op_types
    
    def test_qa_extraction(self):
        """Test extracting from QA task."""
        text = "What is the capital of France?"
        answer = "Paris"
        
        extractor = NLPOperatorExtractor()
        graph = extractor.extract(text, task="qa", target=answer)
        
        op_types = [op.op_type for op in graph.nodes]
        assert "SPAN_EXTRACT" in op_types or "ANSWER_TYPE" in op_types
        assert "RETURN_ANSWER" in op_types
    
    def test_domain_detection(self):
        """Test domain detection."""
        extractor = NLPOperatorExtractor()
        
        # Technology domain
        graph = extractor.extract("AI technology innovation", task="classification")
        domain_ops = [op for op in graph.nodes if op.op_type == "DOMAIN_MATCH"]
        if domain_ops:
            assert domain_ops[0].params.get("domain") == "technology"
        
        # Politics domain
        graph = extractor.extract("정치 선거 정부", task="classification")
        domain_ops = [op for op in graph.nodes if op.op_type == "DOMAIN_MATCH"]
        if domain_ops:
            assert domain_ops[0].params.get("domain") == "politics"


class TestOperatorGraphIntegration:
    """Test integration scenarios."""
    
    def test_humaneval_example(self):
        """Test with actual HumanEval-style code."""
        code = """
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j and abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""
        extractor = CodeOperatorExtractor()
        graph = extractor.extract(code)
        
        # Should have reasonable number of operators
        assert len(graph.nodes) > 5
        
        # Should have control flow
        op_types = [op.op_type for op in graph.nodes]
        assert "FOR" in op_types
        assert "IF" in op_types
        assert "RETURN" in op_types
        
        print(f"\nExtracted {len(graph.nodes)} operators from HumanEval example")
        print(f"Graph signature: {graph.serialize()}")
    
    def test_classification_example(self):
        """Test with actual classification example."""
        text = "애플이 새로운 아이폰을 출시했다. AI 기능이 강화됐다."
        label = "IT과학"
        
        extractor = NLPOperatorExtractor()
        graph = extractor.extract(text, task="classification", target=label)
        
        assert len(graph.nodes) > 0
        
        # Should have classification pipeline
        op_types = [op.op_type for op in graph.nodes]
        assert "CLASSIFY" in op_types
        
        print(f"\nExtracted {len(graph.nodes)} operators from classification example")
        print(f"Operators: {[op.signature() for op in graph.nodes]}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
