"""
Operator Extraction from Code and Text

Extracts operator graphs from:
- Python code (AST-based)
- NLP text (pattern-based)

NO raw text storage - only operator patterns.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ._memory_operator_graph import Operator, OperatorGraph


class CodeOperatorExtractor:
    """
    Extract operators from Python code using AST.
    
    Converts code into operator graph representation:
        def count_even(nums):
            return len([x for x in nums if x % 2 == 0])
    
    Becomes:
        FUNCTION_DEF → ARG_LIST → RETURN → LEN → LIST_COMP → 
        ITERATE → FILTER → MODULO → COMPARE_EQ
    """
    
    def __init__(self):
        self.operators: List[Operator] = []
        self.edges: List[Tuple[int, int]] = []
        self.node_stack: List[int] = []  # Track parent nodes
    
    def extract(self, code: str) -> OperatorGraph:
        """
        Extract operator graph from Python code.
        
        Args:
            code: Python source code string
            
        Returns:
            OperatorGraph representing computation structure
        """
        # Reset state
        self.operators = []
        self.edges = []
        self.node_stack = []
        
        try:
            tree = ast.parse(code)
            self._visit_node(tree)
        except SyntaxError as e:
            # Return empty graph for invalid code
            return OperatorGraph([], [])
        
        return OperatorGraph(self.operators, self.edges)
    
    def _add_operator(self, op: Operator) -> int:
        """Add operator and return its index."""
        idx = len(self.operators)
        self.operators.append(op)
        
        # Connect to parent if exists
        if self.node_stack:
            parent_idx = self.node_stack[-1]
            self.edges.append((parent_idx, idx))
        
        return idx
    
    def _visit_node(self, node: ast.AST) -> None:
        """Visit AST node and extract operators."""
        
        # Function definition
        if isinstance(node, ast.FunctionDef):
            idx = self._add_operator(Operator(
                "FUNCTION_DEF",
                params={"name": node.name}
            ))
            
            # Arguments
            if node.args.args:
                arg_names = [arg.arg for arg in node.args.args]
                self._add_operator(Operator(
                    "ARG_LIST",
                    params={"args": ",".join(arg_names)}
                ))
            
            # Visit body
            self.node_stack.append(idx)
            for child in node.body:
                self._visit_node(child)
            self.node_stack.pop()
        
        # Return statement
        elif isinstance(node, ast.Return):
            idx = self._add_operator(Operator("RETURN"))
            if node.value:
                self.node_stack.append(idx)
                self._visit_node(node.value)
                self.node_stack.pop()
        
        # Function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id.upper()
                idx = self._add_operator(Operator(func_name))
                
                # Visit arguments
                self.node_stack.append(idx)
                for arg in node.args:
                    self._visit_node(arg)
                self.node_stack.pop()
        
        # List comprehension
        elif isinstance(node, ast.ListComp):
            idx = self._add_operator(Operator("LIST_COMPREHENSION"))
            self.node_stack.append(idx)
            
            # Element expression
            self._visit_node(node.elt)
            
            # Generators
            for gen in node.generators:
                # Iterator
                iter_idx = self._add_operator(Operator(
                    "ITERATE",
                    params={"var": gen.target.id if isinstance(gen.target, ast.Name) else "x"}
                ))
                self._visit_node(gen.iter)
                
                # Filters
                for if_clause in gen.ifs:
                    filter_idx = self._add_operator(Operator("FILTER"))
                    self.node_stack.append(filter_idx)
                    self._visit_node(if_clause)
                    self.node_stack.pop()
            
            self.node_stack.pop()
        
        # Binary operations
        elif isinstance(node, ast.BinOp):
            op_type = self._binop_type(node.op)
            idx = self._add_operator(Operator(op_type))
            
            self.node_stack.append(idx)
            self._visit_node(node.left)
            self._visit_node(node.right)
            self.node_stack.pop()
        
        # Comparisons
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                op_type = self._compare_type(op)
                idx = self._add_operator(Operator(op_type))
                
                self.node_stack.append(idx)
                self._visit_node(node.left)
                for comparator in node.comparators:
                    self._visit_node(comparator)
                self.node_stack.pop()
        
        # Boolean operations
        elif isinstance(node, ast.BoolOp):
            op_type = "AND" if isinstance(node.op, ast.And) else "OR"
            idx = self._add_operator(Operator(op_type))
            
            self.node_stack.append(idx)
            for value in node.values:
                self._visit_node(value)
            self.node_stack.pop()
        
        # Unary operations
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                idx = self._add_operator(Operator("NOT"))
                self.node_stack.append(idx)
                self._visit_node(node.operand)
                self.node_stack.pop()
        
        # If statement
        elif isinstance(node, ast.If):
            idx = self._add_operator(Operator("IF"))
            self.node_stack.append(idx)
            self._visit_node(node.test)
            self.node_stack.pop()
            
            # Then branch
            for child in node.body:
                self._visit_node(child)
            
            # Else branch
            if node.orelse:
                else_idx = self._add_operator(Operator("ELSE"))
                self.node_stack.append(else_idx)
                for child in node.orelse:
                    self._visit_node(child)
                self.node_stack.pop()
        
        # For loop
        elif isinstance(node, ast.For):
            idx = self._add_operator(Operator(
                "FOR",
                params={"var": node.target.id if isinstance(node.target, ast.Name) else "i"}
            ))
            
            self.node_stack.append(idx)
            self._visit_node(node.iter)
            for child in node.body:
                self._visit_node(child)
            self.node_stack.pop()
        
        # While loop
        elif isinstance(node, ast.While):
            idx = self._add_operator(Operator("WHILE"))
            self.node_stack.append(idx)
            self._visit_node(node.test)
            for child in node.body:
                self._visit_node(child)
            self.node_stack.pop()
        
        # List, Tuple, Set, Dict
        elif isinstance(node, ast.List):
            idx = self._add_operator(Operator("LIST"))
            self.node_stack.append(idx)
            for elt in node.elts:
                self._visit_node(elt)
            self.node_stack.pop()
        
        elif isinstance(node, ast.Tuple):
            idx = self._add_operator(Operator("TUPLE"))
            self.node_stack.append(idx)
            for elt in node.elts:
                self._visit_node(elt)
            self.node_stack.pop()
        
        elif isinstance(node, ast.Set):
            idx = self._add_operator(Operator("SET"))
            self.node_stack.append(idx)
            for elt in node.elts:
                self._visit_node(elt)
            self.node_stack.pop()
        
        elif isinstance(node, ast.Dict):
            idx = self._add_operator(Operator("DICT"))
            self.node_stack.append(idx)
            for k, v in zip(node.keys, node.values):
                if k:
                    self._visit_node(k)
                self._visit_node(v)
            self.node_stack.pop()
        
        # Variables and constants (leaf nodes, no recursion needed)
        elif isinstance(node, (ast.Name, ast.Constant)):
            pass  # Skip leaf nodes
        
        # Subscript (indexing)
        elif isinstance(node, ast.Subscript):
            idx = self._add_operator(Operator("INDEX"))
            self.node_stack.append(idx)
            self._visit_node(node.value)
            self._visit_node(node.slice)
            self.node_stack.pop()
        
        # Attribute access
        elif isinstance(node, ast.Attribute):
            self._add_operator(Operator(
                "ATTR",
                params={"attr": node.attr}
            ))
            self._visit_node(node.value)
        
        # Module (top level)
        elif isinstance(node, ast.Module):
            for child in node.body:
                self._visit_node(child)
        
        # Default: recurse on child nodes
        else:
            for child in ast.iter_child_nodes(node):
                self._visit_node(child)
    
    def _binop_type(self, op: ast.operator) -> str:
        """Get operator type from binary operation."""
        if isinstance(op, ast.Add):
            return "ADD"
        elif isinstance(op, ast.Sub):
            return "SUB"
        elif isinstance(op, ast.Mult):
            return "MUL"
        elif isinstance(op, ast.Div):
            return "DIV"
        elif isinstance(op, ast.Mod):
            return "MODULO"
        elif isinstance(op, ast.Pow):
            return "POW"
        elif isinstance(op, ast.FloorDiv):
            return "FLOORDIV"
        else:
            return "BINOP"
    
    def _compare_type(self, op: ast.cmpop) -> str:
        """Get operator type from comparison."""
        if isinstance(op, ast.Eq):
            return "COMPARE_EQ"
        elif isinstance(op, ast.NotEq):
            return "COMPARE_NEQ"
        elif isinstance(op, ast.Lt):
            return "COMPARE_LT"
        elif isinstance(op, ast.LtE):
            return "COMPARE_LTE"
        elif isinstance(op, ast.Gt):
            return "COMPARE_GT"
        elif isinstance(op, ast.GtE):
            return "COMPARE_GTE"
        elif isinstance(op, ast.In):
            return "COMPARE_IN"
        elif isinstance(op, ast.NotIn):
            return "COMPARE_NOTIN"
        else:
            return "COMPARE"


class NLPOperatorExtractor:
    """
    Extract operators from NLP tasks using pattern matching.
    
    Converts text + label into operator graph:
        text: "IT 기술 혁신 인공지능 뉴스"
        label: "IT과학"
    
    Becomes:
        KEYWORD_MATCH → DOMAIN_MATCH → CLASSIFY → RETURN_LABEL
    """
    
    def extract(
        self,
        text: str,
        task: str,
        target: Optional[str] = None,
    ) -> OperatorGraph:
        """
        Extract operator graph from NLP task.
        
        Args:
            text: Input text
            task: Task type (classification, nli, qa)
            target: Target output (label, answer, etc.)
            
        Returns:
            OperatorGraph representing reasoning structure
        """
        if task == "classification":
            return self._extract_classification(text, target)
        elif task == "nli":
            return self._extract_nli(text, target)
        elif task == "qa":
            return self._extract_qa(text, target)
        else:
            return OperatorGraph([], [])
    
    def _extract_classification(self, text: str, label: Optional[str]) -> OperatorGraph:
        """Extract operators for classification task."""
        operators = []
        edges = []
        
        # Keyword detection
        keywords = self._detect_keywords(text)
        if keywords:
            op_idx = len(operators)
            operators.append(Operator(
                "KEYWORD_MATCH",
                params={"keywords": ",".join(keywords[:5])}
            ))
        
        # Domain detection
        domain = self._detect_domain(text)
        if domain:
            op_idx = len(operators)
            operators.append(Operator(
                "DOMAIN_MATCH",
                params={"domain": domain}
            ))
            if len(operators) > 1:
                edges.append((len(operators)-2, len(operators)-1))
        
        # Classification operator
        classify_idx = len(operators)
        operators.append(Operator("CLASSIFY"))
        if len(operators) > 1:
            edges.append((len(operators)-2, len(operators)-1))
        
        # Return label
        if label:
            label_idx = len(operators)
            operators.append(Operator(
                "RETURN_LABEL",
                params={"label": label}
            ))
            edges.append((classify_idx, label_idx))
        
        return OperatorGraph(operators, edges)
    
    def _extract_nli(self, text: str, label: Optional[str]) -> OperatorGraph:
        """Extract operators for NLI task."""
        operators = []
        edges = []
        
        # Detect quantifiers
        if any(word in text for word in ["all", "every", "모든", "모두"]):
            operators.append(Operator("UNIVERSAL_QUANTIFIER"))
        
        if any(word in text for word in ["some", "일부", "어떤"]):
            operators.append(Operator("EXISTENTIAL_QUANTIFIER"))
        
        # Logical reasoning
        operators.append(Operator("LOGICAL_REASONING"))
        if operators:
            edges.append((len(operators)-2, len(operators)-1))
        
        # Return entailment label
        if label:
            operators.append(Operator(
                "RETURN_LABEL",
                params={"label": label}
            ))
            edges.append((len(operators)-2, len(operators)-1))
        
        return OperatorGraph(operators, edges)
    
    def _extract_qa(self, text: str, answer: Optional[str]) -> OperatorGraph:
        """Extract operators for QA task."""
        operators = []
        edges = []
        
        # Span extraction
        operators.append(Operator("SPAN_EXTRACT"))
        
        # Answer type detection
        if answer and answer.isdigit():
            operators.append(Operator("ANSWER_TYPE", params={"type": "number"}))
        elif answer:
            operators.append(Operator("ANSWER_TYPE", params={"type": "text"}))
        
        if len(operators) > 1:
            edges.append((0, 1))
        
        # Return answer
        if answer:
            operators.append(Operator("RETURN_ANSWER"))
            edges.append((len(operators)-2, len(operators)-1))
        
        return OperatorGraph(operators, edges)
    
    def _detect_keywords(self, text: str) -> List[str]:
        """Detect important keywords from text."""
        # Simple keyword extraction (can be improved)
        words = text.split()
        # Filter short words and common words
        keywords = [w for w in words if len(w) >= 2 and w not in ["the", "a", "an", "is", "are"]]
        return keywords[:5]  # Top 5
    
    def _detect_domain(self, text: str) -> Optional[str]:
        """Detect domain from text."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["it", "기술", "컴퓨터", "인공지능", "ai"]):
            return "technology"
        elif any(kw in text_lower for kw in ["정치", "선거", "정부", "politics"]):
            return "politics"
        elif any(kw in text_lower for kw in ["경제", "금융", "주식", "economy"]):
            return "economy"
        elif any(kw in text_lower for kw in ["스포츠", "축구", "야구", "sports"]):
            return "sports"
        else:
            return None


__all__ = ["CodeOperatorExtractor", "NLPOperatorExtractor"]
