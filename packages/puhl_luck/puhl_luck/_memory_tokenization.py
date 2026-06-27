"""
Token-level utilities for code generation.

Provides code-aware tokenization that preserves Python syntax structure
while splitting into meaningful tokens for pattern-based generation.
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional


def tokenize_code(code: str) -> List[str]:
    """
    Tokenize Python code into semantic tokens.
    
    Preserves:
    - Keywords (def, class, if, for, etc.)
    - Identifiers (function names, variable names)
    - Operators (=, +, -, *, etc.)
    - Literals (strings, numbers)
    - Structural elements (parentheses, colons, indentation)
    
    Args:
        code: Python source code string
        
    Returns:
        List of token strings
    """
    # Use Python tokenizer for accurate splitting
    import tokenize
    import io
    
    tokens = []
    try:
        # Tokenize using Python's tokenize module
        readline = io.BytesIO(code.encode('utf-8')).readline
        for tok in tokenize.tokenize(readline):
            tok_type = tok.type
            tok_string = tok.string
            
            # Skip encoding token and newlines (we'll handle structure differently)
            if tok_type in (tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            
            # Keep structural newlines and indents
            if tok_type in (tokenize.NEWLINE, tokenize.NL):
                tokens.append("<NEWLINE>")
            elif tok_type == tokenize.INDENT:
                tokens.append("<INDENT>")
            elif tok_type == tokenize.DEDENT:
                tokens.append("<DEDENT>")
            else:
                # Keep actual token
                if tok_string.strip():
                    tokens.append(tok_string)
    
    except tokenize.TokenError:
        # Fallback to simple whitespace splitting if tokenization fails
        tokens = simple_tokenize(code)
    
    return tokens


def simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace-based tokenization with punctuation separation.
    
    Used as fallback for non-code or invalid syntax.
    """
    # Split on whitespace and separate punctuation
    pattern = r'(\w+|[^\w\s])'
    tokens = re.findall(pattern, text)
    return [t for t in tokens if t.strip()]


def detokenize_code(tokens: List[str]) -> str:
    """
    Convert token list back to Python code string.
    
    Args:
        tokens: List of code tokens
        
    Returns:
        Reconstructed code string
    """
    lines = []
    current_line = []
    indent_level = 0
    
    for token in tokens:
        if token == "<NEWLINE>":
            # Finish current line
            lines.append("".join(current_line))
            current_line = []
        elif token == "<INDENT>":
            indent_level += 1
        elif token == "<DEDENT>":
            indent_level = max(0, indent_level - 1)
        else:
            # Add indentation at start of line
            if not current_line and indent_level > 0:
                current_line.append("    " * indent_level)
            
            # Add token with appropriate spacing
            if current_line and not token in ",:()[]{}":
                # Add space before most tokens (except punctuation)
                if not current_line[-1].endswith(("(", "[", "{")):
                    current_line.append(" ")
            
            current_line.append(token)
    
    # Add final line
    if current_line:
        lines.append("".join(current_line))
    
    return "\n".join(lines)


def validate_python(code: str) -> bool:
    """
    Check if code is syntactically valid Python.
    
    Args:
        code: Python source code
        
    Returns:
        True if valid, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_function_body(code: str, function_name: Optional[str] = None) -> Optional[str]:
    """
    Extract the first function definition from code.
    
    Args:
        code: Python source code
        function_name: Optional specific function name to extract
        
    Returns:
        Function code or None if not found
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if function_name is None or node.name == function_name:
                    # Use ast.get_source_segment if available (Python 3.8+)
                    return ast.unparse(node)
    except Exception:
        pass
    
    # Fallback: regex extraction
    if function_name:
        pattern = rf'def\s+{re.escape(function_name)}\s*\([^)]*\):[^\n]*(?:\n(?:    |\t).+)*'
    else:
        pattern = r'def\s+\w+\s*\([^)]*\):[^\n]*(?:\n(?:    |\t).+)*'
    
    match = re.search(pattern, code, re.MULTILINE)
    if match:
        return match.group(0)
    
    return None


def filter_code_tokens(tokens: List[str]) -> List[str]:
    """
    Filter out non-code tokens (Korean text, problem descriptions, etc.).
    
    Args:
        tokens: Mixed token list
        
    Returns:
        Filtered list containing only code-relevant tokens
    """
    filtered = []
    
    # Korean character range
    korean_pattern = re.compile(r'[\u3131-\u3163\uac00-\ud7a3]+')
    
    # Problem description keywords to filter
    description_keywords = {
        "문제", "다음", "작성하라", "함수를", "코드", "설명",
        "problem", "write", "following", "function", "description",
    }
    
    for token in tokens:
        # Skip Korean text
        if korean_pattern.search(token):
            continue
        
        # Skip description keywords
        if token.lower() in description_keywords:
            continue
        
        # Skip feature prefixes (tok:, bi:, tri:, etc.)
        if ":" in token and token.split(":")[0] in ("tok", "bi", "tri", "gram", "pos", "dep"):
            continue
        
        filtered.append(token)
    
    return filtered


__all__ = [
    "tokenize_code",
    "simple_tokenize",
    "detokenize_code",
    "validate_python",
    "extract_function_body",
    "filter_code_tokens",
]
