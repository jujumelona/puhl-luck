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
    if not tokens:
        return ""
    
    result = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        
        # Handle structural tokens
        if token == "<NEWLINE>":
            result.append("\n")
            i += 1
            continue
        elif token == "<INDENT>":
            # Add indentation at start of next line
            if result and result[-1] == "\n":
                result.append("    ")
            i += 1
            continue
        elif token == "<DEDENT>":
            # Just mark - actual dedent handled by newline
            i += 1
            continue
        
        # Fix common operator splits
        # == != <= >= << >> += -= *= /= //=
        if i + 1 < len(tokens):
            next_token = tokens[i + 1]
            
            # Two-char operators
            if token == "=" and next_token == "=":
                result.append("==")
                i += 2
                continue
            elif token == "!" and next_token == "=":
                result.append("!=")
                i += 2
                continue
            elif token == "<" and next_token == "=":
                result.append("<=")
                i += 2
                continue
            elif token == ">" and next_token == "=":
                result.append(">=")
                i += 2
                continue
            elif token == "<" and next_token == "<":
                result.append("<<")
                i += 2
                continue
            elif token == ">" and next_token == ">":
                result.append(">>")
                i += 2
                continue
            elif token == "+" and next_token == "=":
                result.append("+=")
                i += 2
                continue
            elif token == "-" and next_token == "=":
                result.append("-=")
                i += 2
                continue
            elif token == "*" and next_token == "=":
                result.append("*=")
                i += 2
                continue
            elif token == "/" and next_token == "=":
                result.append("/=")
                i += 2
                continue
            elif token == "/" and next_token == "/":
                # Could be // or //=
                if i + 2 < len(tokens) and tokens[i + 2] == "=":
                    result.append("//=")
                    i += 3
                    continue
                else:
                    result.append("//")
                    i += 2
                    continue
        
        # Add spacing
        if result:
            last = result[-1]
            
            # No space after these
            if last in ("(", "[", "{", "\n"):
                result.append(token)
            # No space before these
            elif token in (")", "]", "}", ",", ":", ";"):
                result.append(token)
            # No space around dots
            elif token == "." or last == ".":
                result.append(token)
            else:
                result.append(" " + token)
        else:
            result.append(token)
        
        i += 1
    
    return "".join(result)


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
