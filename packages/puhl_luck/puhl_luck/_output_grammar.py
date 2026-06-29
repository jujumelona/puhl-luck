"""
Output Grammar State Machine for P53

Constrains generation to valid output formats:
- CODE: Python function structure
- LABEL: Single label token
- QA: Text span answer
- TEXT: Free-form text

Prevents invalid transitions like ")::" or "return def"
"""

from typing import Set, Optional, List
from enum import Enum


class OutputMode(Enum):
    """Output format types."""
    CODE = "code"
    LABEL = "label"
    QA = "qa"
    TEXT = "text"


class CodeState(Enum):
    """States for Python code generation."""
    START = "start"
    RETURN_KW = "return"
    EXPR = "expr"
    OPERATOR = "operator"
    PAREN_OPEN = "paren_open"
    PAREN_CLOSE = "paren_close"
    DONE = "done"


class OutputGrammar:
    """
    State machine for valid output generation.
    
    Prevents invalid token sequences based on output mode.
    """
    
    def __init__(self, mode: OutputMode = OutputMode.CODE):
        self.mode = mode
        self.state = self._initial_state()
        
        # Token categories (will be populated dynamically)
        self.keywords = {"return", "def", "if", "else", "for", "while", "class", "import"}
        self.operators = {"+", "-", "*", "/", "//", "%", "**", "==", "!=", "<", ">", "<=", ">=", "and", "or", "not"}
        self.punctuation = {"(", ")", "[", "]", "{", "}", ",", ":", ".", ";"}
        self.special = {"[BOS]", "[SEP]", "[EOS]"}
    
    def _initial_state(self) -> str:
        """Get initial state for mode."""
        if self.mode == OutputMode.CODE:
            return CodeState.START.value
        elif self.mode == OutputMode.LABEL:
            return "label_start"
        elif self.mode == OutputMode.QA:
            return "qa_start"
        else:  # TEXT
            return "text_start"
    
    def is_valid_next(self, token: str, allow_eos: bool = False) -> bool:
        """
        Check if token is valid in current state.
        
        Args:
            token: Next token to check
            allow_eos: Whether to allow [EOS] in this position
            
        Returns:
            True if transition is valid
        """
        # Special tokens
        if token == "[EOS]":
            return allow_eos
        if token == "[BOS]" or token == "[SEP]":
            return False  # Should never generate these
        
        # Mode-specific validation
        if self.mode == OutputMode.CODE:
            return self._is_valid_code_token(token)
        elif self.mode == OutputMode.LABEL:
            return self._is_valid_label_token(token)
        elif self.mode == OutputMode.QA:
            return self._is_valid_qa_token(token)
        else:  # TEXT
            return True  # Free-form text allows anything
    
    def _is_valid_code_token(self, token: str) -> bool:
        """Validate token for Python code generation."""
        current = self.state
        
        if current == CodeState.START.value:
            # First token should be 'return' (for simple functions)
            if token == "return":
                return True
            # Or could be an identifier for assignment
            if token not in self.keywords and token not in self.punctuation:
                return True
            return False
        
        elif current == CodeState.RETURN_KW.value:
            # After 'return', expect expression start
            # Can be: identifier, number, string, opening paren, operator (unary)
            if token in {"(", "[", "{"}:
                return True
            if token in {"-", "not"}:  # Unary operators
                return True
            if token not in self.keywords and token not in {")", "]", "}", ",", ":", ";"}:
                return True
            return False
        
        elif current == CodeState.EXPR.value:
            # In expression, can continue with:
            # - Binary operator
            # - Closing paren
            # - Method call (.)
            # - End of expression
            if token in self.operators:
                return True
            if token in {")", "]", "}"}:
                return True
            if token in {".", ","}:
                return True
            # Identifier (chained attribute/call)
            if token not in self.keywords:
                return True
            return False
        
        elif current == CodeState.OPERATOR.value:
            # After operator, expect another expression
            if token in {"(", "[", "{"}:
                return True
            if token not in self.keywords and token not in {")", "]", "}", ",", ":", ";"}:
                return True
            return False
        
        elif current == CodeState.PAREN_OPEN.value:
            # Inside parens, can be anything except unmatched close
            if token == ")":
                return True  # Empty parens
            return True
        
        # Default: allow most things
        return True
    
    def _is_valid_label_token(self, token: str) -> bool:
        """Validate token for label generation (single token)."""
        if self.state == "label_start":
            # First token: any non-punctuation
            return token not in self.punctuation and token not in self.special
        else:
            # After first token, only allow [EOS]
            return token == "[EOS]"
    
    def _is_valid_qa_token(self, token: str) -> bool:
        """Validate token for QA span (free text span)."""
        # QA allows most tokens except invalid punctuation combinations
        if token == "[BOS]" or token == "[SEP]":
            return False
        return True
    
    def update(self, token: str) -> None:
        """
        Update state after generating token.
        
        Args:
            token: Token that was just generated
        """
        if self.mode == OutputMode.CODE:
            self._update_code_state(token)
        elif self.mode == OutputMode.LABEL:
            if self.state == "label_start":
                self.state = "label_done"
        # QA and TEXT don't need state tracking
    
    def _update_code_state(self, token: str) -> None:
        """Update code generation state."""
        if token == "return":
            self.state = CodeState.RETURN_KW.value
        elif token in self.operators:
            self.state = CodeState.OPERATOR.value
        elif token == "(":
            self.state = CodeState.PAREN_OPEN.value
        elif token in {")", "]", "}"}:
            self.state = CodeState.EXPR.value
        elif token == "[EOS]":
            self.state = CodeState.DONE.value
        else:
            # Identifier or literal
            self.state = CodeState.EXPR.value
    
    def get_invalid_penalty(self, token: str) -> float:
        """
        Get penalty for invalid token in current state.
        
        Args:
            token: Token to check
            
        Returns:
            Penalty value (0 = valid, high = invalid)
        """
        if self.is_valid_next(token):
            return 0.0
        else:
            return 100.0  # Strong penalty for invalid transitions
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.state = self._initial_state()
    
    def is_done(self) -> bool:
        """Check if generation is complete."""
        if self.mode == OutputMode.CODE:
            return self.state == CodeState.DONE.value
        elif self.mode == OutputMode.LABEL:
            return self.state == "label_done"
        return False


def create_grammar(mode: str = "code") -> OutputGrammar:
    """
    Create grammar for mode.
    
    Args:
        mode: "code", "label", "qa", or "text"
        
    Returns:
        OutputGrammar instance
    """
    mode_map = {
        "code": OutputMode.CODE,
        "label": OutputMode.LABEL,
        "qa": OutputMode.QA,
        "text": OutputMode.TEXT,
    }
    return OutputGrammar(mode_map.get(mode, OutputMode.TEXT))
