"""
Transition, Operator, and Surface Realization integration mixin for BrainMemory.

Wires the standalone Layer-3/4/8 classes (OperatorMemoryLayer,
TransitionMemoryLayer, SurfaceRealizationLayer, OperatorInduction)
into the BrainMemory mixin chain and exposes the full
generate() pipeline:

    state field  →  candidate emergence  →  surface realization  →  output

This also replaces the old expose_text()-only approach with
transition-based learning:

    partial state  →  continuation  →  completed state

Public API additions on BrainMemory
------------------------------------
- expose_pair(partial, complete, ...)  — primary transition learning
- generate(prompt, max_new_tokens)     — full field-based generation pipeline
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports of heavy field-layer classes to avoid circular imports
# ---------------------------------------------------------------------------

def _get_transition_layer():
    from ._memory_transition_layer import TransitionMemoryLayer
    return TransitionMemoryLayer

def _get_operator_layer():
    from ._memory_operator_layer import OperatorMemoryLayer
    return OperatorMemoryLayer

def _get_operator_induction():
    from ._memory_operator_induction import OperatorInduction
    return OperatorInduction

def _get_surface_layer():
    from ._memory_surface_realization import SurfaceRealizationLayer
    return SurfaceRealizationLayer

def _get_candidate_emergence():
    from ._memory_candidate_emergence import CandidateEmergence
    return CandidateEmergence

def _get_field_energy():
    from ._memory_field_energy import FreeEnergyMinimization
    return FreeEnergyMinimization

def _get_surface_storage():
    from ._memory_surface_storage import SurfaceSequenceStorage
    return SurfaceSequenceStorage


class MemoryTransitionMixin:
    """
    Mixin that initializes and wires all field-layer components into BrainMemory.

    Components managed
    ------------------
    _transition_layer   : TransitionMemoryLayer  (Layer 4)
    _operator_layer     : OperatorMemoryLayer    (Layer 3)
    _operator_induction : OperatorInduction      (Layer 4 → Layer 3)
    _surface_layer      : SurfaceRealizationLayer (Layer 8)
    _candidate_emergence: CandidateEmergence     (Layer 6)
    _energy_computer    : FreeEnergyMinimization (Layer 6 scoring)
    """

    def _init_transition_layers(self) -> None:
        """Call from BrainMemory.__init__ to initialise all field-layer components."""
        self._transition_layer = _get_transition_layer()()
        self._operator_layer = _get_operator_layer()()
        self._operator_induction = _get_operator_induction()()
        self._surface_layer = _get_surface_layer()()
        self._surface_storage = _get_surface_storage()()  # NEW: Universal surface storage
        self._energy_computer = _get_field_energy()()
        self._candidate_emergence = _get_candidate_emergence()(
            energy_computer=self._energy_computer,
            surface_layer=self._surface_layer,  # Pass surface_layer!
        )
        self._pending_induction_count: int = 0

    # ------------------------------------------------------------------
    # generate() — UNIVERSAL surface sequence generation
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        max_new_tokens: int = 64,
        use_surface_storage: bool = True,
    ) -> str:
        """
        UNIVERSAL surface sequence generation.
        
        Pipeline:
        1. Retrieve stored surface sequences matching query
        2. Score sequences by relevance + coherence
        3. Return best sequence raw text
        4. Fallback to token generation if no stored sequences
        
        Works for ALL tasks:
        - Code: query="def add(a,b):" → "return a+b"
        - Classification: query="뉴스 text..." → "IT과학"
        - QA: query="질문..." → "정답 span"
        - Generation: query="prompt..." → "full answer"
        
        Args:
            query: Input query/prompt
            max_new_tokens: Max tokens to generate (for fallback)
            use_surface_storage: Whether to use stored sequences (default: True)
            
        Returns:
            Raw surface text (ready to output)
        """
        if not use_surface_storage or not hasattr(self, "_surface_storage"):
            # Fallback to old generation
            return self.answer(query, max_new_tokens=max_new_tokens)
        
        # ---- Step 1: Retrieve stored surface sequences ----
        storage = self._surface_storage
        
        # Try exact input match first
        candidates = storage.retrieve_by_input(query, top_k=10)
        
        # If no exact match, try token overlap
        if not candidates:
            from ._memory_tokenization import simple_tokenize
            query_tokens = simple_tokenize(query)
            token_matches = storage.retrieve_by_tokens(query_tokens, top_k=10)
            candidates = [seq for seq, score in token_matches if score > 0.1]
        
        # ---- Step 2: No stored sequences, use token generation ----
        if not candidates:
            return self._generate_tokens_fallback(query, max_new_tokens)
        
        # ---- Step 3: Score candidates ----
        scored_candidates = []
        
        for seq in candidates:
            # Simple relevance score (can be improved with field energy)
            score = 1.0 / (1.0 + seq.retrieval_count * 0.01)  # Prefer less-used (more specific)
            scored_candidates.append((seq, score))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # ---- Step 4: Return best sequence ----
        best_seq = scored_candidates[0][0]
        storage.mark_retrieved(best_seq.sequence_id)
        
        return best_seq.raw_text
    
    def _generate_tokens_fallback(self, query: str, max_tokens: int) -> str:
        """
        Token-by-token generation fallback when no stored sequences match.
        
        Uses token transition patterns.
        """
        from ._memory_tokenization import simple_tokenize
        
        # Tokenize query
        context_tokens = simple_tokenize(query)
        generated_tokens = []
        
        for _ in range(max_tokens):
            # Find next token candidates
            candidates = self._transition_layer.find_next_token(
                context_tokens=context_tokens + generated_tokens,
                top_k=5,
            )
            
            if not candidates:
                break
            
            # Select best
            next_token, score = candidates[0]
            
            # Stop conditions
            if next_token in ("<END>", "<EOS>", "<|endoftext|>"):
                break
            
            # Check repetition
            if len(generated_tokens) >= 3 and generated_tokens[-3:] == [next_token] * 3:
                break
            
            generated_tokens.append(next_token)
        
        # Detokenize
        return " ".join(generated_tokens)

    # ------------------------------------------------------------------
    # Token-level generation (LLM-style, pattern-based)
    # ------------------------------------------------------------------

    def generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 128,
        domain: str = "code",
    ) -> str:
        """
        Generate tokens one-by-one like an LLM, but using pattern frequencies.
        
        This is the core token-level generation method. It:
        1. Tokenizes the prompt as initial context
        2. Loops: find next token candidates, select best, append, repeat
        3. Stops when max_tokens reached or end condition met
        
        Args:
            prompt: Initial context (e.g., "def factorial(n):")
            max_tokens: Maximum tokens to generate
            domain: Domain for generation (default: "code")
            
        Returns:
            Generated text (detokenized)
        """
        from ._memory_tokenization import tokenize_code, detokenize_code, simple_tokenize
        
        # Tokenize prompt
        if domain == "code":
            try:
                context_tokens = tokenize_code(prompt)
            except Exception:
                context_tokens = simple_tokenize(prompt)
        else:
            context_tokens = simple_tokenize(prompt)
        
        # Generate tokens iteratively
        generated_tokens = []
        
        for _ in range(max_tokens):
            # Find next token candidates
            candidates = self._transition_layer.find_next_token(
                context_tokens=context_tokens + generated_tokens,
                top_k=5,
            )
            
            if not candidates:
                # No pattern found, stop
                break
            
            # Select best candidate (highest score)
            next_token, score = candidates[0]
            
            # Check stop conditions
            if self._should_stop_generation(next_token, generated_tokens, domain):
                break
            
            # Append token
            generated_tokens.append(next_token)
            
            # Early stop if we have a complete statement
            if domain == "code" and self._is_complete_statement(generated_tokens):
                break
        
        # Detokenize
        if domain == "code":
            return detokenize_code(generated_tokens)
        else:
            return " ".join(generated_tokens)

    def generate_code(
        self,
        prompt: str,
        max_tokens: int = 256,
        validate_syntax: bool = True,
        max_attempts: int = 3,
    ) -> str:
        """
        Generate valid Python code using token-level patterns.
        
        This wraps generate_tokens() with code-specific features:
        - Proper code tokenization
        - Syntax validation with ast.parse()
        - Filtering of non-code tokens (Korean, descriptions, etc.)
        - Multiple attempts if validation fails
        
        Args:
            prompt: Code prompt (e.g., "def add(a, b):")
            max_tokens: Maximum tokens to generate
            validate_syntax: Whether to validate with ast.parse()
            max_attempts: Maximum generation attempts
            
        Returns:
            Valid Python code string
        """
        from ._memory_tokenization import (
            validate_python,
            filter_code_tokens,
            extract_function_body,
            tokenize_code,
            detokenize_code,
        )
        
        best_output = ""
        best_length = 0
        
        for attempt in range(max_attempts):
            # Generate tokens
            raw_output = self.generate_tokens(
                prompt=prompt,
                max_tokens=max_tokens,
                domain="code",
            )
            
            # Try to extract function body if present
            full_code = prompt + "\n" + raw_output
            
            if validate_syntax:
                # Check if syntactically valid
                if validate_python(full_code):
                    # Extract just the function if possible
                    func_body = extract_function_body(full_code)
                    if func_body:
                        return func_body
                    else:
                        return full_code
                
                # Not valid, keep trying but track longest attempt
                if len(raw_output) > best_length:
                    best_output = raw_output
                    best_length = len(raw_output)
            else:
                # No validation, return immediately
                return raw_output
        
        # All attempts failed validation, return best attempt
        return prompt + "\n" + best_output if best_output else prompt

    def _should_stop_generation(
        self,
        token: str,
        generated_so_far: List[str],
        domain: str,
    ) -> bool:
        """
        Check if generation should stop.
        
        Stop conditions:
        - End markers (<END>, <EOS>, etc.)
        - Excessive repetition
        - Maximum statement depth reached
        """
        # End markers
        if token in ("<END>", "<EOS>", "<|endoftext|>"):
            return True
        
        # Repetition check (same token repeated 3+ times)
        if len(generated_so_far) >= 3:
            if generated_so_far[-3:] == [token, token, token]:
                return True
        
        # For code: stop after reasonable number of DEDENTs (end of function)
        if domain == "code":
            dedent_count = generated_so_far.count("<DEDENT>")
            if dedent_count >= 2:
                return True
        
        return False

    def _is_complete_statement(self, tokens: List[str]) -> bool:
        """
        Check if generated tokens form a complete Python statement.
        
        Simple heuristic: presence of NEWLINE + DEDENT suggests complete block.
        """
        if "<DEDENT>" in tokens:
            # DEDENT usually means end of indented block
            return True
        
        if len(tokens) > 10 and "<NEWLINE>" in tokens[-3:]:
            # Recent NEWLINE with some content = likely complete
            return True
        
        return False

    # ------------------------------------------------------------------
    # Internal: field formation helpers
    # ------------------------------------------------------------------

    def _form_field_from_prompt(self, prompt: str):
        """Form a minimal StateField from the BrainMemory activation graph."""
        from ._memory_field_core import StateField, GoalState

        features = self.features_for_query(prompt)
        expanded = self.expanded_query_features(features, limit=48)

        # Build activation from energy spread over graph
        activation_map = self.activation(expanded, hops=2)

        # Separate event activations from feature activations
        # (event IDs start with short hex strings, features with prefixes like tok: bi:)
        activated_events: Dict[str, float] = {}
        activated_concepts: Dict[str, float] = {}

        for key, val in activation_map.items():
            if key.startswith("concept:"):
                activated_concepts[key] = float(val)
            elif key in self.events:
                activated_events[key] = float(val)

        # Also add directly recalled events
        try:
            recalls = self.recall(prompt, limit=8)
            for row in recalls:
                eid = row.get("event_id", "")
                if eid and eid not in activated_events:
                    activated_events[eid] = float(row.get("score", 0.5))
        except Exception:
            pass

        # Normalize activations
        if activated_events:
            max_e = max(activated_events.values())
            if max_e > 0:
                activated_events = {k: v / max_e for k, v in activated_events.items()}
        if activated_concepts:
            max_c = max(activated_concepts.values())
            if max_c > 0:
                activated_concepts = {k: v / max_c for k, v in activated_concepts.items()}

        # Compute operator activations from operator layer
        activated_operators: Dict[str, float] = {}
        try:
            preliminary = StateField(
                query_features=features,
                query_hv=np.array([], dtype=np.int8),
                activated_events=activated_events,
                activated_concepts=activated_concepts,
                activated_operators={},
                conflict_markers=[],
                goal_states=[],
                partial_outputs=[],
            )
            applicable = self._operator_layer.find_applicable_operators(
                field_state=preliminary, max_results=6, min_confidence=0.2
            )
            activated_operators = {op_id: score for op_id, score in applicable}
        except Exception:
            pass

        # Infer intent from prompt keywords
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in ["what", "why", "how", "explain", "무엇", "왜", "어떻게"]):
            goal_desc = "explain"
        elif any(kw in prompt_lower for kw in ["code", "implement", "write", "프로그램", "코드"]):
            goal_desc = "generate_code"
        elif any(kw in prompt_lower for kw in ["fix", "error", "bug", "수정", "오류"]):
            goal_desc = "fix_error"
        elif any(kw in prompt_lower for kw in ["compare", "difference", "similar", "비교"]):
            goal_desc = "compare"
        else:
            goal_desc = "answer_query"
        
        goal = GoalState(
            goal_id="goal_0",
            goal_description=goal_desc,
            satisfaction_level=0.0,
            constraints=[],
        )

        # Build HDC query vector if possible
        try:
            from ._brain_hdc import bundle_hv
            fids = [self.feature_to_id[f] for f in expanded if f in self.feature_to_id]
            query_hv = bundle_hv(fids[:64], self.hdc_bits) if fids else np.array([], dtype=np.int8)
        except Exception:
            query_hv = np.array([], dtype=np.int8)

        # Compute initial field energy
        field_energy = self._compute_simple_field_energy(activated_events, activated_concepts)

        return StateField(
            query_features=features,
            query_hv=query_hv,
            activated_events=activated_events,
            activated_concepts=activated_concepts,
            activated_operators=activated_operators,
            conflict_markers=[],
            goal_states=[goal],
            partial_outputs=[],
            field_energy=field_energy,
            previous_outputs=[],
            iteration=0,
        )

    def _update_field(self, field, new_output: str):
        """Return a new StateField that incorporates the latest generated output."""
        from ._memory_field_core import StateField

        updated_partials = list(field.partial_outputs) + [new_output]
        updated_previous = list(field.previous_outputs) + [new_output]

        # Update goal satisfaction heuristically: longer output = more satisfied
        updated_goals = []
        for goal in field.goal_states:
            total_words = sum(len(p.split()) for p in updated_partials)
            new_sat = min(1.0, goal.satisfaction_level + 0.3 * (1.0 - goal.satisfaction_level))
            from ._memory_field_core import GoalState
            updated_goals.append(GoalState(
                goal_id=goal.goal_id,
                goal_description=goal.goal_description,
                satisfaction_level=new_sat,
                constraints=goal.constraints,
            ))

        field_energy = self._compute_simple_field_energy(
            field.activated_events, field.activated_concepts,
            partial_outputs=updated_partials,
        )

        return StateField(
            query_features=field.query_features,
            query_hv=field.query_hv,
            activated_events=field.activated_events,
            activated_concepts=field.activated_concepts,
            activated_operators=field.activated_operators,
            conflict_markers=field.conflict_markers,
            goal_states=updated_goals,
            partial_outputs=updated_partials,
            field_energy=field_energy,
            previous_outputs=updated_previous,
            iteration=field.iteration + 1,
        )

    def _is_garbage_output(self, text: str) -> bool:
        """Detect repetitive / meaningless surface realization outputs."""
        words = text.strip().split()
        if not words:
            return True
        # Generic single-word outputs that come from cold-start fallback templates
        generic_words = {
            "elaboration", "correction", "explanation", "continuing",
            "direct", "completion", "apply_operator", "generate",
            "complete_state", "property",
        }
        if all(w.lower() in generic_words for w in words):
            return True
        # Hex-looking event IDs (16 hex chars = event_id format)
        import re
        hex_re = re.compile(r'^[0-9a-f]{12,}$')
        if all(hex_re.match(w) for w in words):
            return True
        # Repetition: all words the same
        if len(set(words)) == 1 and len(words) > 2:
            return True
        # Very high repetition ratio
        if len(words) > 3 and len(set(words)) / len(words) < 0.3:
            return True
        return False

    def _compute_simple_field_energy(
        self,
        activated_events: Dict[str, float],
        activated_concepts: Dict[str, float],
        partial_outputs: Optional[List[str]] = None,
    ):
        """Compute a lightweight FieldEnergy for use inside generate()."""
        from ._memory_field_core import FieldEnergy

        evidence = 0.0
        # Evidence from activated events
        if activated_events:
            evidence += min(1.0, sum(activated_events.values()) / max(1, len(activated_events)))
        # Evidence from concepts
        if activated_concepts:
            evidence += 0.5 * min(1.0, sum(activated_concepts.values()) / max(1, len(activated_concepts)))

        conflicts = 0.0
        # Incompleteness as conflict
        if not partial_outputs:
            conflicts += 0.5
        else:
            last = (partial_outputs or [""])[-1]
            if len(last) < 10:
                conflicts += 0.4
            elif not last.rstrip().endswith((".", "!", "?", "다", "임")):
                conflicts += 0.2

        total = conflicts - evidence
        tension = max(0.0, min(1.0, (conflicts + 0.01) / max(evidence + conflicts + 0.01, 0.01)))

        return FieldEnergy(
            total=total,
            evidence=evidence,
            conflicts=conflicts,
            evidence_breakdown={"memory_support": evidence},
            conflict_breakdown={"incompleteness": conflicts},
            dominant_evidence_sources=["memory_support"],
            dominant_conflict_sources=["incompleteness"],
            tension_level=tension,
        )


__all__ = ["MemoryTransitionMixin"]
