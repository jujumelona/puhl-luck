"""
Layer 6: Candidate Emergence

Implements tension-driven candidate generation where candidates emerge from field
dynamics rather than being retrieved through search. Candidates are generated to
reduce field energy by addressing conflicts, satisfying goals, and completing
incomplete states.

This is the core of the paradigm shift from retrieval-based to field-based generation:
instead of searching for candidates and then ranking them, candidates emerge directly
from high-tension areas in the cognitive field.

Requirements:
- 3.1: Generate candidates from incomplete cognitive field states
- 3.2: Compute field tension as high energy
- 3.3: Identify tension-reducing continuations
- 3.4: Use free energy minimization as generation principle
- 3.5: Generate candidates that reduce conflict markers
- 3.6: Generate candidates that increase evidence accumulation
- 3.7: Generate candidates that satisfy active goals
- 3.8: Drive candidate generation at creation time
- 13.1-13.7: Tension-driven generation principle
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set

import numpy as np

from ._memory_field_core import (
    Candidate,
    CandidateSource,
    StateField,
    TensionSource,
    TensionType,
)

if TYPE_CHECKING:
    from ._memory_field_energy import FreeEnergyMinimization
    from ._memory_operator_layer import OperatorInstance, OperatorMemoryLayer
    from ._memory_transition_layer import TransitionMemoryLayer
    from ._memory_surface_realization import SurfaceRealizationLayer


class CandidateEmergence:
    """
    Process that generates output candidates from field tension.
    
    Unlike traditional candidate retrieval systems that search for and rank
    candidates, this class generates candidates directly from the cognitive
    field dynamics. Candidates emerge to reduce field energy by:
    - Resolving conflicts between activated memories
    - Satisfying unsatisfied goals
    - Completing incomplete outputs
    - Increasing evidence accumulation
    
    The generation process is tension-driven: high field energy (from conflicts,
    incompleteness, unsatisfied goals) drives candidate generation. Free energy
    minimization guides the generation at creation time, not as a post-generation
    scoring function.
    
    Requirements:
    - 3.1: Generate from incomplete field states, not retrieval
    - 3.2: High tension from incomplete information
    - 3.3: Identify tension-reducing continuations
    - 3.4: Free energy minimization as generation principle
    - 3.5-3.7: Generate to reduce conflicts, increase evidence, satisfy goals
    - 3.8: Drive generation at creation time
    - 13.1-13.7: Tension-driven generation
    """
    
    def __init__(
        self,
        energy_computer: Optional[FreeEnergyMinimization] = None,
        surface_layer: Optional[SurfaceRealizationLayer] = None,
    ):
        """
        Initialize the candidate emergence mechanism.
        
        Args:
            energy_computer: Optional FreeEnergyMinimization instance for computing
                           predicted energy reduction. If None, creates a default instance.
            surface_layer: Optional SurfaceRealizationLayer for accessing stored surface forms
        """
        # Import here to avoid circular dependency
        if energy_computer is None:
            from ._memory_field_energy import FreeEnergyMinimization
            energy_computer = FreeEnergyMinimization()
        
        self.energy_computer = energy_computer
        self._surface_layer = surface_layer
        
        # Generation statistics
        self.total_candidates_generated = 0
        self.generation_stats: Dict[CandidateSource, int] = {
            CandidateSource.OPERATOR_BASED: 0,
            CandidateSource.TRANSITION_BASED: 0,
            CandidateSource.HYBRID: 0,
            CandidateSource.FIELD_DYNAMICS: 0,
        }
    
    def generate_candidates(
        self,
        field: StateField,
        operators_layer: OperatorMemoryLayer,
        transitions_layer: TransitionMemoryLayer,
        num_candidates: int = 10,
    ) -> List[Candidate]:
        """
        Generate candidates from field tension.
        
        This is the main entry point for candidate generation. It:
        1. Identifies tension sources in the current field
        2. Generates candidates using operators, transitions, and field dynamics
        3. Computes predicted energy reduction for each candidate
        4. Returns candidates sorted by energy reduction (highest reduction first)
        
        Requirements:
        - 3.1: Generate from incomplete field states, not retrieval
        - 3.3: Identify tension-reducing continuations
        - 13.1: Compute tension from incomplete state markers
        - 13.7: Generate directly from field dynamics
        
        Args:
            field: Current cognitive field state
            operators_layer: Layer 3 for accessing stored operators
            transitions_layer: Layer 4 for accessing stored transitions
            num_candidates: Target number of candidates to generate
            
        Returns:
            List of Candidate objects sorted by energy reduction potential
        """
        # Step 1: Identify tension sources
        tension_sources = self.identify_tension_sources(field)
        
        if not tension_sources:
            # No tension: field is stable, no candidates needed
            return []
        
        # Step 2: Generate candidates from multiple sources
        candidates: List[Candidate] = []
        
        # Generate from operators (if applicable operators exist)
        operator_candidates = self.generate_from_operators(
            field, operators_layer, tension_sources
        )
        candidates.extend(operator_candidates)
        
        # Generate from transitions (if similar transitions exist)
        transition_candidates = self.generate_from_transitions(
            field, transitions_layer, tension_sources
        )
        candidates.extend(transition_candidates)
        
        # Generate from field dynamics (resonance-based generation)
        dynamics_candidates = self.generate_from_field_dynamics(
            field, tension_sources
        )
        candidates.extend(dynamics_candidates)
        
        # Step 3: Compute energy reduction for each candidate
        for candidate in candidates:
            candidate.energy_reduction = self.compute_energy_reduction(
                candidate, field
            )
            candidate.predicted_energy_after = (
                field.field_energy.total - candidate.energy_reduction
                if field.field_energy is not None
                else 0.0
            )
        
        # Step 4: De-duplicate by surface content and sort by energy + confidence.
        unique: Dict[str, Candidate] = {}
        for cand in candidates:
            key = " ".join(cand.content.strip().split())
            if not key:
                continue
            prev = unique.get(key)
            if prev is None or (cand.energy_reduction, cand.confidence) > (prev.energy_reduction, prev.confidence):
                unique[key] = cand
        candidates = list(unique.values())
        candidates.sort(key=lambda c: (c.energy_reduction, c.confidence), reverse=True)
        
        # Step 5: Return top candidates
        top_candidates = candidates[:num_candidates]
        
        # Update statistics
        self.total_candidates_generated += len(top_candidates)
        for candidate in top_candidates:
            self.generation_stats[candidate.source] += 1
        
        return top_candidates
    
    def identify_tension_sources(self, field: StateField) -> List[TensionSource]:
        """
        Identify sources of tension in the current field.
        
        Tension sources indicate high field energy from:
        - Conflicts between memories
        - Unsatisfied goals
        - Incomplete partial outputs
        
        Requirements:
        - 3.2: Compute field tension from incompleteness
        - 13.1: Compute tension from incomplete state markers
        - 13.2: Generate when tension exceeds threshold
        
        Args:
            field: Current cognitive field state
            
        Returns:
            List of TensionSource objects describing what needs to be addressed
        """
        tensions: List[TensionSource] = []
        
        # Tension source 1: Conflicts
        # Each conflict marker represents a tension that needs resolution
        for i, conflict in enumerate(field.conflict_markers):
            tensions.append(
                TensionSource(
                    type=TensionType.CONFLICT,
                    location=f"conflict_{i}",
                    strength=conflict.strength,
                    description=f"{conflict.conflict_type.value}: {conflict.description}",
                )
            )
        
        # Tension source 2: Unsatisfied goals
        # Goals with low satisfaction create tension
        for goal in field.goal_states:
            if goal.satisfaction_level < 0.8:  # Goal not fully satisfied
                unsatisfaction = 1.0 - goal.satisfaction_level
                tensions.append(
                    TensionSource(
                        type=TensionType.UNSATISFIED_GOAL,
                        location=goal.goal_id,
                        strength=unsatisfaction,
                        description=f"Unsatisfied goal: {goal.goal_description} "
                                  f"(satisfaction: {goal.satisfaction_level:.2f})",
                    )
                )
        
        # Tension source 3: Incomplete outputs
        # Partial outputs indicate generation is not complete
        if field.partial_outputs:
            last_output = field.partial_outputs[-1]
            
            # Heuristics for incompleteness:
            # - Very short outputs (< 10 chars) are likely incomplete
            # - Outputs not ending with sentence terminators are incomplete
            is_incomplete = False
            incompleteness_strength = 0.0
            
            if len(last_output) < 10:
                is_incomplete = True
                incompleteness_strength = 0.7
            elif last_output and not last_output.rstrip().endswith(('.', '!', '?', ':', ';')):
                is_incomplete = True
                incompleteness_strength = 0.5
            
            if is_incomplete:
                tensions.append(
                    TensionSource(
                        type=TensionType.INCOMPLETE_OUTPUT,
                        location="partial_output",
                        strength=incompleteness_strength,
                        description=f"Incomplete output: '{last_output[:50]}...'",
                    )
                )
        
        # Tension source 4: High field energy
        # Overall high energy indicates general incompleteness
        if field.field_energy is not None and field.field_energy.tension_level > 0.6:
            tensions.append(
                TensionSource(
                    type=TensionType.INCOMPLETE_OUTPUT,
                    location="field_energy",
                    strength=field.field_energy.tension_level,
                    description=f"High field tension: {field.field_energy.tension_level:.2f}",
                )
            )
        
        return tensions
    
    def generate_from_operators(
        self,
        field: StateField,
        operators_layer: OperatorMemoryLayer,
        tension_sources: List[TensionSource],
    ) -> List[Candidate]:
        """
        Generate candidates by applying activated operators.
        
        Operators provide learned transformation patterns that can be instantiated
        with context-specific parameters to generate tension-reducing continuations.
        
        Requirements:
        - 3.3: Generate tension-reducing continuations
        - 4.7: Apply stored operators to novel inputs
        - 9.1: Identify applicable operators
        - 9.2: Instantiate with context-appropriate parameters
        
        Args:
            field: Current cognitive field state
            operators_layer: Layer 3 containing stored operators
            tension_sources: Identified tension sources to address
            
        Returns:
            List of Candidate objects generated from operators
        """
        candidates: List[Candidate] = []
        
        # Find applicable operators for current field state
        applicable = operators_layer.find_applicable_operators(
            field_state=field,
            max_results=5,
            min_confidence=0.3,
        )
        
        if not applicable:
            return candidates
        
        # Generate candidates from each applicable operator
        for operator_id, match_score in applicable:
            # Instantiate operator with context
            instance = operators_layer.instantiate_operator(operator_id, field)
            
            if instance is None:
                continue
            
            # Generate candidate content from operator
            candidate_content = self._apply_operator_instance(instance, field)
            
            if not candidate_content:
                continue
            
            # Determine which tension sources this candidate addresses
            tensions_addressed = self._match_tensions_to_operator(
                instance, tension_sources
            )
            
            # Create candidate
            candidate = Candidate(
                content=candidate_content,
                tokens=candidate_content.split(),  # Simple tokenization
                energy_reduction=0.0,  # Will be computed later
                predicted_energy_after=0.0,  # Will be computed later
                source=CandidateSource.OPERATOR_BASED,
                source_operators=[operator_id],
                source_transitions=[],
                tensions_addressed=tensions_addressed,
                tensions_resolved_count=len(tensions_addressed),
                confidence=instance.match_score,
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def generate_from_transitions(
        self,
        field: StateField,
        transitions_layer: TransitionMemoryLayer,
        tension_sources: List[TensionSource],
    ) -> List[Candidate]:
        """
        Generate candidates using similar state transitions.
        
        **CRITICAL FIX**: Use stored surface forms, NOT raw complete_state tokens.
        
        Finds transitions where the partial state is similar to the current field,
        and uses the actual observed surface continuation as candidate.
        
        Requirements:
        - 3.4: Use transitions to guide completion
        - 5.5: Apply same completion algorithm across domains
        - 5.6: Use completion dynamics from transitions
        
        Args:
            field: Current cognitive field state
            transitions_layer: Layer 4 containing stored transitions
            tension_sources: Identified tension sources to address
            
        Returns:
            List of Candidate objects generated from transitions
        """
        candidates: List[Candidate] = []
        
        # Find similar transitions based on current partial state
        similar = transitions_layer.find_similar_transitions(
            current_partial=field,
            top_k=5,
        )
        
        if not similar:
            return candidates
        
        # Get surface layer to retrieve actual surface forms
        surface_layer = getattr(self, '_surface_layer', None)
        if surface_layer is None:
            # Try to get from parent if exists
            surface_layer = getattr(transitions_layer, '_surface_layer', None)
        
        # Generate candidates from each similar transition
        for transition_id, similarity in similar:
            transition = transitions_layer.get_transition(transition_id)
            if transition is None:
                continue
            
            # **STRATEGY 1**: Try to get stored surface form
            candidate_content = None
            if surface_layer is not None:
                # Build state pattern from partial features
                partial_features = transition.partial_state.query_features[:10]
                state_pattern = "|".join(sorted(partial_features))
                
                # Look up stored surface forms
                surface_forms = surface_layer.state_to_surface.get(state_pattern, [])
                if surface_forms:
                    # Use the most confident surface form
                    best_surface = max(surface_forms, key=lambda s: s.confidence)
                    candidate_content = best_surface.text.strip()
            
            # **STRATEGY 2**: Extract actual answer tokens (not just features)
            if not candidate_content:
                # Get completion features (what was ADDED, not complete state)
                completion_features = transition.completion_features
                if completion_features:
                    # Extract tokens from completion features
                    completion_tokens = []
                    for feat in completion_features:
                        if feat.startswith("tok:"):
                            completion_tokens.append(feat.split(":", 1)[1])
                    
                    if completion_tokens:
                        # Use only the NEW tokens (continuation), not all complete tokens
                        candidate_content = " ".join(dict.fromkeys(completion_tokens[:32]))
            
            # **STRATEGY 3**: Fallback to completion pattern
            if not candidate_content:
                pattern = transitions_layer.get_completion_pattern(transition_id)
                if pattern:
                    candidate_content = self._apply_completion_pattern(pattern, field)
            
            # Create candidate if we have content
            if candidate_content and candidate_content.strip():
                # Determine which tension sources this addresses
                tensions_addressed = []
                for tension in tension_sources:
                    if tension.type in (TensionType.INCOMPLETE_OUTPUT, TensionType.UNSATISFIED_GOAL):
                        tensions_addressed.append(tension)
                
                pattern = transitions_layer.get_completion_pattern(transition_id)
                reliability = pattern.reliability if pattern else 0.5
                
                candidate = Candidate(
                    content=candidate_content,
                    tokens=candidate_content.split(),
                    energy_reduction=0.0,  # Will be computed later
                    predicted_energy_after=0.0,  # Will be computed later
                    source=CandidateSource.TRANSITION_BASED,
                    source_operators=[],
                    source_transitions=[transition_id],
                    tensions_addressed=tensions_addressed,
                    tensions_resolved_count=len(tensions_addressed),
                    confidence=similarity * reliability,
                )
                candidates.append(candidate)
                
                # Update relevance for successful use
                transitions_layer.update_relevance(transition_id, increment=1)
            
        
        return candidates
    
    def generate_from_field_dynamics(
        self,
        field: StateField,
        tension_sources: List[TensionSource],
    ) -> List[Candidate]:
        """
        Generate candidates directly from field resonance patterns.
        
        When operators and transitions don't provide sufficient candidates,
        generates candidates based on the resonance patterns in the activated
        field. High-resonance memory combinations suggest natural continuations.
        
        Requirements:
        - 3.1: Generate from field states
        - 13.7: Generate directly from field dynamics
        - 14.5: Use resonance to amplify coherent patterns
        
        Args:
            field: Current cognitive field state
            tension_sources: Identified tension sources to address
            
        Returns:
            List of Candidate objects generated from field dynamics
        """
        candidates: List[Candidate] = []
        
        # Find high-resonance memory combinations
        high_resonance_pairs = self._find_high_resonance_patterns(field)
        
        if not high_resonance_pairs:
            return candidates
        
        # Generate candidates from resonance patterns
        for mem1, mem2, resonance in high_resonance_pairs[:3]:  # Top 3 patterns
            # Generate candidate by combining high-resonance memories
            candidate_content = self._generate_from_resonance(
                mem1, mem2, resonance, field
            )
            
            if not candidate_content:
                continue
            
            # Determine which tensions this addresses (heuristic)
            tensions_addressed = []
            for tension in tension_sources:
                # Field dynamics candidates generally address incompleteness
                if tension.type == TensionType.INCOMPLETE_OUTPUT:
                    tensions_addressed.append(tension)
            
            # Create candidate
            candidate = Candidate(
                content=candidate_content,
                tokens=candidate_content.split(),
                energy_reduction=0.0,  # Will be computed later
                predicted_energy_after=0.0,  # Will be computed later
                source=CandidateSource.FIELD_DYNAMICS,
                source_operators=[],
                source_transitions=[],
                tensions_addressed=tensions_addressed,
                tensions_resolved_count=len(tensions_addressed),
                confidence=abs(resonance) * 0.7,  # Lower confidence for dynamics
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def compute_energy_reduction(
        self,
        candidate: Candidate,
        field: StateField,
    ) -> float:
        """
        Compute predicted energy reduction if candidate is incorporated.
        
        Uses the free energy minimization component to estimate how much
        the field energy would decrease if this candidate is added to the
        field state.
        
        **CRITICAL FIX**: Add strong query relevance penalty to prevent
        off-topic candidates (e.g., CSV answer when asked about HDC).
        
        Requirements:
        - 3.3: Identify tension-reducing continuations
        - 3.4: Use free energy minimization as generation principle
        
        Args:
            candidate: Candidate to evaluate
            field: Current field state
            
        Returns:
            Predicted energy reduction (positive = energy decreases)
        """
        if field.field_energy is None:
            # No current energy: cannot compute reduction
            return 0.0
        
        current_energy = field.field_energy.total
        
        # Predict energy after incorporating candidate
        predicted_energy = self.energy_computer.predict_energy_after_update(
            field, candidate.content
        )
        
        # Energy reduction = current - predicted
        # Positive values mean energy decreases (good)
        energy_reduction = current_energy - predicted_energy
        
        # **CRITICAL**: Apply query relevance penalty
        # Normalize BrainMemory feature strings (e.g. tok:csv, bi:save_file)
        # to plain lexical units before comparing with candidate words.
        def _norm_feature(x: str) -> str:
            x = str(x).lower()
            # Features can be nested (text:tok:csv). Keep the lexical tail.
            if ":" in x:
                x = x.rsplit(":", 1)[1]
            return x.replace("_", " ").strip()

        candidate_features = set()
        for raw in candidate.content.lower().replace("`", " ").split():
            raw = raw.strip(".,:;()[]{}<>\"'")
            if raw:
                candidate_features.add(raw)

        query_features = set()
        for feat in field.query_features:
            nf = _norm_feature(feat)
            if not nf:
                continue
            query_features.add(nf)
            for part in nf.split():
                if part:
                    query_features.add(part)
        
        # Compute relevance overlap with substring fallback for Korean/compound tokens.
        if query_features and candidate_features:
            overlap = 0
            for c in candidate_features:
                if c in query_features or any((c in q or q in c) and min(len(c), len(q)) >= 2 for q in query_features):
                    overlap += 1
            total = len(candidate_features.union(query_features))
            relevance = overlap / max(total, 1)
        else:
            relevance = 0.0
        
        # Strong penalty for irrelevant candidates (relevance < 0.1)
        relevance_penalty = 0.0
        if relevance < 0.1:
            # Very irrelevant: -5.0 penalty
            relevance_penalty = -5.0
        elif relevance < 0.3:
            # Somewhat irrelevant: -2.0 penalty
            relevance_penalty = -2.0
        elif relevance < 0.5:
            # Weak relevance: -0.5 penalty
            relevance_penalty = -0.5
        
        # Apply penalty
        energy_reduction += relevance_penalty
        
        return energy_reduction
    
    # =========================================================================
    # Internal helper methods
    # =========================================================================
    
    def _apply_operator_instance(
        self,
        instance: OperatorInstance,
        field: StateField,
    ) -> str:
        """
        Apply an operator instance to generate candidate content.
        
        Uses the operator's transformation rule and context bindings to
        produce output text.
        
        Args:
            instance: Instantiated operator with bindings
            field: Current field state
            
        Returns:
            Generated candidate text
        """
        operator = instance.operator
        transformation = operator.transformation
        bindings = instance.bindings
        
        # Apply transformation based on rule type
        if transformation.rule_type == "template":
            # Template-based: fill in template with bindings
            template = operator.completion_template
            
            # Simple template substitution
            result = template
            for key, value in bindings.items():
                placeholder = f"{{{key}}}"
                if placeholder in result:
                    # Convert value to string
                    if isinstance(value, list):
                        value_str = ", ".join(str(v) for v in value[:3])  # First 3 items
                    else:
                        value_str = str(value)
                    result = result.replace(placeholder, value_str)
            
            # Don't return if still fully unresolved (still has ALL original placeholders)
            # Partial resolution is OK
            if result == template and "{" in result:
                # Template was not resolved at all — try filling with query_features
                features_str = ", ".join(str(f) for f in bindings.get("query_features", [])[:3])
                result = template.replace("{query_features}", features_str)
                # Strip any remaining unresolved placeholders
                import re as _re
                result = _re.sub(r'\{[^}]+\}', '', result).strip()
            
            return result if result.strip() else ""
        
        elif transformation.rule_type == "learned_transition":
            # Use top activated concepts/features as actual output tokens
            if "top_concepts" in bindings and bindings["top_concepts"]:
                tokens = [c.replace("concept:", "") for c in bindings["top_concepts"][:5]]
                tokens = [t for t in tokens if t]
                if tokens:
                    return " ".join(tokens)
            if "query_features" in bindings and bindings["query_features"]:
                tokens = []
                for feat in bindings["query_features"][:10]:
                    if feat.startswith(("tok:", "stem:")):
                        tokens.append(feat.split(":", 1)[1])
                    elif feat.startswith(("bi:", "tri:")):
                        tokens.append(feat.split(":", 1)[1].replace("_", " "))
                    elif ":" not in feat and len(feat) > 1:
                        tokens.append(feat)
                if tokens:
                    return " ".join(dict.fromkeys(tokens[:8]))
            return ""
        
        elif transformation.rule_type == "feature_combination":
            # Combine top features into output
            if "query_features" in bindings:
                tokens = []
                for feat in bindings["query_features"][:8]:
                    if feat.startswith(("tok:", "stem:")):
                        tokens.append(feat.split(":", 1)[1])
                    elif ":" not in feat and len(feat) > 1:
                        tokens.append(feat)
                if tokens:
                    return " ".join(dict.fromkeys(tokens))
            return ""
        
        else:
            # Unknown rule type: return empty
            return ""
    
    def _apply_completion_pattern(
        self,
        pattern: CompletionPattern,
        field: StateField,
    ) -> str:
        """
        Apply a completion pattern to generate candidate content.
        
        Uses the pattern's added features and completion type to generate
        appropriate continuation.
        
        Args:
            pattern: Completion pattern to apply
            field: Current field state
            
        Returns:
            Generated candidate text
        """
        # Primary: use added features as actual token content
        if pattern.added_features:
            # Filter out pure HDC feature identifiers, keep readable tokens
            readable = []
            for feat in pattern.added_features[:20]:
                # Keep tok:, bi:, tri: prefixed features (strip prefix)
                if feat.startswith(("tok:", "stem:")):
                    readable.append(feat.split(":", 1)[1])
                elif feat.startswith(("bi:", "tri:")):
                    readable.append(feat.split(":", 1)[1].replace("_", " "))
                elif ":" not in feat and len(feat) > 1:
                    readable.append(feat)
            if readable:
                return " ".join(dict.fromkeys(readable[:12]))  # deduplicate, max 12 tokens

        # Generate based on completion type
        if pattern.completion_type == "direct":
            if pattern.added_features:
                return " ".join(f.split(":", 1)[-1] for f in pattern.added_features[:10])
            return ""

        elif pattern.completion_type == "elaboration":
            if pattern.added_concepts:
                concepts = [c.replace("concept:", "") for c in list(pattern.added_concepts)[:3]]
                return " ".join(concepts)
            return ""

        elif pattern.completion_type == "correction":
            if field.conflict_markers:
                return field.conflict_markers[0].description
            return ""

        elif pattern.completion_type == "explanation":
            if pattern.added_features:
                key_feature = pattern.added_features[0].split(":", 1)[-1]
                return key_feature
            return ""

        return ""
    
    def _match_tensions_to_operator(
        self,
        instance: OperatorInstance,
        tensions: List[TensionSource],
    ) -> List[TensionSource]:
        """
        Determine which tension sources an operator instance addresses.
        
        Args:
            instance: Operator instance
            tensions: List of tension sources
            
        Returns:
            Subset of tensions that this operator addresses
        """
        addressed = []
        operator = instance.operator
        
        for tension in tensions:
            # Match operator type to tension type
            if tension.type == TensionType.CONFLICT:
                # Repair operators address conflicts
                if operator.operator_type.value == "repair":
                    addressed.append(tension)
            
            elif tension.type == TensionType.UNSATISFIED_GOAL:
                # Completion and explanation operators address goals
                if operator.operator_type.value in ["completion", "explanation"]:
                    addressed.append(tension)
            
            elif tension.type == TensionType.INCOMPLETE_OUTPUT:
                # Completion operators address incompleteness
                if operator.operator_type.value == "completion":
                    addressed.append(tension)
        
        return addressed
    
    def _match_tensions_to_pattern(
        self,
        pattern: CompletionPattern,
        tensions: List[TensionSource],
    ) -> List[TensionSource]:
        """
        Determine which tension sources a completion pattern addresses.
        
        Args:
            pattern: Completion pattern
            tensions: List of tension sources
            
        Returns:
            Subset of tensions that this pattern addresses
        """
        addressed = []
        
        for tension in tensions:
            # Match pattern type to tension type
            if tension.type == TensionType.CONFLICT and pattern.completion_type == "correction":
                addressed.append(tension)
            
            elif tension.type == TensionType.UNSATISFIED_GOAL:
                # Most patterns can address goals
                if pattern.completion_type in ["direct", "explanation", "elaboration"]:
                    addressed.append(tension)
            
            elif tension.type == TensionType.INCOMPLETE_OUTPUT:
                # All patterns address incompleteness
                addressed.append(tension)
        
        return addressed
    
    def _find_high_resonance_patterns(
        self,
        field: StateField,
    ) -> List[Tuple[str, str, float]]:
        """
        Find high-resonance memory pairs in the field.
        
        Args:
            field: Current field state
            
        Returns:
            List of (mem1_id, mem2_id, resonance_score) tuples sorted by resonance
        """
        if not field.resonance:
            return []
        
        # Collect all resonance pairs
        resonance_pairs = []
        for (mem1, mem2), resonance in field.resonance.items():
            # Only positive resonance (mutual support)
            if resonance > 0.3:
                resonance_pairs.append((mem1, mem2, resonance))
        
        # Sort by resonance strength (descending)
        resonance_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return resonance_pairs
    
    def _generate_from_resonance(
        self,
        mem1: str,
        mem2: str,
        resonance: float,
        field: StateField,
    ) -> str:
        """
        Generate candidate content from high-resonance memory pair.
        
        Args:
            mem1: First memory identifier
            mem2: Second memory identifier
            resonance: Resonance strength
            field: Current field state
            
        Returns:
            Generated candidate text
        """
        # Extract memory types and identifiers
        mem1_activation = field.activated_events.get(mem1) or field.activated_concepts.get(mem1) or 0.0
        mem2_activation = field.activated_events.get(mem2) or field.activated_concepts.get(mem2) or 0.0
        
        # Generate based on memory types
        # For concepts, use concept names
        if mem1.startswith("concept:") and mem2.startswith("concept:"):
            concept1 = mem1.replace("concept:", "")
            concept2 = mem2.replace("concept:", "")
            return f"Combining {concept1} and {concept2}"
        
        # For events, use generic continuation
        return f"Resonance-based continuation (strength: {resonance:.2f})"


__all__ = ["CandidateEmergence"]
