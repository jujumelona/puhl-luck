"""
Surface Realization Layer

Converts abstract cognitive states into concrete surface forms (text, code, etc.).
This is the final step in generation: transforming internal field states into
actual output strings.

The layer solves the critical problem: "We have selected what to say (state),
now how do we say it (surface form)?"

Key principle: NOT retrieval (copying stored text), NOT random token generation,
but REALIZATION - mapping abstract meaning to surface strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._brain_hdc import bundle_hv, hv_similarity
from ._memory_field_core import Candidate, CandidateSource, StateField


@dataclass
class SurfaceForm:
    """
    A concrete surface realization of an abstract state.
    
    Attributes:
        text: The actual output string
        modality: Output modality (text, code, etc.)
        confidence: How well this realizes the intended state (0-1)
        source: Where this form came from (generated, retrieved, template)
        features: Surface features (words, tokens, etc.)
    """
    text: str
    modality: str
    confidence: float
    source: str
    features: List[str]


class SurfaceRealizationLayer:
    """
    Layer 8: Surface Realization
    
    Converts abstract candidate states into concrete text/code/output.
    
    Process:
    1. Receive candidate state (intent + object + relation + context)
    2. Find similar past realizations (how did we express this before?)
    3. Generate template-based alternatives (structural patterns)
    4. Compose from atomic surface elements (words, phrases, code tokens)
    5. Select best realization based on fluency, precision, context-fit
    
    This is NOT:
    - Retrieving and copying stored sentences
    - Random token generation (no meaning grounding)
    - Template filling with slots
    
    This IS:
    - Grounded generation: state → surface mapping
    - Compositional: build from learned surface patterns
    - Context-aware: adapt to discourse state
    """
    
    def __init__(self):
        """Initialize surface realization layer."""
        # Store mappings: state patterns → surface forms
        self.state_to_surface: Dict[str, List[SurfaceForm]] = {}
        
        # Store atomic surface patterns (words, phrases, code templates)
        self.surface_atoms: Dict[str, Dict[str, List[str]]] = {
            "text": {},      # Korean/English phrases
            "code": {},      # Code templates
            "explanation": {},  # Explanation patterns
        }
        
        # HDC index for fast pattern matching
        self.pattern_index: Dict[Tuple[int, int], List[str]] = {}
        
    def realize(
        self,
        candidate: Candidate,
        state_field: StateField,
        modality: str = "text",
        max_alternatives: int = 3,
    ) -> SurfaceForm:
        """
        Realize a candidate state as concrete surface form.
        
        Main realization pipeline:
        1. Extract meaning from candidate (intent, object, relation)
        2. Find similar past realizations
        3. Generate template-based forms
        4. Compose from atomic elements
        5. Score alternatives by fluency, precision, fit
        6. Return best realization
        
        Args:
            candidate: The candidate state to realize
            state_field: Current cognitive field state (for context)
            modality: Output modality (text, code, explanation)
            max_alternatives: Number of alternatives to generate
            
        Returns:
            Best surface realization
        """
        # Transition candidates already carry the observed/constructed surface continuation.
        # Return it directly to avoid unrelated partial surface matches.
        if candidate.source == CandidateSource.TRANSITION_BASED and candidate.content and candidate.content.strip():
            text = candidate.content.strip()
            return SurfaceForm(
                text=text,
                modality=modality,
                confidence=max(0.0, min(1.0, candidate.confidence)),
                source="candidate_surface",
                features=text.lower().split(),
            )

        # Extract meaning structure from candidate
        meaning = self._extract_meaning(candidate, state_field)
        
        # Generate alternative realizations
        alternatives = []
        
        # Strategy 1: Retrieve similar past realizations
        retrieved = self._retrieve_similar_realizations(meaning, modality, max_alternatives)
        alternatives.extend(retrieved)
        
        # Strategy 2: Template-based generation
        template_based = self._generate_from_templates(meaning, modality, max_alternatives)
        alternatives.extend(template_based)
        
        # Strategy 3: Compositional generation from atoms
        compositional = self._compose_from_atoms(meaning, modality, max_alternatives)
        alternatives.extend(compositional)
        
        # If no alternatives, use fallback
        if not alternatives:
            return self._fallback_realization(candidate, modality)
        
        # Score and select best realization
        best = self._select_best_realization(alternatives, candidate, state_field)
        
        # Store this realization for future use
        self._store_realization(meaning, best)
        
        return best
    
    def learn_surface_form(
        self,
        state_pattern: str,
        surface_text: str,
        modality: str = "text",
        features: Optional[List[str]] = None,
    ) -> None:
        """
        Learn a new state → surface mapping.
        
        Called during exposure when we observe:
        partial state → continuation → completed state + actual output
        
        Args:
            state_pattern: Abstract state pattern (intent, object, relation)
            surface_text: The actual surface form
            modality: Output modality
            features: Optional surface features
        """
        if features is None:
            features = surface_text.lower().split()
        
        surface_form = SurfaceForm(
            text=surface_text,
            modality=modality,
            confidence=1.0,  # Perfect confidence (observed data)
            source="observed",
            features=features,
        )
        
        # Store mapping
        if state_pattern not in self.state_to_surface:
            self.state_to_surface[state_pattern] = []
        self.state_to_surface[state_pattern].append(surface_form)
        
        # Extract and store atomic surface patterns
        self._extract_surface_atoms(surface_text, modality, features)
    
    def _extract_meaning(
        self,
        candidate: Candidate,
        state_field: StateField,
    ) -> Dict[str, any]:
        """
        Extract meaning structure from candidate.
        
        Returns structure like:
        {
            'intent': 'explain',
            'object': 'HDC indexing',
            'relation': 'speedup',
            'context': ['field formation', 'candidate selection'],
            'features': [...],
        }
        """
        # Extract features from candidate content
        features = candidate.content.lower().split()
        
        # Extract activated concepts from field (context)
        context_concepts = list(state_field.activated_concepts.keys())[:5]
        
        # Infer intent from candidate source
        if candidate.source == "operator":
            intent = "apply_operator"
        elif candidate.source == "transition":
            intent = "complete_state"
        else:
            intent = "generate"
        
        # Extract object (main topic) from features
        # Simple heuristic: most activated concept
        if state_field.activated_concepts:
            object_concept = max(
                state_field.activated_concepts.items(),
                key=lambda x: x[1]
            )[0]
        else:
            object_concept = features[0] if features else "unknown"
        
        # Extract relation (what we're saying about object)
        # From candidate content or field goals
        relation = "property"
        if state_field.goal_states:
            goal_desc = state_field.goal_states[0].goal_description
            if "explain" in goal_desc:
                relation = "explanation"
            elif "compare" in goal_desc:
                relation = "comparison"
            elif "define" in goal_desc:
                relation = "definition"
        
        return {
            'intent': intent,
            'object': object_concept,
            'relation': relation,
            'context': context_concepts,
            'features': features,
            'candidate_content': candidate.content,
        }
    
    def _retrieve_similar_realizations(
        self,
        meaning: Dict,
        modality: str,
        max_results: int,
    ) -> List[SurfaceForm]:
        """
        Retrieve similar past realizations from memory.
        
        NOT copying - adapting. We find how similar meanings were realized
        and adapt them to current context.
        """
        # Create pattern signature
        pattern = f"{meaning['intent']}:{meaning['object']}:{meaning['relation']}"
        
        # Direct lookup
        if pattern in self.state_to_surface:
            similar = self.state_to_surface[pattern]
            return [
                SurfaceForm(
                    text=sf.text,
                    modality=sf.modality,
                    confidence=sf.confidence * 0.9,  # Slight reduction (not exact match)
                    source="retrieved_similar",
                    features=sf.features,
                )
                for sf in similar[:max_results]
            ]
        
        # Partial match fallback
        results = []
        for stored_pattern, surface_forms in self.state_to_surface.items():
            # Check if any component matches
            if (meaning['intent'] in stored_pattern or
                meaning['object'] in stored_pattern or
                meaning['relation'] in stored_pattern):
                
                for sf in surface_forms:
                    if sf.modality == modality and len(results) < max_results:
                        results.append(
                            SurfaceForm(
                                text=sf.text,
                                modality=sf.modality,
                                confidence=sf.confidence * 0.7,  # Lower confidence (partial match)
                                source="retrieved_partial",
                                features=sf.features,
                            )
                        )
        
        return results
    
    def _generate_from_templates(
        self,
        meaning: Dict,
        modality: str,
        max_results: int,
    ) -> List[SurfaceForm]:
        """
        Generate using structural templates.
        
        Templates capture realization patterns:
        - explain(X) → "X는 ... 이다"
        - compare(X,Y) → "X와 Y의 차이점은 ..."
        - define(X) → "X의 정의는 ..."
        """
        templates = {
            'text': {
                'explain': [
                    "{object}는 {property}하다",
                    "{object}의 목적은 {property}이다",
                    "{object}를 통해 {property}할 수 있다",
                ],
                'compare': [
                    "{object}와 {context_0}의 차이는 {property}이다",
                    "{object}는 {context_0}보다 {property}하다",
                ],
                'define': [
                    "{object}는 {property}하는 시스템이다",
                    "{object}의 정의: {property}",
                ],
                'complete_state': [
                    "{candidate_content}",  # Use candidate content as-is
                ],
            },
            'code': {
                'implement': [
                    "def {object}():\\n    {property}",
                    "class {object}:\\n    {property}",
                ],
            },
        }
        
        # Get templates for this modality and relation
        modality_templates = templates.get(modality, {})
        relation_templates = modality_templates.get(meaning['relation'], [])
        
        if not relation_templates:
            # Fallback: use candidate content directly
            relation_templates = ["{candidate_content}"]
        
        results = []
        for template in relation_templates[:max_results]:
            try:
                # Fill template with meaning components
                text = template.format(
                    object=meaning['object'],
                    property=meaning.get('features', [''])[0] if meaning.get('features') else '',
                    context_0=meaning['context'][0] if meaning['context'] else '',
                    candidate_content=meaning.get('candidate_content', ''),
                )
                
                results.append(
                    SurfaceForm(
                        text=text,
                        modality=modality,
                        confidence=0.8,
                        source="template",
                        features=text.lower().split(),
                    )
                )
            except (KeyError, IndexError):
                continue
        
        return results
    
    def _compose_from_atoms(
        self,
        meaning: Dict,
        modality: str,
        max_results: int,
    ) -> List[SurfaceForm]:
        """
        Compose surface form from atomic elements.
        
        Build output by combining learned surface atoms (words, phrases, patterns).
        """
        # For now, use simple composition
        # In full implementation, this would use learned phrase patterns
        
        object_str = meaning['object']
        relation_str = meaning['relation']
        features_str = ' '.join(meaning['features'][:3])
        
        # Compose simple sentence
        composed_text = f"{object_str} {relation_str} {features_str}"
        
        return [
            SurfaceForm(
                text=composed_text,
                modality=modality,
                confidence=0.6,
                source="compositional",
                features=composed_text.lower().split(),
            )
        ]
    
    def _select_best_realization(
        self,
        alternatives: List[SurfaceForm],
        candidate: Candidate,
        state_field: StateField,
    ) -> SurfaceForm:
        """
        Select best surface realization from alternatives.
        
        Scoring criteria:
        - Confidence: How well it realizes the state
        - Fluency: How natural the output sounds
        - Precision: How precisely it captures meaning
        - Context-fit: How well it fits current discourse state
        """
        if not alternatives:
            return self._fallback_realization(candidate, "text")
        
        scored = []
        for alt in alternatives:
            score = (
                alt.confidence * 0.4 +  # Base confidence
                self._score_fluency(alt.text) * 0.3 +  # Fluency
                self._score_precision(alt, candidate) * 0.2 +  # Precision
                self._score_context_fit(alt, state_field) * 0.1  # Context fit
            )
            scored.append((score, alt))
        
        # Return highest scoring
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    
    def _score_fluency(self, text: str) -> float:
        """Score how natural/fluent the text is."""
        # Simple heuristic: not too short, not too long, has substance
        if len(text) < 5:
            return 0.3
        if len(text) > 200:
            return 0.7
        return 0.9
    
    def _score_precision(self, surface_form: SurfaceForm, candidate: Candidate) -> float:
        """Score how precisely this realizes the candidate."""
        # Check feature overlap
        candidate_features = set(candidate.content.lower().split())
        surface_features = set(surface_form.features)
        
        if not candidate_features:
            return 0.5
        
        overlap = len(candidate_features.intersection(surface_features))
        return overlap / len(candidate_features)
    
    def _score_context_fit(self, surface_form: SurfaceForm, state_field: StateField) -> float:
        """Score how well this fits current discourse context."""
        # Check if surface features match activated concepts
        surface_features = set(surface_form.features)
        activated_concepts = set(state_field.activated_concepts.keys())
        
        if not activated_concepts:
            return 0.5
        
        overlap = len(surface_features.intersection(activated_concepts))
        return min(1.0, overlap / 3)  # Normalize
    
    def _fallback_realization(self, candidate: Candidate, modality: str) -> SurfaceForm:
        """Fallback when no realizations available."""
        return SurfaceForm(
            text=candidate.content if candidate.content else "...",
            modality=modality,
            confidence=0.3,
            source="fallback",
            features=candidate.content.lower().split() if candidate.content else [],
        )
    
    def _store_realization(self, meaning: Dict, surface_form: SurfaceForm) -> None:
        """Store successful realization for future retrieval."""
        pattern = f"{meaning['intent']}:{meaning['object']}:{meaning['relation']}"
        
        if pattern not in self.state_to_surface:
            self.state_to_surface[pattern] = []
        
        # Store with slightly lower confidence (generated, not observed)
        stored_form = SurfaceForm(
            text=surface_form.text,
            modality=surface_form.modality,
            confidence=surface_form.confidence * 0.9,
            source=f"generated_{surface_form.source}",
            features=surface_form.features,
        )
        self.state_to_surface[pattern].append(stored_form)
    
    def _extract_surface_atoms(
        self,
        surface_text: str,
        modality: str,
        features: List[str],
    ) -> None:
        """
        Extract atomic surface patterns from observed text.
        
        Learns reusable surface elements (words, phrases, patterns).
        """
        # Extract bigrams and trigrams as atomic patterns
        tokens = features
        
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            if bigram not in self.surface_atoms[modality]:
                self.surface_atoms[modality][bigram] = []
            if surface_text not in self.surface_atoms[modality][bigram]:
                self.surface_atoms[modality][bigram].append(surface_text)
        
        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            if trigram not in self.surface_atoms[modality]:
                self.surface_atoms[modality][trigram] = []
            if surface_text not in self.surface_atoms[modality][trigram]:
                self.surface_atoms[modality][trigram].append(surface_text)


__all__ = ["SurfaceRealizationLayer", "SurfaceForm"]
