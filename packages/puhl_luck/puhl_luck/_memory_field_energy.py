"""
Field energy computation and free energy minimization for Predictive Field Memory.

This module implements the FreeEnergyMinimization class that computes field energy
from the cognitive field state. Energy = conflicts - evidence, where lower energy
indicates more stable, coherent states.

Evidence sources (decrease energy):
- Memory resonance: Activated memories support each other
- Goal satisfaction: Goals are being satisfied
- Constraint satisfaction: Constraints are met
- Memory support: Strong backing from memory
- Coherence: Internal consistency

Conflict sources (increase energy):
- Contradiction: Memories conflict
- Constraint violation: Constraints violated
- Repetition: Unwanted repetition
- Incompleteness: Missing information
- Goal conflict: Goals conflict with each other

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ._memory_field_core import (
    CONFLICT_SOURCES,
    EVIDENCE_SOURCES,
    ConflictType,
    FieldEnergy,
    StateField,
)

# Try to import Rust implementations
try:
    import puhl_luck_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class FreeEnergyMinimization:
    """
    Computes field energy and guides stabilization toward lower energy states.
    
    The free energy principle drives the cognitive field toward stable, coherent
    states by minimizing field energy. Energy is computed as:
    
        Energy = Conflicts - Evidence
    
    Lower energy indicates:
    - High evidence: Memories resonate, goals satisfied, constraints met
    - Low conflicts: No contradictions, violations, or repetitions
    
    Higher energy indicates:
    - Low evidence: Weak memory support, unsatisfied goals
    - High conflicts: Contradictions, violations, incompleteness
    
    Requirements:
    - 7.1: Compute field energy as conflicts minus evidence
    - 7.2: Increase evidence when activated memories resonate
    - 7.3: Increase evidence when field has consistent goal-satisfying patterns
    - 7.4: Increase evidence when field has strong memory support
    - 7.5: Increase conflict when memories contradict each other
    - 7.6: Increase conflict when field contains constraint violations
    - 7.7: Increase conflict when field contains repetition patterns
    - 7.8, 7.9: Free energy acts as state field property driving stabilization
    - 17.1-17.8: Provide energy decomposition showing component breakdowns
    """

    def __init__(
        self,
        evidence_weights: Dict[str, float] | None = None,
        conflict_weights: Dict[str, float] | None = None,
    ):
        """
        Initialize the free energy minimization component.
        
        Args:
            evidence_weights: Optional weights for evidence sources (default: uniform)
            conflict_weights: Optional weights for conflict sources (default: uniform)
        """
        # Default equal weighting for all evidence sources
        self.evidence_weights = evidence_weights or {
            "memory_resonance": 1.0,
            "goal_satisfaction": 1.0,
            "constraint_satisfaction": 1.0,
            "memory_support": 1.0,
            "coherence": 1.0,
        }
        
        # Default equal weighting for all conflict sources
        self.conflict_weights = conflict_weights or {
            "contradiction": 1.0,
            "constraint_violation": 1.0,
            "repetition": 1.0,
            "incompleteness": 1.0,
            "goal_conflict": 1.0,
        }

    def compute_field_energy(self, field: StateField) -> FieldEnergy:
        """
        Compute total field energy with detailed breakdown.
        
        Energy = Conflicts - Evidence
        
        Lower energy is more stable. Field dynamics drive toward lower energy states
        through iterative candidate emergence and field updates.
        
        This is the main entry point for energy computation, providing complete
        decomposition of evidence and conflict sources.
        
        Requirements:
        - 7.1: Compute field energy as conflicts minus evidence
        - 7.8: Free energy drives field toward lower energy states
        - 17.1: Provide energy decomposition showing evidence components
        - 17.2: Provide energy decomposition showing conflict components
        
        Args:
            field: Current cognitive field state
            
        Returns:
            FieldEnergy with total energy, components, and detailed breakdown
        """
        # Use Rust for core computation if available
        if RUST_AVAILABLE and hasattr(field, 'conflicts') and hasattr(field, 'goals'):
            # Convert to Rust format
            conflict_list = [
                (c.source_id, c.target_id, c.strength)
                for c in field.conflicts
            ]
            evidence_list = []  # Simplified - full implementation would extract from field
            
            # Fast Rust computation
            total_energy_rust = puhl_luck_core.compute_field_energy_rust(
                conflict_list, evidence_list, 1.0, 1.0
            )
        
        # Compute evidence and conflicts with breakdowns
        evidence, evidence_breakdown = self.compute_evidence(field)
        conflicts, conflict_breakdown = self.compute_conflicts(field)
        
        # Total energy = conflicts - evidence
        total_energy = conflicts - evidence
        
        # Identify dominant sources
        dominant_evidence = self._get_dominant_sources(evidence_breakdown, top_k=3)
        dominant_conflicts = self._get_dominant_sources(conflict_breakdown, top_k=3)
        
        # Compute normalized tension level [0, 1]
        # High tension = high conflicts, low evidence
        # Tension is normalized by maximum possible conflict + maximum possible evidence
        max_possible = len(EVIDENCE_SOURCES) + len(CONFLICT_SOURCES)
        tension_level = (conflicts + (1.0 - evidence)) / max_possible if max_possible > 0 else 0.0
        tension_level = min(max(tension_level, 0.0), 1.0)  # Clamp to [0, 1]
        
        return FieldEnergy(
            total=total_energy,
            evidence=evidence,
            conflicts=conflicts,
            evidence_breakdown=evidence_breakdown,
            conflict_breakdown=conflict_breakdown,
            dominant_evidence_sources=dominant_evidence,
            dominant_conflict_sources=dominant_conflicts,
            tension_level=tension_level,
        )

    def compute_evidence(self, field: StateField) -> Tuple[float, Dict[str, float]]:
        """
        Compute evidence score with breakdown by source.
        
        Evidence decreases field energy and represents support for the current
        field state from memory, goals, and constraints.
        
        Evidence sources:
        1. Memory resonance: How strongly activated memories support each other
        2. Goal satisfaction: How well current state satisfies active goals
        3. Constraint satisfaction: How well current state satisfies constraints
        4. Memory support: How strongly memories back the current state
        5. Coherence: Internal consistency of the field
        
        Requirements:
        - 7.2: Increase evidence when activated memories resonate with input
        - 7.3: Increase evidence when field has consistent goal-satisfying patterns
        - 7.4: Increase evidence when field has strong memory support
        - 17.3: Compute evidence from memory resonance strength
        - 17.4: Compute evidence from goal satisfaction level
        - 17.5: Compute evidence from constraint satisfaction level
        
        Args:
            field: Current cognitive field state
            
        Returns:
            Tuple of (total_evidence, evidence_breakdown_dict)
        """
        breakdown: Dict[str, float] = {}
        
        # 1. Memory resonance evidence (Requirement 7.2, 17.3)
        breakdown["memory_resonance"] = self._compute_memory_resonance_evidence(field)
        
        # 2. Goal satisfaction evidence (Requirement 7.3, 17.4)
        breakdown["goal_satisfaction"] = self._compute_goal_satisfaction_evidence(field)
        
        # 3. Constraint satisfaction evidence (Requirement 17.5)
        breakdown["constraint_satisfaction"] = self._compute_constraint_satisfaction_evidence(field)
        
        # 4. Memory support evidence (Requirement 7.4)
        breakdown["memory_support"] = self._compute_memory_support_evidence(field)
        
        # 5. Coherence evidence
        breakdown["coherence"] = self._compute_coherence_evidence(field)
        
        # Compute weighted total
        total_evidence = sum(
            breakdown[source] * self.evidence_weights.get(source, 1.0)
            for source in breakdown
        )
        
        return total_evidence, breakdown

    def compute_conflicts(self, field: StateField) -> Tuple[float, Dict[str, float]]:
        """
        Compute conflict score with breakdown by source.
        
        Conflicts increase field energy and represent tensions, incompatibilities,
        and violations in the current field state.
        
        Conflict sources:
        1. Contradiction: Activated memories contradict each other
        2. Constraint violation: Hard constraints are violated
        3. Repetition: Unwanted repetitive patterns detected
        4. Incompleteness: Missing information or partial outputs
        5. Goal conflict: Active goals conflict with each other
        
        Requirements:
        - 7.5: Increase conflict when activated memories contradict each other
        - 7.6: Increase conflict when field contains constraint violations
        - 7.7: Increase conflict when field contains repetition patterns
        - 17.6: Compute conflict from contradiction detection
        - 17.7: Compute conflict from constraint violation detection
        - 17.8: Compute conflict from repetition detection
        
        Args:
            field: Current cognitive field state
            
        Returns:
            Tuple of (total_conflicts, conflict_breakdown_dict)
        """
        breakdown: Dict[str, float] = {}
        
        # 1. Contradiction conflicts (Requirement 7.5, 17.6)
        breakdown["contradiction"] = self._compute_contradiction_conflicts(field)
        
        # 2. Constraint violation conflicts (Requirement 7.6, 17.7)
        breakdown["constraint_violation"] = self._compute_constraint_violation_conflicts(field)
        
        # 3. Repetition conflicts (Requirement 7.7, 17.8)
        breakdown["repetition"] = self._compute_repetition_conflicts(field)
        
        # 4. Incompleteness conflicts (implied by Requirements 3.2, 13.1)
        breakdown["incompleteness"] = self._compute_incompleteness_conflicts(field)
        
        # 5. Goal conflict
        breakdown["goal_conflict"] = self._compute_goal_conflict(field)
        
        # Compute weighted total
        total_conflicts = sum(
            breakdown[source] * self.conflict_weights.get(source, 1.0)
            for source in breakdown
        )
        
        return total_conflicts, breakdown

    def predict_energy_after_update(
        self, field: StateField, candidate_content: str
    ) -> float:
        """
        Predict what field energy would be after incorporating a candidate.
        
        This is used during candidate emergence to estimate which candidates
        will reduce field energy most effectively.
        
        NOTE: This is a simplified prediction that doesn't create a full updated
        field. It estimates energy reduction based on:
        - How the candidate addresses unsatisfied goals
        - How it resolves conflicts
        - How it completes incomplete outputs
        
        Args:
            field: Current field state
            candidate_content: Proposed candidate text
            
        Returns:
            Predicted energy value after incorporating candidate
        """
        # Start with current energy
        current_energy = self.compute_field_energy(field)
        predicted_energy = current_energy.total
        
        # Estimate energy reduction from addressing goals
        for goal in field.goal_states:
            if goal.satisfaction_level < 1.0:
                # Simple heuristic: check if candidate mentions goal keywords
                goal_keywords = goal.goal_description.lower().split()
                candidate_lower = candidate_content.lower()
                
                mentions = sum(1 for kw in goal_keywords if kw in candidate_lower)
                if mentions > 0:
                    # Reduce energy proportional to keyword matches
                    predicted_energy -= 0.1 * mentions
        
        # Estimate energy reduction from resolving conflicts
        for conflict in field.conflict_markers:
            if conflict.conflict_type == ConflictType.CONTRADICTION:
                # If candidate might address the contradiction
                # (very simple heuristic - could be improved)
                predicted_energy -= 0.05 * conflict.strength
        
        # Estimate energy reduction from completing partial outputs
        if field.partial_outputs and len(field.partial_outputs[-1]) < 10:
            # If last output was incomplete and candidate adds content
            if len(candidate_content) > 5:
                predicted_energy -= 0.2
        
        return predicted_energy

    # =========================================================================
    # Evidence computation helpers
    # =========================================================================

    def _compute_memory_resonance_evidence(self, field: StateField) -> float:
        """
        Compute evidence from memory resonance patterns.
        
        High resonance between activated memories indicates they mutually
        support each other, providing strong evidence for the field state.
        
        Requirement 7.2, 17.3: Evidence from memory resonance strength
        
        Args:
            field: Current field state
            
        Returns:
            Memory resonance evidence score (0.0 to 1.0)
        """
        if not field.resonance:
            return 0.0
        
        # Compute average positive resonance
        positive_resonances = [r for r in field.resonance.values() if r > 0]
        
        if not positive_resonances:
            return 0.0
        
        avg_resonance = sum(positive_resonances) / len(positive_resonances)
        
        # Normalize to [0, 1] - assuming resonance values are typically in [-1, 1]
        return min(max(avg_resonance, 0.0), 1.0)

    def _compute_goal_satisfaction_evidence(self, field: StateField) -> float:
        """
        Compute evidence from goal satisfaction levels.
        
        When goals are satisfied, the field state has high evidence supporting it.
        Unsatisfied goals create tension that drives candidate generation.
        
        Requirement 7.3, 17.4: Evidence from goal satisfaction
        
        Args:
            field: Current field state
            
        Returns:
            Goal satisfaction evidence score (0.0 to 1.0)
        """
        if not field.goal_states:
            # No goals means no goal-based evidence, but not a conflict
            return 0.5  # Neutral
        
        # Average satisfaction across all goals
        total_satisfaction = sum(goal.satisfaction_level for goal in field.goal_states)
        avg_satisfaction = total_satisfaction / len(field.goal_states)
        
        return avg_satisfaction

    def _compute_constraint_satisfaction_evidence(self, field: StateField) -> float:
        """
        Compute evidence from constraint satisfaction.
        
        When all constraints are satisfied, field has high evidence.
        Violated constraints reduce evidence and increase conflicts.
        
        Requirement 17.5: Evidence from constraint satisfaction
        
        Args:
            field: Current field state
            
        Returns:
            Constraint satisfaction evidence score (0.0 to 1.0)
        """
        # Check for constraint violations in conflict markers
        constraint_violations = [
            c for c in field.conflict_markers
            if c.conflict_type == ConflictType.CONSTRAINT_VIOLATION
        ]
        
        if not constraint_violations:
            # No violations: full evidence
            return 1.0
        
        # Evidence decreases with number and strength of violations
        total_violation_strength = sum(c.strength for c in constraint_violations)
        
        # Reduce evidence proportionally (clamped to [0, 1])
        evidence = max(1.0 - (0.2 * total_violation_strength), 0.0)
        
        return evidence

    def _compute_memory_support_evidence(self, field: StateField) -> float:
        """
        Compute evidence from overall memory activation strength.
        
        Strong activation of many memories indicates good support from
        episodic memory for the current field state.
        
        Requirement 7.4: Evidence from strong memory support
        
        Args:
            field: Current field state
            
        Returns:
            Memory support evidence score (0.0 to 1.0)
        """
        # Collect all activations
        all_activations = (
            list(field.activated_events.values())
            + list(field.activated_concepts.values())
            + list(field.activated_operators.values())
        )
        
        if not all_activations:
            return 0.0
        
        # Memory support is based on:
        # 1. Number of activated memories (breadth)
        # 2. Average activation strength (depth)
        
        num_activated = len(all_activations)
        avg_activation = sum(all_activations) / num_activated
        
        # Normalize: high support when many memories are strongly activated
        # Use logarithmic scaling for number of activations to avoid over-emphasizing large numbers
        breadth_score = min(np.log1p(num_activated) / 5.0, 1.0)  # log(1+n)/5, capped at 1
        depth_score = avg_activation  # Already in [0, 1]
        
        # Combine breadth and depth (equal weighting)
        memory_support = 0.5 * breadth_score + 0.5 * depth_score
        
        return memory_support

    def _compute_coherence_evidence(self, field: StateField) -> float:
        """
        Compute evidence from internal field coherence.
        
        Coherence measures how well the field "hangs together" - whether
        activated memories, goals, and outputs form a consistent whole.
        
        Args:
            field: Current field state
            
        Returns:
            Coherence evidence score (0.0 to 1.0)
        """
        # Coherence factors:
        # 1. Low number of conflicts indicates coherence
        # 2. Positive resonance patterns indicate coherence
        # 3. Consistent progression in outputs indicates coherence
        
        # Factor 1: Conflict-based coherence
        num_conflicts = len(field.conflict_markers)
        conflict_penalty = min(num_conflicts * 0.1, 1.0)  # -0.1 per conflict, capped at 1.0
        
        # Factor 2: Resonance-based coherence
        if field.resonance:
            all_resonances = list(field.resonance.values())
            avg_resonance = sum(all_resonances) / len(all_resonances)
            resonance_boost = max(avg_resonance, 0.0)  # Only positive resonance contributes
        else:
            resonance_boost = 0.0
        
        # Factor 3: Output progression coherence
        if len(field.previous_outputs) > 1:
            # Outputs exist and are building on each other: positive coherence
            output_coherence = 0.3
        else:
            output_coherence = 0.0
        
        # Combine factors
        coherence = max(0.5 - conflict_penalty + 0.3 * resonance_boost + output_coherence, 0.0)
        coherence = min(coherence, 1.0)
        
        return coherence

    # =========================================================================
    # Conflict computation helpers
    # =========================================================================

    def _compute_contradiction_conflicts(self, field: StateField) -> float:
        """
        Compute conflicts from contradictions between activated memories.
        
        Contradictions occur when activated memories have negative resonance
        or are explicitly marked as conflicting.
        
        Requirement 7.5, 17.6: Conflict from contradictions
        
        Args:
            field: Current field state
            
        Returns:
            Contradiction conflict score (0.0+)
        """
        # Count explicit contradiction markers
        contradiction_markers = [
            c for c in field.conflict_markers
            if c.conflict_type == ConflictType.CONTRADICTION
        ]
        
        if not contradiction_markers:
            # Check for negative resonances as implicit contradictions
            if field.resonance:
                negative_resonances = [r for r in field.resonance.values() if r < -0.3]
                if negative_resonances:
                    # Implicit contradictions from negative resonance
                    avg_negative = abs(sum(negative_resonances) / len(negative_resonances))
                    return min(avg_negative, 1.0)
            return 0.0
        
        # Sum contradiction strengths
        total_contradiction = sum(c.strength for c in contradiction_markers)
        
        return total_contradiction

    def _compute_constraint_violation_conflicts(self, field: StateField) -> float:
        """
        Compute conflicts from constraint violations.
        
        Constraints are hard requirements that must be satisfied. Violations
        create high conflict that needs resolution.
        
        Requirement 7.6, 17.7: Conflict from constraint violations
        
        Args:
            field: Current field state
            
        Returns:
            Constraint violation conflict score (0.0+)
        """
        # Count constraint violation markers
        violation_markers = [
            c for c in field.conflict_markers
            if c.conflict_type == ConflictType.CONSTRAINT_VIOLATION
        ]
        
        if not violation_markers:
            return 0.0
        
        # Sum violation strengths
        total_violations = sum(c.strength for c in violation_markers)
        
        return total_violations

    def _compute_repetition_conflicts(self, field: StateField) -> float:
        """
        Compute conflicts from unwanted repetition patterns.
        
        Repetition in outputs or circular reasoning creates conflict that
        signals the field is stuck in a local minimum.
        
        Requirement 7.7, 17.8: Conflict from repetition
        
        Args:
            field: Current field state
            
        Returns:
            Repetition conflict score (0.0+)
        """
        # Check explicit repetition markers
        repetition_markers = [
            c for c in field.conflict_markers
            if c.conflict_type == ConflictType.REPETITION
        ]
        
        if repetition_markers:
            return sum(c.strength for c in repetition_markers)
        
        # Detect implicit repetition in outputs
        if len(field.previous_outputs) >= 2:
            # Check if recent outputs are very similar (simple heuristic)
            last_output = field.previous_outputs[-1].lower()
            second_last = field.previous_outputs[-2].lower()
            
            # If outputs are identical or very similar, that's repetition
            if last_output == second_last:
                return 0.5
            
            # Check for high overlap (>70% of words repeated)
            last_words = set(last_output.split())
            second_last_words = set(second_last.split())
            
            if last_words and second_last_words:
                overlap = len(last_words & second_last_words)
                total = len(last_words | second_last_words)
                overlap_ratio = overlap / total if total > 0 else 0.0
                
                if overlap_ratio > 0.7:
                    return 0.3  # Moderate repetition conflict
        
        return 0.0

    def _compute_incompleteness_conflicts(self, field: StateField) -> float:
        """
        Compute conflicts from incomplete information or partial outputs.
        
        Incompleteness creates tension that drives candidate generation.
        This is a primary source of field energy in generative tasks.
        
        Requirements implied by 3.2, 13.1: High incompleteness implies high energy
        
        Args:
            field: Current field state
            
        Returns:
            Incompleteness conflict score (0.0+)
        """
        conflict = 0.0
        
        # Factor 1: Unsatisfied goals indicate incompleteness
        unsatisfied_goals = [
            g for g in field.goal_states if g.satisfaction_level < 1.0
        ]
        if unsatisfied_goals:
            avg_unsatisfaction = sum(
                1.0 - g.satisfaction_level for g in unsatisfied_goals
            ) / len(unsatisfied_goals)
            conflict += avg_unsatisfaction
        
        # Factor 2: Partial outputs indicate incompleteness
        if field.partial_outputs:
            last_output = field.partial_outputs[-1]
            
            # Very short outputs are likely incomplete
            if len(last_output) < 10:
                conflict += 0.5
            
            # Outputs ending mid-sentence are incomplete
            if last_output and not last_output.rstrip().endswith(('.', '!', '?', ':', ';')):
                conflict += 0.3
        
        # Factor 3: Low number of iterations might indicate incompleteness
        # (early in generation process)
        if field.iteration < 2:
            conflict += 0.2
        
        return min(conflict, 2.0)  # Cap to avoid over-penalizing incompleteness

    def _compute_goal_conflict(self, field: StateField) -> float:
        """
        Compute conflicts from contradictory goals.
        
        When multiple goals are active and they conflict with each other,
        the field has high tension from goal incompatibility.
        
        Args:
            field: Current field state
            
        Returns:
            Goal conflict score (0.0+)
        """
        if len(field.goal_states) < 2:
            # Can't have goal conflicts with fewer than 2 goals
            return 0.0
        
        # Simple heuristic: check if goal descriptions contain contradictory terms
        # This is a placeholder - real implementation would need semantic analysis
        
        goal_texts = [g.goal_description.lower() for g in field.goal_states]
        
        # Check for obvious contradictions (e.g., "maximize" vs "minimize")
        contradictory_pairs = [
            ("maximize", "minimize"),
            ("increase", "decrease"),
            ("add", "remove"),
            ("expand", "reduce"),
        ]
        
        conflict_count = 0
        for g1 in goal_texts:
            for g2 in goal_texts:
                if g1 >= g2:  # Avoid double-counting
                    continue
                for word1, word2 in contradictory_pairs:
                    if word1 in g1 and word2 in g2:
                        conflict_count += 1
                    elif word2 in g1 and word1 in g2:
                        conflict_count += 1
        
        # Return normalized conflict
        return min(conflict_count * 0.5, 1.0)

    # =========================================================================
    # Utility helpers
    # =========================================================================

    def _get_dominant_sources(
        self, breakdown: Dict[str, float], top_k: int = 3
    ) -> List[str]:
        """
        Identify the dominant sources from a breakdown dictionary.
        
        Args:
            breakdown: Dictionary mapping source names to contribution values
            top_k: Number of top sources to return
            
        Returns:
            List of source names sorted by contribution (highest first)
        """
        sorted_sources = sorted(
            breakdown.items(), key=lambda x: x[1], reverse=True
        )
        return [source for source, _ in sorted_sources[:top_k]]


__all__ = ["FreeEnergyMinimization"]
