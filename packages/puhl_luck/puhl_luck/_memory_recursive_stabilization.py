"""
Layer 8: Recursive Stabilization

Implements the iterative field update loop that drives generation toward
convergence. This is the core of the predictive field memory architecture:
a loop that repeatedly forms fields, generates candidates, updates state,
and checks for convergence until the field reaches a stable low-energy state.

The stabilization loop replaces the traditional "retrieve-then-rank" pattern
with a dynamic field-based generation process.

Requirements:
- 6.1-6.6: Iterative field updates, convergence detection, oscillation handling
- 20.1-20.5: Convergence guarantees, max iterations, energy thresholds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from ._memory_field_core import (
    Candidate,
    ConvergenceReason,
    ConvergenceStats,
    FieldEnergy,
    InputContext,
    StateField,
)

if TYPE_CHECKING:
    from ._memory_cognitive_field import CognitiveField


@dataclass
class StabilizationResult:
    """
    Result of the recursive stabilization process.
    
    Attributes:
        final_output: Generated output text
        iterations: Number of iterations completed
        converged: Whether the field converged to stable state
        final_energy: Field energy at completion
        energy_history: Energy values across iterations
        convergence_stats: Statistics about convergence process
        intermediate_outputs: Outputs generated at each iteration
        oscillation_detected: Whether oscillation was detected
        damping_applied: Whether damping was applied to prevent oscillation
    """
    final_output: str
    iterations: int
    converged: bool
    final_energy: float
    energy_history: List[float]
    convergence_stats: ConvergenceStats
    intermediate_outputs: List[str] = field(default_factory=list)
    oscillation_detected: bool = False
    damping_applied: bool = False


class RecursiveStabilization:
    """
    Iterative field stabilization loop.
    
    Executes the core generation loop:
    1. Form cognitive field from input and previous state
    2. Generate candidates from field tension
    3. Select best candidate (highest energy reduction)
    4. Update field with selected output
    5. Check convergence
    6. Repeat until convergence or max iterations
    
    The loop implements predictive processing principles: the system
    generates outputs that minimize field energy (prediction error).
    
    Requirements:
    - 6.1: Execute iterative field update loop
    - 6.2: Monitor field energy across iterations
    - 6.3: Detect convergence when energy stabilizes
    - 6.4: Detect oscillation patterns
    - 6.5: Apply damping to prevent oscillation
    - 6.6: Enforce maximum iteration limits
    - 20.1-20.5: Convergence guarantees
    """
    
    def __init__(
        self,
        max_iterations: int = 20,
        convergence_threshold: float = 0.05,
        oscillation_window: int = 5,
        damping_factor: float = 0.7,
    ):
        """
        Initialize the recursive stabilization loop.
        
        Args:
            max_iterations: Maximum number of iterations before stopping
            convergence_threshold: Energy change threshold for convergence detection
            oscillation_window: Number of iterations to check for oscillation
            damping_factor: Factor to apply when damping oscillation (0-1)
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.oscillation_window = oscillation_window
        self.damping_factor = damping_factor
        
        # Statistics
        self.total_stabilizations = 0
        self.convergence_count = 0
        self.oscillation_count = 0
        self.timeout_count = 0
    
    def stabilize(
        self,
        initial_context: InputContext,
        cognitive_field: CognitiveField,
        custom_max_iterations: Optional[int] = None,
        custom_threshold: Optional[float] = None,
    ) -> StabilizationResult:
        """
        Execute recursive stabilization loop.
        
        Main generation algorithm:
        1. Form field from input (simultaneous activation of all layers)
        2. Generate candidates from field tension
        3. Select best candidate (highest energy reduction)
        4. Append candidate to output
        5. Update field with new output
        6. Check convergence
        7. Repeat
        
        Requirements:
        - 6.1: Execute iterative loop
        - 6.2: Monitor energy
        - 6.3: Detect convergence
        - 6.4: Detect oscillation
        - 6.5: Apply damping
        - 6.6: Enforce max iterations
        
        Args:
            initial_context: Input context to process
            cognitive_field: CognitiveField instance with all layers
            custom_max_iterations: Override default max iterations
            custom_threshold: Override default convergence threshold
            
        Returns:
            StabilizationResult with final output and convergence info
        """
        max_iter = custom_max_iterations or self.max_iterations
        threshold = custom_threshold or self.convergence_threshold
        
        # Initialize state
        current_context = initial_context
        energy_history: List[float] = []
        outputs: List[str] = []
        converged = False
        convergence_reason = ConvergenceReason.NOT_CONVERGED
        oscillation_detected = False
        damping_applied = False
        
        # Main stabilization loop
        for iteration in range(max_iter):
            # Step 1: Form cognitive field
            current_field = cognitive_field.form_field(current_context)
            
            # Compute and record field energy
            field_energy = cognitive_field.energy_computer.compute_field_energy(current_field)
            current_field.field_energy = field_energy
            energy_history.append(field_energy.total)
            
            # Step 2: Check convergence
            if iteration > 0:
                converged, reason = self.detect_convergence(energy_history, threshold)
                if converged:
                    convergence_reason = reason
                    break
            
            # Step 3: Check for oscillation
            if iteration >= self.oscillation_window:
                if self.detect_oscillation(energy_history):
                    oscillation_detected = True
                    # Apply damping
                    current_field = self.apply_damping(current_field)
                    damping_applied = True
            
            # Step 4: Generate candidates from field tension
            candidates = cognitive_field.candidate_emergence.generate_candidates(
                field=current_field,
                operators_layer=cognitive_field.operators_layer,
                transitions_layer=cognitive_field.transitions_layer,
                num_candidates=5,
            )
            
            # Step 5: Select best candidate (highest energy reduction)
            if not candidates:
                # No candidates: field is stable or stuck
                convergence_reason = ConvergenceReason.NO_CANDIDATES
                converged = True
                break
            
            best_candidate = candidates[0]  # Already sorted by energy reduction
            
            # Step 6: Append output
            outputs.append(best_candidate.content)
            
            # Step 7: Update context for next iteration
            # Accumulate outputs so far
            cumulative_output = " ".join(outputs)
            
            # Create new context with updated output
            current_context = InputContext(
                query_text=initial_context.query_text,
                query_features=initial_context.query_features,
                query_hv=initial_context.query_hv,
                modality=initial_context.modality,
                domain=initial_context.domain,
                partial_output=cumulative_output,
            )
            
            # Step 8: Check if output seems complete
            if self._output_seems_complete(cumulative_output):
                convergence_reason = ConvergenceReason.OUTPUT_COMPLETE
                converged = True
                break
        
        # Finalize
        if not converged:
            convergence_reason = ConvergenceReason.MAX_ITERATIONS
        
        final_output = " ".join(outputs) if outputs else ""
        final_energy = energy_history[-1] if energy_history else 0.0
        
        # Update statistics
        self.total_stabilizations += 1
        if converged:
            self.convergence_count += 1
        if oscillation_detected:
            self.oscillation_count += 1
        if convergence_reason == ConvergenceReason.MAX_ITERATIONS:
            self.timeout_count += 1
        
        # Create convergence stats
        conv_stats = ConvergenceStats(
            iterations=iteration + 1,
            final_energy=final_energy,
            energy_reduction=energy_history[0] - final_energy if energy_history else 0.0,
            convergence_reason=convergence_reason,
            oscillation_detected=oscillation_detected,
        )
        
        return StabilizationResult(
            final_output=final_output,
            iterations=iteration + 1,
            converged=converged,
            final_energy=final_energy,
            energy_history=energy_history,
            convergence_stats=conv_stats,
            intermediate_outputs=outputs,
            oscillation_detected=oscillation_detected,
            damping_applied=damping_applied,
        )
    
    def detect_convergence(
        self,
        energy_history: List[float],
        threshold: float,
    ) -> Tuple[bool, ConvergenceReason]:
        """
        Detect if field has converged to stable state.
        
        Convergence indicators:
        - Energy change below threshold for multiple iterations
        - Energy approaching zero (fully satisfied state)
        - Energy stabilizing (derivative close to zero)
        
        Requirements:
        - 6.3: Detect convergence from energy stabilization
        - 20.2: Energy change threshold
        
        Args:
            energy_history: List of energy values across iterations
            threshold: Energy change threshold for convergence
            
        Returns:
            Tuple of (converged: bool, reason: ConvergenceReason)
        """
        if len(energy_history) < 2:
            return False, ConvergenceReason.NOT_CONVERGED
        
        current_energy = energy_history[-1]
        
        # Check 1: Very low energy (nearly perfect state)
        if current_energy < 0.1:
            return True, ConvergenceReason.ENERGY_MINIMUM
        
        # Check 2: Energy change below threshold
        energy_change = abs(energy_history[-1] - energy_history[-2])
        if energy_change < threshold:
            # Verify stability over multiple iterations
            if len(energy_history) >= 3:
                prev_change = abs(energy_history[-2] - energy_history[-3])
                if prev_change < threshold:
                    return True, ConvergenceReason.ENERGY_STABLE
        
        # Check 3: Energy gradient approaching zero
        if len(energy_history) >= 4:
            recent_energies = energy_history[-4:]
            gradient = np.gradient(recent_energies)
            if np.abs(gradient[-1]) < threshold / 2:
                return True, ConvergenceReason.ENERGY_STABLE
        
        return False, ConvergenceReason.NOT_CONVERGED
    
    def detect_oscillation(self, energy_history: List[float]) -> bool:
        """
        Detect oscillation patterns in energy history.
        
        Oscillation occurs when energy alternates between similar values
        without progressing toward convergence. This indicates the field
        is stuck in a limit cycle.
        
        Requirements:
        - 6.4: Detect oscillation patterns
        - 20.4: Oscillation detection
        
        Args:
            energy_history: List of energy values across iterations
            
        Returns:
            True if oscillation detected, False otherwise
        """
        if len(energy_history) < self.oscillation_window:
            return False
        
        recent = energy_history[-self.oscillation_window:]
        
        # Check for alternating pattern
        # Oscillation signature: E[t] ~ E[t-2] ~ E[t-4], E[t-1] ~ E[t-3]
        even_indices = recent[::2]
        odd_indices = recent[1::2]
        
        if len(even_indices) >= 2 and len(odd_indices) >= 2:
            # Check if even values are similar to each other
            even_std = np.std(even_indices)
            odd_std = np.std(odd_indices)
            
            # Check if even and odd values are different
            even_mean = np.mean(even_indices)
            odd_mean = np.mean(odd_indices)
            mean_diff = abs(even_mean - odd_mean)
            
            # Oscillation: low variance within groups, high difference between groups
            if even_std < 0.1 and odd_std < 0.1 and mean_diff > 0.2:
                return True
        
        # Check for periodic pattern using autocorrelation
        if len(recent) >= 6:
            autocorr = np.correlate(recent, recent, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
            
            # Normalize
            autocorr = autocorr / autocorr[0]
            
            # Check for peaks at non-zero lags
            if len(autocorr) > 2:
                # Look for high correlation at lag 2 (period-2 oscillation)
                if autocorr[2] > 0.8:
                    return True
        
        return False
    
    def apply_damping(self, field: StateField) -> StateField:
        """
        Apply damping to field to prevent oscillation.
        
        Reduces activation strengths to dampen oscillatory dynamics.
        This helps the field escape limit cycles and progress toward
        stable convergence.
        
        Requirements:
        - 6.5: Apply damping when oscillation detected
        - 20.5: Damping mechanism
        
        Args:
            field: Current field state
            
        Returns:
            Damped field state
        """
        # Dampen all activation strengths
        damped_events = {
            event_id: activation * self.damping_factor
            for event_id, activation in field.activated_events.items()
        }
        
        damped_concepts = {
            concept_id: activation * self.damping_factor
            for concept_id, activation in field.activated_concepts.items()
        }
        
        damped_operators = {
            op_id: activation * self.damping_factor
            for op_id, activation in field.activated_operators.items()
        }
        
        # Create damped field
        damped_field = StateField(
            query_features=field.query_features,
            query_hv=field.query_hv,
            activated_events=damped_events,
            activated_concepts=damped_concepts,
            activated_operators=damped_operators,
            conflict_markers=field.conflict_markers,
            goal_states=field.goal_states,
            partial_outputs=field.partial_outputs,
            resonance=field.resonance,
            field_energy=field.field_energy,
        )
        
        return damped_field
    
    def _output_seems_complete(self, output: str) -> bool:
        """
        Heuristic check if output seems complete.
        
        Args:
            output: Cumulative output string
            
        Returns:
            True if output seems complete, False otherwise
        """
        if not output:
            return False
        
        # Check for sentence terminators
        if output.rstrip().endswith(('.', '!', '?', ':')):
            # Check minimum length
            if len(output) > 20:
                return True
        
        return False


__all__ = ["RecursiveStabilization", "StabilizationResult"]
