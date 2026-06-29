"""
Main Orchestrator: CognitiveField

Coordinates all four memory layers and the recursive stabilization loop.
This is the top-level component that manages the complete predictive field
memory system.

Requirements:
- 1.1, 1.2, 1.3, 1.4: Initialize and coordinate all four layers
- 6.4: Persistence (save/load)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ._memory_candidate_emergence import CandidateEmergence
from ._memory_exposure_layer import ExposureEventsLayer
from ._memory_field_core import (
    CognitiveFieldSnapshot,
    InputContext,
)
from ._memory_field_energy import FreeEnergyMinimization
from ._memory_field_formation import FieldFormation
from ._memory_operator_layer import OperatorMemoryLayer
from ._memory_recursive_stabilization import (
    RecursiveStabilization,
    StabilizationResult,
)
from ._memory_state_field import StateFieldLayer
from ._memory_transition_layer import TransitionMemoryLayer


@dataclass
class GenerationResult:
    """Result from generate() call.
    
    Contains the generated output and detailed information about
    the generation process.
    """
    output: str
    converged: bool
    iterations: int
    final_energy: float
    energy_history: List[float]


class CognitiveField:
    """
    Main orchestrator for the predictive field memory system.
    
    Manages all four memory layers:
    - Layer 1: ExposureEventsLayer (raw observational data)
    - Layer 2: StateFieldLayer (currently activated cognitive field)
    - Layer 3: OperatorMemoryLayer (learned transformation patterns)
    - Layer 4: TransitionMemoryLayer (partial-to-complete transitions)
    
    Coordinates the generation process:
    1. Field formation (simultaneous activation)
    2. Candidate emergence (tension-driven generation)
    3. Recursive stabilization (iterative field updates)
    
    Requirements:
    - 1.1, 1.2, 1.3, 1.4: Initialize all four layers
    - 6.4: Provide save/load functionality
    """
    
    def __init__(
        self,
        window_size: int = 100,
        decay: float = 0.95,
        hdc_dimensions: int = 10000,
        max_stabilization_iterations: int = 20,
        convergence_threshold: float = 0.05,
    ):
        """
        Initialize the cognitive field with all layers.
        
        Args:
            window_size: Size of sliding window for event retention
            decay: Decay factor for activation strengths
            hdc_dimensions: Dimensionality of HDC hypervectors
            max_stabilization_iterations: Maximum iterations for stabilization loop
            convergence_threshold: Energy change threshold for convergence
        """
        # Configuration
        self.window_size = window_size
        self.decay = decay
        self.hdc_dimensions = hdc_dimensions
        self.max_stabilization_iterations = max_stabilization_iterations
        self.convergence_threshold = convergence_threshold
        
        # Layer 1: Exposure Events Layer
        self.events_layer = ExposureEventsLayer(
            window_size=window_size,
            decay=decay,
        )
        
        # Layer 2: State Field Layer (created dynamically during field formation)
        self.state_field_layer = StateFieldLayer()
        
        # Layer 3: Operator Memory Layer
        self.operators_layer = OperatorMemoryLayer()
        
        # Layer 4: Transition Memory Layer
        self.transitions_layer = TransitionMemoryLayer()
        
        # Process components
        self.field_formation = FieldFormation()
        self.energy_computer = FreeEnergyMinimization()
        self.candidate_emergence = CandidateEmergence(
            energy_computer=self.energy_computer
        )
        self.recursive_stabilization = RecursiveStabilization(
            max_iterations=max_stabilization_iterations,
            convergence_threshold=convergence_threshold,
        )
        
        # Statistics
        self.total_generations = 0
        self.total_exposures = 0
    
    def form_field(self, input_context: InputContext):
        """
        Form cognitive field from input context.
        
        Activates all memory layers simultaneously and creates the
        interactive cognitive field.
        
        Args:
            input_context: Input context containing query and metadata
            
        Returns:
            StateField with activated memories
        """
        field = self.field_formation.form_field(
            input_context=input_context,
            events_layer=self.events_layer,
            operators_layer=self.operators_layer,
            previous_field=None,
        )
        
        return field
    
    def generate(
        self,
        query: str,
        modality: str = "text",
        domain: str = "conversation",
        max_iterations: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate output using recursive stabilization.
        
        Main generation pipeline:
        1. Create input context from query
        2. Run recursive stabilization loop
        3. Return final output with statistics
        
        Args:
            query: Input query text
            modality: Input modality (text, code, etc.)
            domain: Domain context (conversation, code, etc.)
            max_iterations: Override default max iterations
            
        Returns:
            GenerationResult with output and process info
        """
        # Create input context
        # For simplicity, we compute features inline here
        # In production, this would use proper feature extraction
        query_features = query.lower().split()[:20]  # Simple tokenization
        
        # Create HDC hypervector (simplified)
        query_hv = np.zeros(100, dtype=np.int8)  # Placeholder
        
        input_context = InputContext(
            query_text=query,
            query_features=query_features,
            query_hv=query_hv,
            modality=modality,
            domain=domain,
        )
        
        # Run recursive stabilization
        result = self.recursive_stabilization.stabilize(
            initial_context=input_context,
            cognitive_field=self,
            custom_max_iterations=max_iterations,
        )
        
        # Update statistics
        self.total_generations += 1
        
        # Convert to GenerationResult
        return GenerationResult(
            output=result.final_output,
            converged=result.converged,
            iterations=result.iterations,
            final_energy=result.final_energy,
            energy_history=result.energy_history,
        )
    
    def expose_text(
        self,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Expose text for learning (backward compatible API).
        
        Stores text as an event in Layer 1 and may induce operators
        or store transitions depending on context.
        
        Args:
            text: Text content to expose
            metadata: Optional metadata dictionary
            
        Returns:
            Event ID
        """
        if metadata is None:
            metadata = {}
        
        # Compute features from text
        features = self.events_layer.compute_event_features(text, "text")
        
        # Store in Layer 1 using store_event
        event_id = self.events_layer.store_event(
            modality="text",
            features=features,
            sequence=features[:20],  # First 20 features as sequence
            source=metadata.get("source", ""),
            label=metadata.get("label"),
            preview=text[:100],  # First 100 chars as preview
        )
        
        # Update statistics
        self.total_exposures += 1
        
        return event_id
    
    def expose_file(
        self,
        filepath: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Expose file for learning (backward compatible API).
        
        Args:
            filepath: Path to file
            metadata: Optional metadata dictionary
            
        Returns:
            Event ID
        """
        if metadata is None:
            metadata = {}
        
        # Read file content
        import pathlib
        path = pathlib.Path(filepath)
        
        if path.suffix in ['.txt', '.md']:
            content = path.read_text(encoding='utf-8', errors='ignore')
            modality = "text"
            features = self.events_layer.compute_event_features(content, modality)
        else:
            content = str(path.read_bytes())
            modality = "file"
            features = [f"file:{path.name}", f"ext:{path.suffix}"]
        
        # Store in Layer 1
        event_id = self.events_layer.store_event(
            modality=modality,
            features=features,
            sequence=features[:20],
            source=filepath,
            label=metadata.get("label"),
            preview=content[:100] if isinstance(content, str) else f"File: {path.name}",
        )
        
        # Update statistics
        self.total_exposures += 1
        
        return event_id
    
    def rank(
        self,
        query: str,
        candidates: List[str],
        mode: str = "event",
    ) -> List[float]:
        """
        Rank candidates using field-based scoring (backward compatible API).
        
        Uses field energy reduction as the ranking criterion.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            mode: Ranking mode (currently ignored, uses field-based scoring)
            
        Returns:
            List of scores (higher is better)
        """
        # Form field for query
        query_features = query.lower().split()[:20]
        query_hv = np.zeros(100, dtype=np.int8)
        
        query_context = InputContext(
            query_text=query,
            query_features=query_features,
            query_hv=query_hv,
            modality="text",
            domain="conversation",
        )
        
        query_field = self.form_field(query_context)
        
        # Compute query field energy
        query_energy = self.energy_computer.compute_field_energy(query_field)
        query_field.field_energy = query_energy
        
        # Score each candidate by predicted energy reduction
        scores = []
        for candidate in candidates:
            # Predict energy after incorporating candidate
            predicted_energy = self.energy_computer.predict_energy_after_update(
                query_field,
                candidate,
            )
            
            # Energy reduction (higher is better)
            energy_reduction = query_energy.total - predicted_energy
            scores.append(energy_reduction)
        
        return scores
    
    def save(self, path: str) -> None:
        """
        Save cognitive field state to disk.
        
        Serializes all four layers and configuration to a snapshot file.
        
        Args:
            path: Path to save snapshot
        """
        import pickle
        from datetime import datetime
        
        snapshot = CognitiveFieldSnapshot(
            version="1.0.0",
            timestamp=datetime.now().timestamp(),
            
            # Layer 1 data
            events=self.events_layer.events,
            edges=self.events_layer.edges,
            feature_to_id=self.events_layer.feature_to_id,
            id_to_feature=self.events_layer.id_to_feature,
            event_hv=self.events_layer.event_hv,
            
            # Layer 3 data
            operators=self.operators_layer.operators,
            
            # Layer 4 data
            transitions=self.transitions_layer.transitions,
            
            # Statistics
            total_exposures=self.total_exposures,
            total_operators_induced=len(self.operators_layer.operators),
            total_transitions_stored=len(self.transitions_layer.transitions),
            
            # Configuration
            window_size=self.window_size,
            decay=self.decay,
            hdc_dimensions=self.hdc_dimensions,
        )
        
        with open(path, 'wb') as f:
            pickle.dump(snapshot, f)
    
    def load(self, path: str) -> None:
        """
        Load cognitive field state from disk.
        
        Restores all four layers and configuration from a snapshot file.
        
        Args:
            path: Path to snapshot file
        """
        import pickle
        
        with open(path, 'rb') as f:
            snapshot: CognitiveFieldSnapshot = pickle.load(f)
        
        # Restore Layer 1
        self.events_layer.events = snapshot.events
        self.events_layer.edges = snapshot.edges
        self.events_layer.feature_to_id = snapshot.feature_to_id
        self.events_layer.id_to_feature = snapshot.id_to_feature
        self.events_layer.event_hv = snapshot.event_hv
        
        # Restore Layer 3
        self.operators_layer.operators = snapshot.operators
        
        # Restore Layer 4
        self.transitions_layer.transitions = snapshot.transitions
        
        # Restore configuration
        self.window_size = snapshot.window_size
        self.decay = snapshot.decay
        self.hdc_dimensions = snapshot.hdc_dimensions
        
        # Restore statistics
        self.total_exposures = snapshot.total_exposures


__all__ = ["CognitiveField", "GenerationResult"]
