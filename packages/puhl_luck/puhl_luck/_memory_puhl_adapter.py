"""
PUHLCompatibilityAdapter: Backward compatibility layer for original PUHL API.

This adapter translates old PUHL API calls to the new field-based architecture,
allowing existing code to work without modification.
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from pathlib import Path

from ._memory_cognitive_field import CognitiveField
from ._memory_field_core import InputContext
from .brain_memory import BrainMemory


class PUHLCompatibilityAdapter:
    """
    Adapter that provides backward compatibility with original PUHL API.
    
    Translates old API calls (expose_text, expose_file, rank) to new
    field-based operations while maintaining the same interface.
    """
    
    def __init__(
        self,
        memory_file: str = "brain_memory.pkl",
        window_size: int = 50,
        decay: float = 0.01
    ):
        """Initialize compatibility adapter.
        
        Args:
            memory_file: Path to memory persistence file
            window_size: Sliding window size for events
            decay: Decay rate for activation
        """
        self.memory_file = memory_file
        self.window_size = window_size
        self.decay = decay
        
        # Initialize cognitive field (new architecture)
        self.cognitive_field = CognitiveField(
            window_size=window_size,
            decay=decay
        )
        
        # Also maintain old BrainMemory for compatibility
        self.brain_memory = BrainMemory(
            window_size=window_size,
            decay=decay
        )
        
        # Load existing memory if available
        self._load_if_exists()
    
    def _load_if_exists(self):
        """Load existing memory file if it exists."""
        cf_path = self.memory_file.replace('.pkl', '_cognitive_field.pkl')
        if Path(cf_path).exists():
            try:
                # Load cognitive field state into existing instance
                self.cognitive_field.load(cf_path)
                # Migrate events from cognitive field to brain_memory
                self._sync_from_cognitive_field()
            except Exception as e:
                print(f"Warning: Could not load existing memory: {e}")
    
    def _migrate_events(self):
        """Migrate events from old BrainMemory to new CognitiveField."""
        # Get all events from old memory
        if hasattr(self.brain_memory, 'events') and self.brain_memory.events:
            for event_id, event in self.brain_memory.events.items():
                # Extract data from EventRecord
                text = getattr(event, 'text', '')
                features = getattr(event, 'features', [])
                feature_hv = self.brain_memory.event_hv.get(event_id, np.array([], dtype=np.uint64))
                
                # Store event in new Layer 1
                self.cognitive_field.events_layer.store_event(
                    event_id=event_id,
                    text=text,
                    features=features,
                    feature_hv=feature_hv,
                    metadata={}
                )
    
    def _sync_from_cognitive_field(self):
        """Sync events from CognitiveField to BrainMemory after load."""
        from ._brain_defs import EventRecord
        import time
        
        # Migrate all events from cognitive_field to brain_memory
        for event_id, event_record in self.cognitive_field.events_layer.events.items():
            # event_record is an EventRecord object, not a dict
            now = int(time.time())
            
            # Get stored HDC vector
            fhv = self.cognitive_field.events_layer.event_hv.get(event_id, np.array([], dtype=np.uint64))
            
            # Create EventRecord in brain_memory (EventRecord is already an EventRecord)
            # Just copy it directly
            self.brain_memory.events[event_id] = event_record
            self.brain_memory.event_hv[event_id] = fhv
    
    def expose_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        mode: str = "general"
    ) -> str:
        """
        Expose text to memory (backward compatible API).
        
        Args:
            text: Text to expose
            metadata: Optional metadata
            mode: Exposure mode (general, code, document)
        
        Returns:
            Event ID
        """
        from ._brain_text_features import tokenize, text_feature_list
        from ._brain_hdc import bundle_hv
        import time
        
        # Extract features
        tokens = tokenize(text)
        features = text_feature_list(tokens)
        
        # Store in new cognitive field first (this generates the event_id)
        event_id = self.cognitive_field.events_layer.store_event(
            modality=mode,
            features=features,
            sequence=tokens,
            source="adapter",
            label=None,
            preview=text
        )
        
        # Also maintain in brain_memory for compatibility
        # Use same event_id that cognitive_field generated
        from ._brain_defs import EventRecord
        now = int(time.time())
        
        # Get the HDC vector that was created
        fhv = self.cognitive_field.events_layer.event_hv.get(event_id, np.array([], dtype=np.uint64))
        
        self.brain_memory.events[event_id] = EventRecord(
            event_id=event_id,
            modality=mode,
            source="adapter",
            label=None,
            features=features,
            sequence=tokens,
            preview=text[:240],
            novelty=1.0,
            hv=fhv,
            created_at=now,
            last_accessed_at=now
        )
        self.brain_memory.event_hv[event_id] = fhv
        
        return event_id
    
    def expose_file(
        self,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
        mode: str = "general"
    ) -> List[str]:
        """
        Expose file contents to memory (backward compatible API).
        
        Args:
            filepath: Path to file
            metadata: Optional metadata
            mode: Exposure mode
        
        Returns:
            List of event IDs
        """
        event_ids = []
        
        # Read file and expose each line
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    event_id = self.expose_text(line, metadata, mode)
                    event_ids.append(event_id)
        
        return event_ids
    
    def rank(
        self,
        query: str,
        candidates: List[str],
        mode: str = "similarity",
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Rank candidates using field-based scoring (backward compatible API).
        
        Args:
            query: Query text
            candidates: List of candidate texts
            mode: Ranking mode (similarity, energy, hybrid)
            top_k: Number of top results to return
        
        Returns:
            List of (candidate, score) tuples
        """
        if mode == "similarity":
            # Simple HDC similarity-based ranking
            from ._brain_hdc import bundle_hv, hv_similarity
            from ._brain_text_features import tokenize, text_feature_list
            
            # Get query features
            query_tokens = tokenize(query)
            query_features = text_feature_list(query_tokens)
            if query_features:
                query_hv = bundle_hv(query_features)
            else:
                query_hv = np.array([], dtype=np.uint64)
            
            # Score each candidate
            scored = []
            for candidate in candidates:
                cand_tokens = tokenize(candidate)
                cand_features = text_feature_list(cand_tokens)
                if cand_features:
                    cand_hv = bundle_hv(cand_features)
                else:
                    cand_hv = np.array([], dtype=np.uint64)
                
                similarity = hv_similarity(query_hv, cand_hv)
                scored.append((candidate, similarity))
            
            # Sort and return top_k
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        
        elif mode in ["energy", "hybrid"]:
            # Use new field-based ranking
            return self._rank_with_field(query, candidates, mode, top_k)
        
        else:
            # Default to similarity
            return self.rank(query, candidates, "similarity", top_k)
    
    def _rank_with_field(
        self,
        query: str,
        candidates: List[str],
        mode: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Rank candidates using field energy."""
        # Create input context
        context = InputContext.from_text(query)
        
        # Form initial field
        field = self.cognitive_field.form_field(context)
        
        # Score each candidate by predicted energy reduction
        scored_candidates = []
        
        for candidate in candidates:
            # Create context with candidate as partial output
            candidate_context = InputContext.from_text(
                query,
                partial_output=candidate
            )
            
            # Form field with candidate
            candidate_field = self.cognitive_field.form_field(candidate_context)
            
            # Compute energy
            if candidate_field.field_energy:
                energy = candidate_field.field_energy.total_energy
                # Lower energy = better (negate for ranking)
                score = -energy
            else:
                score = 0.0
            
            scored_candidates.append((candidate, score))
        
        # Sort by score (higher is better)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return scored_candidates[:top_k]
    
    def save(self, filepath: Optional[str] = None):
        """Save memory state.
        
        Args:
            filepath: Optional path to save to (defaults to memory_file)
        """
        # Save new cognitive field
        cf_filepath = filepath or self.memory_file
        cf_filepath = cf_filepath.replace('.pkl', '_cognitive_field.pkl')
        self.cognitive_field.save(cf_filepath)
    
    def load(self, filepath: Optional[str] = None):
        """Load memory state.
        
        Args:
            filepath: Optional path to load from (defaults to memory_file)
        """
        # Try to load new cognitive field
        cf_filepath = filepath or self.memory_file
        cf_filepath = cf_filepath.replace('.pkl', '_cognitive_field.pkl')
        
        try:
            # Load into existing cognitive_field instance
            self.cognitive_field.load(cf_filepath)
            # Sync events from cognitive_field to brain_memory
            self._sync_from_cognitive_field()
        except FileNotFoundError:
            # No saved state yet
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary with statistics about memory state
        """
        # Get stats from new field-based system
        field_stats = {
            'cognitive_field': {
                'layers': 4,
                'events_count': len(self.cognitive_field.events_layer.events),
                'operators_count': len(self.cognitive_field.operators_layer.operators),
                'transitions_count': len(self.cognitive_field.transitions_layer.transitions)
            },
            'brain_memory': {
                'events_count': len(self.brain_memory.events),
                'window_size': self.window_size,
                'decay': self.decay
            }
        }
        
        return field_stats
    
    # Delegate other methods to brain_memory for compatibility
    def __getattr__(self, name):
        """Delegate unknown methods to brain_memory."""
        return getattr(self.brain_memory, name)

