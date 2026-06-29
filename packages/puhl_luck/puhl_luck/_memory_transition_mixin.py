
from __future__ import annotations

from typing import Any, Optional, Tuple


def _get_transition_layer():
    from ._memory_transition_layer import TransitionMemoryLayer
    return TransitionMemoryLayer


class MemoryTransitionMixin:
    """Single generation path: weightless sparse count next-token model."""

    def _init_transition_layers(self) -> None:
        self._transition_layer = _get_transition_layer()()

        # P66: the live path is no longer a disabled 9-layer diagram.  These
        # layers are instantiated and attached to the sparse generator.  The
        # generator performs the actual multi-hop loop so generation has one
        # source of truth and does not fall back to old answer retrieval.
        try:
            from ._memory_operator_layer import OperatorMemoryLayer
            self._operator_layer = OperatorMemoryLayer()
        except Exception:
            self._operator_layer = None
        try:
            from ._memory_operator_induction import OperatorInduction
            self._operator_induction = OperatorInduction()
        except Exception:
            self._operator_induction = None
        try:
            from ._memory_surface_realization import SurfaceRealizationLayer
            self._surface_layer = SurfaceRealizationLayer()
        except Exception:
            self._surface_layer = None
        try:
            from ._memory_surface_storage import SurfaceStorageLayer
            self._surface_storage = SurfaceStorageLayer()
        except Exception:
            self._surface_storage = None
        try:
            from ._memory_operator_storage import OperatorStorage
            self._operator_storage = OperatorStorage()
        except Exception:
            self._operator_storage = None
        try:
            from ._memory_field_energy import FreeEnergyMinimization
            self._energy_computer = FreeEnergyMinimization()
        except Exception:
            self._energy_computer = None
        try:
            from ._memory_candidate_emergence import CandidateEmergence
            self._candidate_emergence = CandidateEmergence(energy_computer=self._energy_computer)
        except Exception:
            self._candidate_emergence = None
        try:
            from ._memory_recursive_stabilization import RecursiveStabilization
            self._recursive_stabilization = RecursiveStabilization(max_iterations=0)
        except Exception:
            self._recursive_stabilization = None
        self._pending_induction_count = 0

        try:
            from ._logit_generator import SparseLogitGenerator
            self._logit_generator = SparseLogitGenerator(
                output_mode='text',
                use_rust=True,
                adaptive_readout=True,
                readout_auto_resize=True,
                fast_runtime=True,
                parallel_inference=True,
                batch_learning=True,
            )
        except Exception as e:
            print(f"ERROR: Failed to create logit generator: {e}")
            import traceback
            traceback.print_exc()
            self._logit_generator = None
        if hasattr(self._logit_generator, 'attach_live_layers'):
            self._logit_generator.attach_live_layers(
                transition_layer=self._transition_layer,
                operator_layer=self._operator_layer,
                operator_induction=self._operator_induction,
                surface_layer=self._surface_layer,
                surface_storage=self._surface_storage,
                operator_storage=self._operator_storage,
                energy_computer=self._energy_computer,
                candidate_emergence=self._candidate_emergence,
                recursive_stabilization=self._recursive_stabilization,
            )

        try:
            from ._memory_generation_metrics import MetricsTracker
            self._metrics_tracker = MetricsTracker()
        except Exception:
            self._metrics_tracker = None

    def generate(
        self,
        query: str,
        max_new_tokens: int = 64,
        domain: str = 'text',
        return_metrics: bool = False,
    ):
        result, metrics = self._generate_from_logits(str(query), int(max_new_tokens), domain)
        if return_metrics:
            return result, metrics
        return result, metrics  # Always return tuple for consistency

    def generate_batch(
        self,
        queries,
        max_new_tokens: int = 64,
        domain: str = 'text',
        return_metrics: bool = False,
        max_workers: Optional[int] = None,
    ):
        lg = getattr(self, '_logit_generator', None)
        if lg is None or not hasattr(lg, 'generate_many'):
            out = [self.generate(q, max_new_tokens=max_new_tokens, domain=domain, return_metrics=return_metrics) for q in queries]
            return out
        results = lg.generate_many([str(q) for q in queries], max_tokens=max_new_tokens, max_workers=max_workers)
        if return_metrics:
            return results
        return [r[0] for r in results]

    def get_statistics(self):
        """Get learning and generation statistics from the logit generator.
        
        Returns a dictionary containing:
        - pairs_learned: Number of training pairs learned
        - total_transitions: Total state transitions stored
        - total_contexts: Number of unique context sketches
        - total_unique_tokens: Number of unique tokens in vocabulary
        - And other internal statistics
        """
        lg = getattr(self, '_logit_generator', None)
        if lg is None or not hasattr(lg, 'get_statistics'):
            return {
                'pairs_learned': 0,
                'total_transitions': 0,
                'total_contexts': 0,
                'total_unique_tokens': 0,
            }
        return lg.get_statistics()

    def _generate_from_logits(self, query: str, max_tokens: int, domain: str) -> Tuple[str, Optional[Any]]:
        try:
            from ._memory_generation_metrics import GenerationMetrics
            query_features = self.features_for_query(query)
            result, logit_metrics = self._logit_generator.generate(
                input_text=query,
                field_features=query_features[:128] if query_features else None,
                max_tokens=max_tokens,
                return_metrics=True,  # Always request metrics for detailed tracking
            )
            
            # logit_metrics is already a dict, potentially with detailed_metrics
            # We keep the dict format for backward compatibility
            return result or '', logit_metrics
        except Exception:
            # No old fallback. Failure must be visible as empty output.
            return '', None


__all__ = ['MemoryTransitionMixin']
