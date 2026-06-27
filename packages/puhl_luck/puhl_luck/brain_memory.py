from __future__ import annotations

from ._brain_common import *
from ._micro_rank import MicroRankModel
from ._memory_generation import MemoryGenerationMixin
from ._memory_learning import MemoryLearningMixin
from ._memory_persistence import MemoryPersistenceMixin
from ._memory_ranking import MemoryRankingMixin
from ._memory_transition_mixin import MemoryTransitionMixin


class BrainMemory(
    MemoryLearningMixin,
    MemoryRankingMixin,
    MemoryGenerationMixin,
    MemoryPersistenceMixin,
    MemoryTransitionMixin,
):
    """Unified non-gradient event memory with transition learning and field-based generation."""
    def __init__(self, window_size: int = 12, decay: float = 0.72):
        self.window_size = int(window_size)
        self.decay = float(decay)
        self.events: Dict[str, EventRecord] = {}
        self.feature_freq: Counter[str] = Counter()
        self.total_feature_count = 0
        self.modality_freq: Counter[str] = Counter()
        self.label_freq: Counter[str] = Counter()
        self.feature_to_id: Dict[str, int] = {}
        self.id_to_feature: List[str] = []
        self.edges: Dict[Tuple[int, int], float] = {}
        self.edge_last_seen: Dict[Tuple[int, int], int] = {}
        self._neighbors: Dict[int, List[Tuple[int, float]]] = {}
        self._neighbors_dirty = True
        self.feature_to_events: Dict[int, Counter[str]] = defaultdict(Counter)
        self.feature_top_events: Dict[int, Counter[str]] = defaultdict(Counter)
        self.token_unigrams: Counter[str] = Counter()
        self.sequence_starts: Counter[str] = Counter()
        self.token_successors: Dict[str, Counter[str]] = defaultdict(Counter)
        self.order_contexts: Dict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.event_novelty: Dict[str, float] = {}
        self.event_content_sets: Dict[str, set[str]] = {}
        self.event_hv: Dict[str, np.ndarray] = {}
        self.hdc_words = dynamic_hdc_words(0, 0)
        self.hdc_bits = dynamic_hdc_bits(0, 0)
        self.hdc_indexed_bands = hdc_band_count(self.hdc_words, 0)
        self.hdc_index: Dict[Tuple[int, int], set[str]] = defaultdict(set)
        self.cluster_freq: Counter[Tuple[int, ...]] = Counter()
        self.concept_members: Dict[str, List[int]] = {}
        self.short_term_events: List[str] = []
        self.total_exposures = 0
        self._updates_since_prune = 0
        self._rank_feature_cache: Dict[str, List[str]] = {}
        self._rank_content_cache: Dict[str, List[str]] = {}
        self._rank_state_cache: Dict[str, Dict[str, Any]] = {}
        self._rank_choice_cache: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
        self._rank_result_cache: Dict[Tuple[int, str, Tuple[str, ...]], Tuple[int, List[float]]] = {}
        self._feature_weight_cache: Dict[str, float] = {}
        self.rank_mode = "event"

        # Initialise transition / operator / surface layers (Layer 3, 4, 8)
        self._init_transition_layers()



__all__ = [
    "BrainMemory",
    "EventRecord",
    "MicroRankModel",
    "tokenize",
    "text_feature_list",
    "bundle_hv",
    "hv_similarity",
    "write_varuint",
]
