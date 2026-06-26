from __future__ import annotations

from ._brain_common import *


class MemorySaveMixin:
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = self.__getstate__()
        with lzma.open(p, "wb", preset=0) as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_rank_feature_cache"] = {}
        state["_rank_content_cache"] = {}
        state["_rank_state_cache"] = {}
        state["_rank_choice_cache"] = {}
        state["_rank_result_cache"] = {}
        state["_feature_weight_cache"] = {}
        state["hdc_index"] = defaultdict(set)
        state["event_hv"] = {}
        state["event_content_sets"] = {}
        state["feature_top_events"] = defaultdict(Counter)
        return state

    def save_uncompressed(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BrainMemory":
        p = Path(path)
        with p.open("rb") as raw:
            magic = raw.read(6)
        if magic.startswith(b"\xfd7zXZ"):
            opener = lzma.open
        elif magic.startswith(b"\x1f\x8b"):
            opener = gzip.open
        else:
            opener = open
        with opener(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            mem = cls(window_size=obj.get("window_size", 12), decay=obj.get("decay", 0.72))
            mem.__dict__.update(obj)
            obj = mem
        elif not isinstance(obj, cls):
            raise TypeError("brain memory file has incompatible type")
        obj.ensure_runtime_fields()
        return obj

    def ensure_runtime_fields(self) -> None:
        if not hasattr(self, "hdc_words"):
            self.hdc_words = dynamic_hdc_words(len(getattr(self, "feature_to_id", {})), len(getattr(self, "events", {})))
        if not hasattr(self, "hdc_bits"):
            self.hdc_bits = self.hdc_words * HDC_WORD_BITS
        if not hasattr(self, "hdc_indexed_bands"):
            self.hdc_indexed_bands = hdc_band_count(self.hdc_words, len(getattr(self, "events", {})))
        if not hasattr(self, "hdc_index"):
            self.hdc_index = defaultdict(set)
        if not hasattr(self, "event_hv"):
            self.event_hv = {}
        if not hasattr(self, "feature_top_events"):
            self.feature_top_events = defaultdict(Counter)
        if not hasattr(self, "total_feature_count"):
            self.total_feature_count = sum(getattr(self, "feature_freq", Counter()).values())
        if not hasattr(self, "event_content_sets"):
            self.event_content_sets = {}
        if not hasattr(self, "_rank_feature_cache"):
            self._rank_feature_cache = {}
        if not hasattr(self, "_rank_content_cache"):
            self._rank_content_cache = {}
        if not hasattr(self, "_rank_state_cache"):
            self._rank_state_cache = {}
        if not hasattr(self, "_rank_choice_cache"):
            self._rank_choice_cache = {}
        if not hasattr(self, "_rank_result_cache"):
            self._rank_result_cache = {}
        if not hasattr(self, "_feature_weight_cache"):
            self._feature_weight_cache = {}
        if not hasattr(self, "token_successors"):
            self.token_successors = defaultdict(Counter)
        if not hasattr(self, "order_contexts"):
            self.order_contexts = defaultdict(Counter)
        if not hasattr(self, "token_unigrams"):
            self.token_unigrams = Counter()
        if not hasattr(self, "sequence_starts"):
            self.sequence_starts = Counter()
        if not self.order_contexts:
            for rec in getattr(self, "events", {}).values():
                self.learn_order_trace(rec.sequence)
        if not self.token_unigrams or not self.sequence_starts:
            self.token_unigrams = Counter()
            self.sequence_starts = Counter()
            for rec in getattr(self, "events", {}).values():
                tokens = [symbol.split(":", 1)[1] for symbol in rec.sequence if symbol.startswith("text:")]
                if tokens:
                    self.sequence_starts[tokens[0]] += 1
                    self.token_unigrams.update(tokens)
        if not hasattr(self, "rank_mode"):
            self.rank_mode = "free_energy"
        if not hasattr(self, "cluster_freq"):
            self.cluster_freq = Counter()
        if not hasattr(self, "concept_members"):
            self.concept_members = {}
        if not hasattr(self, "short_term_events"):
            self.short_term_events = list(getattr(self, "events", {}).keys())[-self.dynamic_short_term_limit():]
        if not hasattr(self, "_neighbors_dirty"):
            self._neighbors_dirty = True
        for eid, rec in self.events.items():
            if not isinstance(getattr(rec, "hv", None), np.ndarray) or rec.hv.size != self.hdc_words:
                rec.hv = self.bundle_event(rec.features, rec.sequence)
            if not hasattr(rec, "created_at"):
                rec.created_at = self.total_exposures
            if not hasattr(rec, "last_accessed_at"):
                rec.last_accessed_at = rec.created_at
            self.event_hv[eid] = rec.hv
            self.event_content_sets[eid] = set(content_features(rec.features))
        self.hdc_index = defaultdict(set)
        self.hdc_indexed_bands = hdc_band_count(self.hdc_words, len(self.events))
        for eid, vec in self.event_hv.items():
            self.index_event_hv(eid, vec)
        if not self.feature_top_events:
            cap = self.dynamic_rank_event_cap()
            self.feature_top_events = defaultdict(Counter)
            for fid, rows in self.feature_to_events.items():
                self.feature_top_events[fid] = Counter(dict(Counter(rows).most_common(cap)))

