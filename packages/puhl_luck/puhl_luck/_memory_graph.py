from __future__ import annotations

from ._brain_common import *


class MemoryGraphMixin:
    def dynamic_generation_order(self) -> int:
        return max(2, min(5, self.dynamic_sequence_order()))

    def clear_rank_caches(self) -> None:
        self._rank_feature_cache.clear()
        self._rank_content_cache.clear()
        self._rank_state_cache.clear()
        self._rank_choice_cache.clear()
        self._rank_result_cache.clear()
        self._feature_weight_cache.clear()

    def refresh_dynamic_hdc_if_needed(self, extra_features: int = 0, extra_events: int = 0) -> None:
        target_words = dynamic_hdc_words(len(self.feature_to_id) + extra_features, len(self.events) + extra_events)
        target_bands = hdc_band_count(target_words, len(self.events) + extra_events)
        if target_words <= self.hdc_words and target_bands <= self.hdc_indexed_bands:
            return
        old_bands = self.hdc_indexed_bands
        old_words = self.hdc_words
        self.hdc_words = target_words
        self.hdc_bits = target_words * HDC_WORD_BITS
        self.hdc_indexed_bands = target_bands
        for eid, rec in self.events.items():
            full_vec = self.bundle_event(rec.features, rec.sequence)
            old_vec = self.event_hv.get(eid)
            if old_vec is not None and old_vec.size == old_words:
                full_vec[:old_words] = old_vec
            rec.hv = full_vec
            self.event_hv[eid] = full_vec
            self.index_event_hv(eid, full_vec, start_word=old_bands)

    def bundle_event(self, features: List[str], sequence: List[str]) -> np.ndarray:
        return bundle_hv(features, self.hdc_bits)

    def index_event_hv(self, event_id: str, event_vec: np.ndarray, start_word: int = 0) -> None:
        for band in hdc_bands(event_vec, max(1, len(self.events)), start_word=start_word):
            self.hdc_index[band].add(event_id)

    def observe_concepts(self, ids: List[int]) -> Tuple[List[str], List[int]]:
        content_ids = [
            fid for fid in dict.fromkeys(ids)
            if fid < len(self.id_to_feature)
            and not self.id_to_feature[fid].startswith(("mod:", "label:", "concept:"))
        ]
        if len(content_ids) < 3:
            return [], []
        width = max(3, int(math.sqrt(len(content_ids))) + 1)
        cluster = tuple(sorted(content_ids[:width]))
        self.cluster_freq[cluster] += 1
        threshold = self.dynamic_concept_threshold()
        if self.cluster_freq[cluster] < threshold:
            return [], []
        concept = f"concept:{stable_id(','.join(map(str, cluster)), 8)}"
        concept_id = self.feature_id(concept)
        self.concept_members[concept] = list(cluster)
        gain = math.log1p(self.cluster_freq[cluster])
        for member in cluster:
            self.add_edge(member, concept_id, gain)
            self.add_edge(concept_id, member, gain)
        return [concept], [concept_id]

    def dynamic_concept_threshold(self) -> int:
        return max(2, int(math.sqrt(max(1, self.total_exposures + 1))))

    def remember_short_term(self, event_id: str) -> None:
        self.short_term_events.append(event_id)
        limit = self.dynamic_short_term_limit()
        if len(self.short_term_events) > limit:
            del self.short_term_events[: len(self.short_term_events) - limit]

    def dynamic_short_term_limit(self) -> int:
        return max(4, int(math.sqrt(max(1, len(self.events)))) + self.dynamic_sequence_order())

    def add_edge(self, left: int, right: int, weight: float) -> None:
        if left == right:
            return
        key = (left, right)
        last_seen = self.edge_last_seen.get(key, self.total_exposures)
        age = max(0, self.total_exposures - last_seen)
        aged = self.edges.get(key, 0.0) * (self.decay ** min(age, 64))
        self.edges[key] = aged + weight
        self.edge_last_seen[key] = self.total_exposures
        self._neighbors_dirty = True

    def neighbors(self) -> Dict[int, List[Tuple[int, float]]]:
        if not self._neighbors_dirty:
            return self._neighbors
        graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (left, right), weight in self.edges.items():
            graph[left].append((right, weight))
        self._neighbors = dict(graph)
        self._neighbors_dirty = False
        return self._neighbors

    def dynamic_sequence_order(self) -> int:
        if self.total_exposures < 16:
            return 2
        entropy_proxy = math.log2(max(2, len(self.feature_freq)))
        return max(2, min(12, 2 + int(entropy_proxy // 2)))

    def novelty_score(self, features: List[str]) -> float:
        if not features or not self.feature_freq:
            return 1.0
        total = max(1, sum(self.feature_freq.values()))
        vocab = max(1, len(self.feature_freq))
        surprisal = 0.0
        for feature in features:
            prob = (self.feature_freq.get(feature, 0) + 1.0) / (total + vocab)
            surprisal += -math.log2(prob)
        return max(0.1, min(2.0, (surprisal / max(1, len(features))) / 8.0))

    def surprisal_gain(self, features: List[str]) -> float:
        return 0.5 + self.novelty_score(features)

