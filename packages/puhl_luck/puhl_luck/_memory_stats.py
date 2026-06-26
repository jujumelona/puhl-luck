from __future__ import annotations

from ._brain_common import *


class MemoryStatsMixin:
    def prune(self, min_edge: float = 0.02, max_edges_per_feature: int = 64) -> int:
        removed = 0
        per_feature: Dict[int, List[Tuple[Tuple[int, int], float]]] = defaultdict(list)
        for key, weight in self.edges.items():
            if weight >= min_edge:
                left, _ = key
                per_feature[left].append((key, weight))
        keep_keys = set()
        for rows in per_feature.values():
            rows.sort(key=lambda item: item[1], reverse=True)
            keep_keys.update(key for key, _ in rows[:max_edges_per_feature])
        old_count = len(self.edges)
        self.edges = {key: weight for key, weight in self.edges.items() if key in keep_keys and weight >= min_edge}
        self.edge_last_seen = {key: seen for key, seen in self.edge_last_seen.items() if key in self.edges}
        removed += old_count - len(self.edges)
        self._neighbors_dirty = True
        return removed

    def stats(self) -> Dict[str, Any]:
        edge_count = len(self.edges)
        return {
            "events": len(self.events),
            "features": len(self.feature_freq),
            "edges": edge_count,
            "concepts": len(self.concept_members),
            "short_term_events": len(self.short_term_events),
            "modalities": dict(self.modality_freq),
            "labels": dict(self.label_freq),
            "total_exposures": self.total_exposures,
            "avg_novelty": sum(self.event_novelty.values()) / max(1, len(self.event_novelty)),
            "hdc_bits": self.hdc_bits,
            "hdc_words": self.hdc_words,
            "hdc_indexed_bands": self.hdc_indexed_bands,
            "activation_hops": self.dynamic_hops(),
            "recall_top_k": self.dynamic_recall_top_k(),
            "rank_event_cap": self.dynamic_rank_event_cap(),
            "memory_bytes_estimate": self.memory_bytes_estimate(),
        }

    def memory_bytes_estimate(self) -> int:
        edge_count = len(self.edges)
        hdc_bytes = sum(vec.nbytes for vec in self.event_hv.values() if isinstance(vec, np.ndarray))
        return int(len(self.events) * 384 + len(self.feature_freq) * 80 + edge_count * 16 + hdc_bytes)

