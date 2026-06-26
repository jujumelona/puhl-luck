from __future__ import annotations

from ._brain_common import *


class MemoryActivationCoreMixin:
    def activation(self, seed_features: Iterable[str], hops: Optional[int] = None, decay: Optional[float] = None) -> Dict[str, float]:
        energy_ids: Dict[int, float] = {}
        hop_count = hops if hops is not None else self.dynamic_hops()
        decay_value = decay if decay is not None else self.dynamic_activation_decay()
        for feature in seed_features:
            fid = self.feature_to_id.get(feature)
            if fid is not None:
                energy_ids[fid] = max(energy_ids.get(fid, 0.0), self.feature_weight(feature))
        for _ in range(max(0, hop_count)):
            next_energy: Dict[int, float] = {}
            graph = self.neighbors()
            for fid, energy in energy_ids.items():
                for neighbor, weight in graph.get(fid, ()):
                    neighbor_feature = self.id_to_feature[neighbor] if neighbor < len(self.id_to_feature) else ""
                    next_energy[neighbor] = next_energy.get(neighbor, 0.0) + energy * weight * decay_value * self.feature_weight(neighbor_feature)
            if next_energy:
                mean_energy = sum(next_energy.values()) / len(next_energy)
                for fid in list(next_energy.keys()):
                    if next_energy[fid] < mean_energy:
                        next_energy[fid] *= 0.5
            for fid, energy in next_energy.items():
                energy_ids[fid] = energy_ids.get(fid, 0.0) + energy
        return {self.id_to_feature[fid]: val for fid, val in energy_ids.items() if fid < len(self.id_to_feature)}

    def dynamic_hops(self) -> int:
        return max(1, min(4, 1 + int(math.log2(max(1, len(self.events))) // 4)))

    def dynamic_activation_decay(self) -> float:
        density = len(self.edges) / max(1, len(self.feature_to_id))
        return max(0.25, min(0.75, 0.65 - 0.02 * math.log1p(density)))

    def dynamic_recall_top_k(self) -> int:
        return max(4, min(64, int(math.sqrt(max(1, len(self.events)))) + 4))

    def dynamic_rank_event_cap(self) -> int:
        return max(4, min(16, int(math.log2(max(2, len(self.events)))) * 2))

    def feature_weight(self, feature: str) -> float:
        if not feature or not self.feature_freq:
            return 1.0
        cached = self._feature_weight_cache.get(feature)
        if cached is not None:
            return cached
        total = max(1, int(getattr(self, "total_feature_count", 0)) or sum(self.feature_freq.values()))
        vocab = max(1, len(self.feature_freq))
        freq = self.feature_freq.get(feature, 0)
        canonical = canonical_feature(feature)
        rarity = 1.0 + math.log1p((total + vocab) / (freq + 1.0)) / math.log2(total + vocab + 2.0)
        if canonical.startswith("id:"):
            value = rarity * 6.0
        elif canonical.startswith(("bi:", "tri:")):
            value = rarity * 2.0
        elif canonical.startswith("tok:"):
            value = rarity * 1.5
        elif canonical.startswith("c3:"):
            value = rarity * 0.25
        elif ":ascii:" in canonical or ":phash:" in canonical or ":zcr" in canonical:
            value = rarity * 2.5
        else:
            value = rarity
        if len(self._feature_weight_cache) > 8192:
            self._feature_weight_cache.clear()
        self._feature_weight_cache[feature] = value
        return value

    def retrieval_feature_weight(self, feature: str, anchor_present: bool = False) -> float:
        canonical = canonical_feature(feature)
        if canonical.startswith("c3:"):
            return 0.02 if anchor_present else 0.15
        if canonical.startswith("id:"):
            return self.feature_weight(feature) * 4.0
        return self.feature_weight(feature)

    def anchor_features(self, features: Iterable[str]) -> List[str]:
        return [feature for feature in dict.fromkeys(features) if canonical_feature(feature).startswith("id:")]

    def expanded_query_features(self, query_features: List[str], limit: Optional[int] = None) -> List[str]:
        base = list(dict.fromkeys(query_features))
        if not base:
            return []
        expansion_limit = limit if limit is not None else max(len(base), int(math.sqrt(max(1, len(self.feature_to_id)))) + self.dynamic_sequence_order())
        energy = self.activation(base, hops=1)
        base_set = set(base)
        ranked = sorted(
            (
                (feature, value * self.feature_weight(feature))
                for feature, value in energy.items()
                if feature not in base_set and not feature.startswith("mod:")
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        expanded = base + [feature for feature, _ in ranked[: max(0, expansion_limit - len(base))]]
        return expanded

    def recency_weight(self, event_id: str) -> float:
        rec = self.events.get(event_id)
        if rec is None:
            return 1.0
        age = max(0, self.total_exposures - rec.last_accessed_at)
        half_life = max(1.0, math.sqrt(max(1, len(self.events))) * self.dynamic_sequence_order())
        return 1.0 / (1.0 + age / half_life)

    def hdc_candidates(self, state: np.ndarray) -> set[str]:
        candidates: set[str] = set()
        for band in hdc_bands(state, max(1, len(self.events))):
            candidates.update(self.hdc_index.get(band, ()))
        return candidates
