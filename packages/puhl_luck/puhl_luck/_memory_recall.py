from __future__ import annotations

from ._brain_common import *


class MemoryRecallMixin:
    def weighted_feature_overlap(self, query_set: set[str], choice_set: set[str]) -> float:
        shared = query_set & choice_set
        if not shared:
            return 0.0
        numerator = sum(self.feature_weight(feature) for feature in shared)
        denominator = math.sqrt(
            max(1e-9, sum(self.feature_weight(feature) for feature in query_set))
            * max(1e-9, sum(self.feature_weight(feature) for feature in choice_set))
        )
        return numerator / denominator

    def event_support_from_energy(self, energy: Dict[str, float], choice_set: set[str]) -> float:
        event_scores: Counter[str] = Counter()
        for feature, value in energy.items():
            fid = self.feature_to_id.get(feature)
            if fid is None:
                continue
            for eid, count in self.feature_to_events.get(fid, {}).items():
                event_scores[eid] += value * count
        support = 0.0
        for eid, event_score in event_scores.most_common(8):
            event_features = set(content_features(self.events[eid].features))
            shared = len(event_features & choice_set) / max(1, len(choice_set))
            support = max(support, shared * math.log1p(event_score))
        return support

    def event_support_from_scores(self, event_scores: Counter[str], choice_set: set[str]) -> float:
        if not event_scores or not choice_set:
            return 0.0
        support = 0.0
        for eid, event_score in event_scores.most_common(8):
            rec = self.events.get(eid)
            if rec is None:
                continue
            event_features = self.event_content_sets.get(eid)
            if event_features is None:
                event_features = set(content_features(rec.features))
                self.event_content_sets[eid] = event_features
            shared = len(event_features & choice_set) / math.sqrt(max(1.0, float(len(event_features) * len(choice_set))))
            support = max(support, shared * math.log1p(event_score))
        return support

    def event_alignment_from_scores(self, event_scores: Counter[str], choice_set: set[str]) -> float:
        if not event_scores or not choice_set:
            return 0.0
        numerator = 0.0
        denominator = 0.0
        for eid, event_score in event_scores.most_common(8):
            rec = self.events.get(eid)
            if rec is None:
                continue
            event_features = self.event_content_sets.get(eid)
            if event_features is None:
                event_features = set(content_features(rec.features))
                self.event_content_sets[eid] = event_features
            shared = len(event_features & choice_set) / max(1, len(event_features | choice_set))
            weight = math.log1p(event_score)
            numerator += shared * weight
            denominator += weight
        return numerator / max(1e-9, denominator)

    def choice_contrast(self, state: Dict[str, Any], choice_set: set[str]) -> float:
        if not choice_set:
            return 0.0
        query_set = state["query_set"]
        if not query_set:
            return 0.0
        shared = query_set & choice_set
        only_choice = choice_set - query_set
        positive = sum(self.feature_weight(feature) for feature in shared)
        negative = sum(1.0 / self.feature_weight(feature) for feature in only_choice)
        return positive / (positive + negative + 1e-9)

    def rank(self, query: str, choices: List[str], mode: Optional[str] = None) -> Tuple[int, List[float]]:
        selected_mode = mode or self.rank_mode
        result_key = (self.total_exposures, selected_mode, str(query), tuple(str(choice) for choice in choices))
        cached = self._rank_result_cache.get(result_key)
        if cached is not None:
            return cached
        state = self._rank_query_state(query)
        compiled = self.compiled_choices(choices)
        raw_scores = [self.mode_score(self.energy_score(state, row), selected_mode) for row in compiled]
        scores = self.normalize_compiled_choice_scores(raw_scores, compiled)
        if not scores:
            return 0, []
        result = (max(range(len(scores)), key=lambda i: scores[i]), scores)
        if len(self._rank_result_cache) > 2048:
            self._rank_result_cache.clear()
        self._rank_result_cache[result_key] = result
        return result

    def normalize_choice_scores(self, raw_scores: List[float], choices: List[str]) -> List[float]:
        return self.normalize_compiled_choice_scores(raw_scores, self.compiled_choices(choices))

    def normalize_compiled_choice_scores(self, raw_scores: List[float], compiled_choices: List[Dict[str, Any]]) -> List[float]:
        if not raw_scores:
            return []
        adjusted = []
        for score, row in zip(raw_scores, compiled_choices):
            adjusted.append(float(score) / max(1e-9, float(row["length_penalty"])))
        if len(adjusted) == 1:
            return [1.0]
        max_score = max(adjusted)
        min_score = min(adjusted)
        if max_score - min_score < 1e-12:
            return [1.0 / len(adjusted) for _ in adjusted]
        mean = sum(adjusted) / len(adjusted)
        var = sum((score - mean) ** 2 for score in adjusted) / len(adjusted)
        scale = math.sqrt(var) if var > 1e-12 else (max_score - min_score)
        z_scores = [(score - mean) / max(1e-9, scale) for score in adjusted]
        peak = max(z_scores)
        exps = [math.exp(max(-20.0, min(20.0, score - peak))) for score in z_scores]
        total = sum(exps) + 1e-12
        return [value / total for value in exps]

    def recall(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_features = self.features_for_query(query)
        expanded_query = self.expanded_query_features(query_features)
        energy = self.activation(expanded_query)
        event_scores: Counter[str] = Counter()
        for feature, value in energy.items():
            fid = self.feature_to_id.get(feature)
            if fid is None:
                continue
            for eid, count in self.feature_to_events.get(fid, {}).items():
                event_scores[eid] += value * count * self.recency_weight(eid)
        for recent_id in self.short_term_events:
            if recent_id in self.events:
                rec = self.events[recent_id]
                recent_overlap = len(set(content_features(rec.features)) & set(content_features(query_features)))
                if recent_overlap:
                    event_scores[recent_id] += recent_overlap * self.recency_weight(recent_id)
        for eid, score in self.hopfield_recall(expanded_query, iterations=2, top_k=max(self.dynamic_recall_top_k(), limit * 2)).items():
            event_scores[eid] += score * 10.0 * self.recency_weight(eid)
        rows = []
        for eid, score in event_scores.most_common(limit):
            rec = self.events[eid]
            rec.last_accessed_at = self.total_exposures
            rows.append({
                "event_id": eid,
                "score": float(score),
                "modality": rec.modality,
                "source": rec.source,
                "label": rec.label,
                "preview": rec.preview,
                "novelty": rec.novelty,
            })
        return rows

