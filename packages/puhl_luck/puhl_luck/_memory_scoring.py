from __future__ import annotations

from ._brain_common import *


class MemoryScoringMixin:
    def score(self, query: str, choice: str) -> float:
        state = self._rank_query_state(query)
        compiled = self.compiled_choices([choice])[0]
        if not state["query_signal"] or not compiled["features"]:
            return 0.0
        return self._score_compiled_choice_from_state(state, compiled)

    def _rank_query_state(self, query: str) -> Dict[str, Any]:
        cached_state = self._rank_state_cache.get(str(query))
        if cached_state is not None:
            return cached_state
        query_features = self.cached_features_for_query(query)
        if not query_features:
            return {
                "query_signal": [],
                "query_set": set(),
                "energy": {},
                "event_scores": Counter(),
            }
        anchors = self.anchor_features(query_features)
        expanded_query = list(dict.fromkeys([*anchors, *query_features[:64]]))
        query_signal = self.cached_content_for_query(query)
        anchor_present = bool(anchors)
        energy = {feature: self.retrieval_feature_weight(feature, anchor_present=anchor_present) for feature in expanded_query}
        event_scores: Counter[str] = Counter()
        cap = self.dynamic_rank_event_cap()
        for anchor in anchors:
            fid = self.feature_to_id.get(anchor)
            if fid is None:
                continue
            for eid, count in self.feature_to_events.get(fid, {}).items():
                event_scores[eid] += self.retrieval_feature_weight(anchor, anchor_present=True) * max(1, count) * 100.0
        for feature, value in energy.items():
            fid = self.feature_to_id.get(feature)
            if fid is None:
                continue
            rows = self.feature_top_events.get(fid)
            if not rows:
                rows = self.feature_to_events.get(fid, {})
            for eid, count in rows.items():
                event_scores[eid] += value * count * self.recency_weight(eid)
        state = {
            "query_signal": query_signal,
            "query_set": set(query_signal),
            "energy": energy,
            "event_scores": event_scores,
        }
        if len(self._rank_state_cache) > 1024:
            self._rank_state_cache.clear()
        self._rank_state_cache[str(query)] = state
        return state

    def _score_choice_from_state(self, state: Dict[str, Any], choice_features: List[str]) -> float:
        return self._score_compiled_choice_from_state(
            state,
            {
                "features": choice_features,
                "content": content_features(choice_features),
                "content_set": set(content_features(choice_features)),
                "freq_penalty": 0.0,
            },
        )

    def _score_compiled_choice_from_state(self, state: Dict[str, Any], compiled_choice: Dict[str, Any]) -> float:
        return self.mode_score(self.energy_score(state, compiled_choice), self.rank_mode)

    def energy_score(self, state: Dict[str, Any], compiled_choice: Dict[str, Any]) -> Dict[str, float]:
        choice_signal = compiled_choice["content"]
        if not choice_signal:
            return {"score": 0.0, "free_energy": 1e9}
        energy = state["energy"]
        query_set = state["query_set"]
        choice_set = compiled_choice["content_set"]
        overlap = len(query_set & choice_set) / math.sqrt(max(1.0, float(len(query_set) * len(choice_set))))
        weighted_overlap = self.weighted_feature_overlap(query_set, choice_set)
        resonance = sum(energy.get(feature, 0.0) for feature in choice_set) / math.sqrt(max(1.0, float(len(choice_set))))
        event_support = self.event_support_from_scores(state["event_scores"], choice_set)
        alignment = self.event_alignment_from_scores(state["event_scores"], choice_set)
        contrast = self.choice_contrast(state, choice_set)
        freq_penalty = compiled_choice.get("freq_penalty", 0.0)
        evidence = (
            1.0 * overlap
            + 2.2 * weighted_overlap
            + 1.6 * event_support
            + 1.2 * alignment
            + 0.7 * contrast
            + resonance / (1.0 + 0.05 * freq_penalty)
        )
        entropy = self.choice_entropy(choice_set)
        conflict = self.choice_conflict(query_set, choice_set)
        free_energy = conflict + 0.20 * entropy + 0.05 * freq_penalty - evidence
        return {
            "score": evidence - 0.20 * entropy - 0.05 * freq_penalty - conflict,
            "free_energy": free_energy,
            "overlap": overlap,
            "weighted_overlap": weighted_overlap,
            "resonance": resonance,
            "event_support": event_support,
            "alignment": alignment,
            "contrast": contrast,
            "entropy": entropy,
            "conflict": conflict,
            "freq_penalty": freq_penalty,
        }

    def mode_score(self, breakdown: Dict[str, float], mode: Optional[str] = None) -> float:
        selected = mode or self.rank_mode
        weights = ENERGY_MODES.get(selected, ENERGY_MODES["free_energy"])
        return sum(float(breakdown.get(name, 0.0)) * weight for name, weight in weights.items())

    def choice_entropy(self, choice_set: set[str]) -> float:
        if not choice_set:
            return 0.0
        weights = [1.0 / max(1e-9, self.feature_weight(feature)) for feature in choice_set]
        total = sum(weights) + 1e-12
        entropy = 0.0
        for weight in weights:
            p = weight / total
            entropy -= p * math.log(p + 1e-12)
        return entropy / math.log(len(weights) + 1.0)

    def choice_conflict(self, query_set: set[str], choice_set: set[str]) -> float:
        if not query_set or not choice_set:
            return 1.0
        missing = choice_set - query_set
        common = query_set & choice_set
        missing_cost = sum(1.0 / max(1e-9, self.feature_weight(feature)) for feature in missing)
        common_gain = sum(self.feature_weight(feature) for feature in common)
        return missing_cost / (missing_cost + common_gain + 1e-9)

    def rank_energy_breakdown(self, query: str, choices: List[str]) -> List[Dict[str, float]]:
        state = self._rank_query_state(query)
        return [self.energy_score(state, row) for row in self.compiled_choices(choices)]

    def explain_rank(self, query: str, choices: List[str], mode: Optional[str] = None, top_events: int = 5) -> Dict[str, Any]:
        selected_mode = mode or self.rank_mode
        state = self._rank_query_state(query)
        compiled = self.compiled_choices(choices)
        breakdown = [self.energy_score(state, row) for row in compiled]
        raw_scores = [self.mode_score(row, selected_mode) for row in breakdown]
        scores = self.normalize_compiled_choice_scores(raw_scores, compiled)
        prediction = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
        event_rows = []
        for eid, event_score in state["event_scores"].most_common(max(1, top_events)):
            rec = self.events.get(eid)
            if rec is None:
                continue
            event_features = self.event_content_sets.get(eid)
            if event_features is None:
                event_features = set(content_features(rec.features))
                self.event_content_sets[eid] = event_features
            event_rows.append({
                "event_id": eid,
                "score": float(event_score),
                "source": rec.source,
                "modality": rec.modality,
                "preview": rec.preview,
                "shared_query_features": sorted(set(state["query_signal"]) & event_features)[:16],
            })
        choice_rows = []
        for idx, row in enumerate(compiled):
            choice_rows.append({
                "choice": row["choice"],
                "score": scores[idx] if idx < len(scores) else 0.0,
                "raw_score": raw_scores[idx] if idx < len(raw_scores) else 0.0,
                "breakdown": breakdown[idx],
                "shared_query_features": sorted(state["query_set"] & row["content_set"])[:16],
            })
        return {
            "query": query,
            "mode": selected_mode,
            "prediction": prediction,
            "answer": choices[prediction] if choices and prediction < len(choices) else "",
            "choices": choice_rows,
            "events": event_rows,
        }

