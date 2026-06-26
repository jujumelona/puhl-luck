from __future__ import annotations

from ._brain_common import *


class MemoryHopfieldMixin:
    def hopfield_recall(self, query_features: List[str], iterations: int = 2, top_k: Optional[int] = None) -> Dict[str, float]:
        state = bundle_hv(query_features, self.hdc_bits)
        if state.size == 0 or not self.event_hv:
            return {}
        k = top_k if top_k is not None else self.dynamic_recall_top_k()
        event_scores: Dict[str, float] = {}
        for _ in range(max(1, iterations)):
            candidates = self.hdc_candidates(state)
            if not candidates:
                break
            ranked = sorted(
                ((eid, hv_similarity(state, self.event_hv[eid], self.hdc_bits)) for eid in candidates),
                key=lambda item: item[1],
                reverse=True,
            )[:k]
            event_scores = {eid: score for eid, score in ranked if score > 0.45}
            if not event_scores:
                break
            vectors = [self.event_hv[eid] for eid in event_scores if eid in self.event_hv]
            if not vectors:
                break
            state = np.bitwise_xor.reduce(np.stack(vectors), axis=0).astype(np.uint64, copy=False)
        return event_scores

    def hopfield_recall_continuous(
        self,
        query_features: List[str],
        iterations: int = 2,
        top_k: Optional[int] = None,
        beta: float = 8.0,
        max_patterns: int = 512,
    ) -> Dict[str, float]:
        query_vec = bundle_hv(query_features, self.hdc_bits)
        if query_vec.size == 0 or not self.event_hv:
            return {}
        candidate_ids = list(self.event_hv.keys())
        if len(candidate_ids) > max_patterns:
            indexed = self.hdc_candidates(query_vec)
            if indexed:
                candidate_ids = list(indexed)
        if not candidate_ids:
            return {}
        if len(candidate_ids) > max_patterns:
            candidate_ids = sorted(
                candidate_ids,
                key=lambda eid: hv_similarity(query_vec, self.event_hv[eid], self.hdc_bits),
                reverse=True,
            )[:max_patterns]
        patterns = np.stack([self.hv_bipolar(self.event_hv[eid]) for eid in candidate_ids]).astype(np.float32, copy=False)
        state = self.hv_bipolar(query_vec).astype(np.float32, copy=False)
        norm = max(1.0, float(state.size))
        for _ in range(max(1, iterations)):
            logits = (patterns @ state) / norm
            logits = logits * float(beta)
            logits = logits - float(np.max(logits))
            weights = np.exp(logits)
            weights = weights / (float(np.sum(weights)) + 1e-12)
            state = weights @ patterns
            state_norm = float(np.linalg.norm(state))
            if state_norm > 1e-9:
                state = state / state_norm * math.sqrt(norm)
        scores = (patterns @ state) / norm
        ranked = sorted(zip(candidate_ids, scores.tolist()), key=lambda item: item[1], reverse=True)
        k = top_k if top_k is not None else self.dynamic_recall_top_k()
        return {eid: float(score) for eid, score in ranked[:k] if score > 0.0}

    def hv_bipolar(self, value: np.ndarray) -> np.ndarray:
        if value.size == 0:
            return np.zeros(0, dtype=np.float32)
        bits = np.unpackbits(value[: self.hdc_words].view(np.uint8)).astype(np.float32, copy=False)
        return bits * 2.0 - 1.0

    def hopfield_recall_feature_continuous(
        self,
        query_features: List[str],
        iterations: int = 2,
        top_k: Optional[int] = None,
        beta: float = 4.0,
        max_patterns: int = 512,
    ) -> Dict[str, float]:
        query_content = content_features(query_features)
        if not query_content or not self.events:
            return {}
        candidate_ids: set[str] = set()
        anchors = self.anchor_features(query_features)
        anchor_ids: set[str] = set()
        for anchor in anchors:
            fid = self.feature_to_id.get(anchor)
            if fid is not None:
                anchor_ids.update(self.feature_to_events.get(fid, {}).keys())
        if anchor_ids:
            candidate_ids = anchor_ids
        for feature in ([] if anchor_ids else [*anchors, *query_content]):
            fid = self.feature_to_id.get(feature)
            if fid is None:
                continue
            candidate_ids.update(self.feature_to_events.get(fid, {}).keys())
        if not candidate_ids:
            candidate_ids = set(self.events.keys())
        if len(candidate_ids) > max_patterns:
            seed = Counter()
            for feature in query_content:
                fid = self.feature_to_id.get(feature)
                if fid is None:
                    continue
                for eid, count in self.feature_to_events.get(fid, {}).items():
                    seed[eid] += count * self.retrieval_feature_weight(feature, anchor_present=bool(anchors))
            candidate_ids = set(eid for eid, _ in seed.most_common(max_patterns))
        state = Counter({feature: self.retrieval_feature_weight(feature, anchor_present=bool(anchors)) for feature in query_content})
        event_sets: Dict[str, set[str]] = {}
        event_norms: Dict[str, float] = {}
        for eid in candidate_ids:
            rec = self.events.get(eid)
            if rec is None:
                continue
            features = self.event_content_sets.get(eid)
            if features is None:
                features = set(content_features(rec.features))
                self.event_content_sets[eid] = features
            event_sets[eid] = features
            event_norms[eid] = math.sqrt(max(1e-9, sum(self.retrieval_feature_weight(f, anchor_present=bool(anchors)) for f in features)))
        scores: Dict[str, float] = {}
        for _ in range(max(1, iterations)):
            state_norm = math.sqrt(max(1e-9, sum(value * value for value in state.values())))
            logits = []
            for eid, features in event_sets.items():
                shared = sum(state.get(feature, 0.0) for feature in features)
                logits.append((eid, shared / max(1e-9, state_norm * event_norms[eid])))
            if not logits:
                break
            peak = max(score for _, score in logits)
            weights = [(eid, math.exp((score - peak) * float(beta))) for eid, score in logits]
            total = sum(weight for _, weight in weights) + 1e-12
            scores = {eid: weight / total for eid, weight in weights}
            next_state: Counter[str] = Counter()
            for eid, weight in scores.items():
                for feature in event_sets[eid]:
                    next_state[feature] += weight * self.retrieval_feature_weight(feature, anchor_present=bool(anchors))
            state = next_state
        k = top_k if top_k is not None else self.dynamic_recall_top_k()
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return {eid: float(score) for eid, score in ranked if score > 0.0}
