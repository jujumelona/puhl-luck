from __future__ import annotations

from ._brain_common import *


class MemoryFragmentsMixin:
    def token_from_feature(self, feature: str) -> str:
        canonical = canonical_feature(feature)
        if canonical.startswith("tok:"):
            token = canonical.split(":", 1)[1]
            if len(token) > 1 and not token.isdigit():
                return token
        return ""

    def recall_fragments(self, prompt: str, recalls: List[Dict[str, Any]]) -> List[str]:
        query_features = set(content_features(self.features_for_query(prompt)))
        scored: List[Tuple[float, str]] = []
        for row in recalls:
            rec = self.events.get(row["event_id"])
            if rec is None:
                continue
            rec_features = self.event_content_sets.get(rec.event_id)
            if rec_features is None:
                rec_features = set(content_features(rec.features))
                self.event_content_sets[rec.event_id] = rec_features
            overlap = self.weighted_feature_overlap(query_features, rec_features)
            for fragment in self.preview_fragments(rec.preview):
                scored.append((float(row.get("score", 0.0)) + overlap * 10.0, fragment))
        scored.sort(key=lambda item: item[0], reverse=True)
        return list(dict.fromkeys(fragment for _, fragment in scored if fragment))

    def preview_fragments(self, text: str) -> List[str]:
        cleaned = " ".join(str(text).replace("\ufeff", " ").split())
        if not cleaned:
            return []
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if part.strip()]
        if parts:
            return parts[:4]
        tokens = cleaned.split()
        if len(tokens) <= 24:
            return [cleaned]
        return [" ".join(tokens[i:i + 24]) for i in range(0, min(len(tokens), 72), 18)]

    def event_chain_summary(self, prompt: str, recalls: List[Dict[str, Any]], limit: int = 24) -> str:
        query_features = self.features_for_query(prompt)
        expanded = self.expanded_query_features(query_features, limit=max(16, min(64, limit * 2)))
        energy = self.activation(expanded, hops=2)
        terms = []
        for feature, _ in sorted(energy.items(), key=lambda item: item[1], reverse=True):
            for term in self.feature_terms(feature):
                if term not in terms:
                    terms.append(term)
            if len(terms) >= limit:
                break
        recalled_terms = []
        for row in recalls[:4]:
            rec = self.events.get(row["event_id"])
            if rec is None:
                continue
            for symbol in rec.sequence:
                if not symbol.startswith("text:"):
                    continue
                term = symbol.split(":", 1)[1]
                if term not in recalled_terms:
                    recalled_terms.append(term)
                if len(recalled_terms) >= limit:
                    break
        merged = list(dict.fromkeys([*tokenize(prompt, max_tokens=64), *terms, *recalled_terms]))
        if not merged:
            return ""
        return " ".join(merged[:limit])

