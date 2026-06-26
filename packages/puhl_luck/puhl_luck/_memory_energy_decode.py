from __future__ import annotations

from ._brain_common import *


class MemoryEnergyDecodeMixin:
    def memory_energy_decode_text(
        self,
        prompt: str,
        tokens: List[str],
        max_new_tokens: int = 24,
        temperature: float = 0.0,
        beam_size: int = 1,
    ) -> str:
        if not tokens:
            return ""
        if beam_size > 1:
            return self.energy_beam_decode_text(prompt, tokens, max_new_tokens=max_new_tokens, beam_size=beam_size)
        context = list(tokens[-min(8, len(tokens)):])
        query_features = self.features_for_query(prompt)
        generation_features = self.generation_query_features(query_features)
        event_scores = self.event_scores_for_features(generation_features)
        semantic_energy = self.semantic_token_energy(query_features, event_scores)
        for _ in range(max(1, int(max_new_tokens))):
            scores = self.next_token_energy_scores(context, semantic_energy, event_scores)
            if not scores:
                break
            next_token = self.select_next_token(scores, context, temperature=temperature)
            if not next_token:
                break
            context.append(next_token)
        if len(context) <= len(tokens[-min(8, len(tokens)):]):
            return ""
        return " ".join(context)

    def event_scores_for_features(self, features: Iterable[str]) -> Counter[str]:
        scores: Counter[str] = Counter()
        for feature in features:
            fid = self.feature_to_id.get(feature)
            if fid is None:
                continue
            weight = self.retrieval_feature_weight(feature, anchor_present=canonical_feature(feature).startswith("id:"))
            for eid, count in self.feature_to_events.get(fid, {}).items():
                scores[eid] += weight * count
        return scores

    def generation_query_features(self, query_features: Iterable[str]) -> List[str]:
        strong_prefixes = ("id:", "tok:", "bi:", "tri:")
        out = []
        for feature in query_features:
            canonical = canonical_feature(feature)
            if canonical.startswith(strong_prefixes):
                out.append(feature)
        return list(dict.fromkeys(out))

    def semantic_token_energy(self, query_features: Iterable[str], event_scores: Counter[str], limit: int = 8) -> Counter[str]:
        scores: Counter[str] = Counter()
        for feature in query_features:
            token = self.token_from_feature(feature)
            if token:
                scores[token] += self.retrieval_feature_weight(feature)
        for eid, event_score in event_scores.most_common(max(1, int(limit))):
            rec = self.events.get(eid)
            if not rec:
                continue
            gain = math.log1p(float(event_score))
            for symbol in rec.sequence:
                if not symbol.startswith("text:"):
                    continue
                token = symbol.split(":", 1)[1]
                if token:
                    scores[token] += gain * 0.25
        return scores

    def next_token_energy_scores(
        self,
        context_tokens: List[str],
        semantic_energy: Dict[str, float],
        event_scores: Counter[str],
    ) -> Counter[str]:
        scores: Counter[str] = Counter()
        order_options, matched_order = self.order_backoff_options(context_tokens)
        for token, count in order_options.items():
            scores[token] += math.log1p(count) * (1.0 + matched_order)
        event_options = self.event_next_token_options(context_tokens, event_scores)
        for token, value in event_options.items():
            scores[token] += value * 1.4
        if not scores:
            for token, value in self.global_next_token_options(context_tokens).items():
                scores[token] += value
        for feature, value in semantic_energy.items():
            token = self.token_from_feature(feature) or str(feature)
            if token in scores:
                scores[token] += float(value) * 0.20
        if not scores:
            return scores
        recent = Counter(context_tokens[-8:])
        for token in list(scores.keys()):
            if recent[token]:
                scores[token] *= 1.0 / ((1.0 + recent[token]) ** 2)
            if len(context_tokens) >= 2 and any(context_tokens[i] == context_tokens[-1] and context_tokens[i + 1] == token for i in range(0, len(context_tokens) - 1)):
                scores[token] *= 0.05
        return Counter({token: score for token, score in scores.items() if score > 1e-9})

    def order_backoff_options(self, context_tokens: List[str]) -> Tuple[Counter[str], int]:
        max_order = min(self.dynamic_generation_order(), len(context_tokens))
        for order in range(max_order, 0, -1):
            options = self.order_contexts.get(tuple(context_tokens[-order:]))
            if options:
                return Counter(options), order
        return Counter(), 0

    def global_next_token_options(self, context_tokens: List[str], limit: int = 64) -> Counter[str]:
        scores: Counter[str] = Counter()
        recent = set(context_tokens[-8:])
        if context_tokens:
            for token, count in self.token_successors.get(context_tokens[-1], Counter()).most_common(limit):
                if token not in recent:
                    scores[token] += math.log1p(count) * 0.75
        if scores:
            return scores
        for token, count in self.sequence_starts.most_common(limit):
            if token not in recent:
                scores[token] += math.log1p(count) * 0.55
        if scores:
            return scores
        for token, count in self.token_unigrams.most_common(limit):
            if token not in recent:
                scores[token] += math.log1p(count) * 0.35
        return scores

    def event_next_token_options(self, context_tokens: List[str], event_scores: Counter[str], limit: int = 8) -> Counter[str]:
        out: Counter[str] = Counter()
        if not context_tokens:
            return out
        recent = set(context_tokens[-6:])
        for eid, score in event_scores.most_common(limit):
            rec = self.events.get(eid)
            if not rec or not rec.sequence:
                continue
            seq = [symbol.split(":", 1)[1] for symbol in rec.sequence if symbol.startswith("text:")]
            if not seq:
                continue
            max_width = min(self.dynamic_generation_order(), len(context_tokens), len(seq))
            for width in range(max_width, 0, -1):
                needle = context_tokens[-width:]
                found = False
                for pos in range(0, len(seq) - width):
                    if seq[pos:pos + width] == needle:
                        out[seq[pos + width]] += math.log1p(score) * (1.0 + width)
                        found = True
                if found:
                    break
        if out or len(event_scores) < 2:
            return out
        context_bigrams = set(zip(context_tokens, context_tokens[1:]))
        for eid, score in event_scores.most_common(limit):
            rec = self.events.get(eid)
            if not rec or not rec.sequence:
                continue
            seq = [symbol.split(":", 1)[1] for symbol in rec.sequence if symbol.startswith("text:")]
            prefix = seq[: min(4, len(seq))]
            if any(pair in context_bigrams for pair in zip(prefix, prefix[1:])):
                continue
            for pos, token in enumerate(seq[: min(4, len(seq))]):
                if token and token not in recent:
                    out[token] += math.log1p(float(score)) * (0.35 / (1.0 + pos))
        return out

    def energy_beam_decode_text(self, prompt: str, tokens: List[str], max_new_tokens: int = 24, beam_size: int = 3) -> str:
        query_features = self.features_for_query(prompt)
        generation_features = self.generation_query_features(query_features)
        event_scores = self.event_scores_for_features(generation_features)
        semantic_energy = self.semantic_token_energy(query_features, event_scores)
        beams: List[Tuple[List[str], float]] = [(list(tokens[-min(8, len(tokens)):]), 0.0)]
        width = max(1, min(8, int(beam_size)))
        for _ in range(max(1, int(max_new_tokens))):
            next_beams: List[Tuple[List[str], float]] = []
            for seq, score in beams:
                scores = self.next_token_energy_scores(seq, semantic_energy, event_scores)
                if not scores:
                    next_beams.append((seq, score))
                    continue
                for token, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:width]:
                    next_beams.append((seq + [token], score + math.log1p(value)))
            if not next_beams:
                break
            next_beams.sort(key=lambda item: (item[1] / max(1, len(item[0])), item[1]), reverse=True)
            beams = next_beams[:width]
        best = beams[0][0] if beams else []
        if len(best) <= len(tokens[-min(8, len(tokens)):]):
            return ""
        return " ".join(best)

