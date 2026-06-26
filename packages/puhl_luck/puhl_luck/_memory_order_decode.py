from __future__ import annotations

from ._brain_common import *


class MemoryOrderDecodeMixin:
    def graph_decode_text(
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
            return self.beam_decode_text(prompt, tokens, max_new_tokens=max_new_tokens, beam_size=beam_size)
        query_features = self.features_for_query(prompt)
        expanded = self.expanded_query_features(query_features, limit=64)
        energy = self.activation(expanded, hops=2)
        out = list(tokens[-min(6, len(tokens)):])
        used = Counter(out)
        for _ in range(max(1, int(max_new_tokens))):
            scores, matched_order = self.next_token_scores(out, energy)
            if not scores:
                break
            if matched_order == 0:
                break
            for token, count in list(used.items()):
                if token in scores:
                    scores[token] *= 1.0 / ((1.0 + count) ** 2)
            next_token = self.select_next_token(scores, out, temperature=temperature)
            score = scores.get(next_token, 0.0)
            if score <= 0.0 or (len(out) >= 3 and next_token == out[-1] == out[-2] == out[-3]):
                break
            out.append(next_token)
            used[next_token] += 1
        if len(out) <= len(tokens[-min(6, len(tokens)):]):
            return ""
        return " ".join(out)

    def next_token_scores(self, context_tokens: List[str], energy: Dict[str, float]) -> Tuple[Counter[str], int]:
        max_order = min(self.dynamic_generation_order(), len(context_tokens))
        for order in range(max_order, 0, -1):
            context = tuple(context_tokens[-order:])
            options = self.order_contexts.get(context)
            if not options:
                continue
            scores: Counter[str] = Counter()
            for token, count in options.most_common(64):
                scores[token] += float(count) * (1.0 + order)
            for feature, value in energy.items():
                token = self.token_from_feature(feature)
                if token and token in scores:
                    scores[token] += float(value) * 0.35
            return scores, order
        return Counter(), 0

    def select_next_token(self, scores: Counter[str], context_tokens: List[str], temperature: float = 0.0) -> str:
        recent = set(context_tokens[-4:])
        fresh_scores = Counter({token: score for token, score in scores.items() if token not in recent})
        if fresh_scores:
            scores = fresh_scores
        if temperature <= 0.0:
            return max(scores.items(), key=lambda item: (item[1], item[0]))[0]
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:16]
        if not ranked:
            return ""
        peak = ranked[0][1]
        temp = max(1e-6, float(temperature))
        weights = [math.exp((score - peak) / temp) for _, score in ranked]
        total = sum(weights) + 1e-12
        seed_text = "|".join(context_tokens[-8:]) + "|" + ",".join(token for token, _ in ranked)
        draw = int.from_bytes(hashlib.blake2b(seed_text.encode("utf-8", errors="ignore"), digest_size=8).digest(), "little") / float(2 ** 64)
        acc = 0.0
        for (token, _), weight in zip(ranked, weights):
            acc += weight / total
            if draw <= acc:
                return token
        return ranked[-1][0]

    def beam_decode_text(self, prompt: str, tokens: List[str], max_new_tokens: int = 24, beam_size: int = 3) -> str:
        query_features = self.features_for_query(prompt)
        expanded = self.expanded_query_features(query_features, limit=64)
        energy = self.activation(expanded, hops=2)
        beams: List[Tuple[List[str], float]] = [(list(tokens[-min(6, len(tokens)):]), 0.0)]
        width = max(1, min(8, int(beam_size)))
        for _ in range(max(1, int(max_new_tokens))):
            next_beams: List[Tuple[List[str], float]] = []
            for seq, score in beams:
                scores, matched_order = self.next_token_scores(seq, energy)
                if not scores or matched_order == 0:
                    next_beams.append((seq, score))
                    continue
                recent = set(seq[-4:])
                scored = [(token, value) for token, value in scores.items() if token not in recent]
                if not scored:
                    scored = list(scores.items())
                for token, value in sorted(scored, key=lambda item: item[1], reverse=True)[:width]:
                    if len(seq) >= 2 and any(seq[i] == seq[-1] and seq[i + 1] == token for i in range(0, len(seq) - 1)):
                        continue
                    next_beams.append((seq + [token], score + math.log1p(max(0.0, value))))
            if not next_beams:
                break
            next_beams.sort(key=lambda item: (item[1] / max(1, len(item[0])), item[1]), reverse=True)
            beams = next_beams[:width]
        best = beams[0][0] if beams else []
        if len(best) <= len(tokens[-min(6, len(tokens)):]):
            return ""
        return " ".join(best)

