from __future__ import annotations

from ._brain_common import *


class MemoryAnsweringMixin:
    def answer(self, prompt: str, max_new_tokens: int = 24) -> str:
        tokens = tokenize(prompt, max_tokens=64)
        if not tokens:
            return "I need input."
        generated = self.compose_answer(prompt, max_new_tokens=max_new_tokens)
        if generated:
            return generated
        return "I do not have enough memory for that yet."

    def compose_answer(self, prompt: str, max_new_tokens: int = 24) -> str:
        tokens = tokenize(prompt, max_tokens=64)
        energy_text = self.memory_energy_decode_text(prompt, tokens, max_new_tokens=max_new_tokens)
        if energy_text:
            return energy_text
        recalls = self.recall(prompt, limit=max(4, min(12, int(math.sqrt(max(1, len(self.events)))) + 3)))
        candidates = self.answer_candidates(prompt, tokens, recalls, max_new_tokens=max_new_tokens)
        if not candidates:
            return ""
        unique_candidates = list(dict.fromkeys(candidate for candidate in candidates if candidate.strip()))
        if len(unique_candidates) == 1:
            return self.trim_answer(unique_candidates[0], max_new_tokens)
        pred, scores = self.rank(prompt, unique_candidates, mode="event")
        chosen = unique_candidates[pred]
        return self.trim_answer(chosen, max_new_tokens)

    def answer_candidates(
        self,
        prompt: str,
        tokens: List[str],
        recalls: List[Dict[str, Any]],
        max_new_tokens: int = 24,
    ) -> List[str]:
        candidates = []
        analogy = self.answer_by_analogy(tokens, max_new_tokens)
        if analogy:
            candidates.append(analogy)
        graph_text = self.graph_decode_text(prompt, tokens, max_new_tokens=max_new_tokens)
        if graph_text:
            candidates.append(graph_text)
        ranked_fragments = self.recall_fragments(prompt, recalls)
        candidates.extend(ranked_fragments[:6])
        chain = self.event_chain_summary(prompt, recalls, limit=max(8, max_new_tokens))
        if chain:
            candidates.append(chain)
        if ranked_fragments:
            candidates.append(" ".join(ranked_fragments[: min(3, len(ranked_fragments))]))
        return candidates

    def feature_terms(self, feature: str) -> List[str]:
        canonical = canonical_feature(feature)
        if canonical.startswith("tok:") or canonical.startswith("stem:"):
            return [canonical.split(":", 1)[1]]
        if canonical.startswith("bi:") or canonical.startswith("tri:"):
            return [part for part in canonical.split(":", 1)[1].split("_") if part]
        return []

    def trim_answer(self, text: str, max_new_tokens: int) -> str:
        words = str(text).split()
        limit = max(1, int(max_new_tokens)) + 16
        return " ".join(words[:limit])

    def answer_by_analogy(self, tokens: List[str], max_new_tokens: int) -> str:
        if not tokens:
            return ""
        recalls = self.recall(" ".join(tokens), limit=max(2, int(math.sqrt(max(1, len(self.events))))))
        best_tail: List[str] = []
        suffix_limit = min(len(tokens), self.dynamic_sequence_order())
        for row in recalls:
            rec = self.events.get(row["event_id"])
            if not rec or not rec.sequence:
                continue
            seq = [symbol.split(":", 1)[-1] for symbol in rec.sequence if symbol.startswith("text:")]
            if not seq:
                continue
            for width in range(suffix_limit, 0, -1):
                needle = tokens[-width:]
                for pos in range(0, len(seq) - width + 1):
                    if seq[pos:pos + width] == needle:
                        tail = seq[pos + width:pos + width + max_new_tokens]
                        if len(tail) > len(best_tail):
                            best_tail = tail
                        break
                if best_tail:
                    break
        if not best_tail:
            return ""
        return " ".join(tokens + best_tail)

