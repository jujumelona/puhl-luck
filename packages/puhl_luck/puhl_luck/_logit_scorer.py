
"""Deterministic greedy scorer for weightless sparse count evidence."""
from __future__ import annotations
from collections import deque
from typing import List, Tuple, Optional


class LogitScorer:
    def __init__(self, repetition_penalty_weight: float = 2.0, repetition_window: int = 8, **_: object) -> None:
        self.repetition_penalty_weight = float(repetition_penalty_weight)
        self.repetition_window = int(repetition_window)
        self.EOS = '[EOS]'
        self.BOS = '[BOS]'
        self.SEP = '[SEP]'

    def score_candidates(
        self,
        candidates: List[Tuple[str, float]],
        recent_tokens: Optional[deque] = None,
        step: int = 0,
        **_: object,
    ) -> List[Tuple[str, float]]:
        recent_tokens = recent_tokens or deque(maxlen=self.repetition_window)
        recent = list(recent_tokens)
        scored: List[Tuple[str, float]] = []
        for tok, base in candidates:
            if tok in {self.BOS, self.SEP}:
                continue
            s = float(base)
            if tok in recent:
                s -= self.repetition_penalty_weight * recent.count(tok)
            if tok == self.EOS and step < 1:
                s -= 4.0
            scored.append((tok, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def get_top_token(self, scored_candidates: List[Tuple[str, float]]) -> str:
        return scored_candidates[0][0] if scored_candidates else self.EOS

    def sample_top_k(self, scored_candidates: List[Tuple[str, float]], *args, **kwargs) -> str:
        return self.get_top_token(scored_candidates)
