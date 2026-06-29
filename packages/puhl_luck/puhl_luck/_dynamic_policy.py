
"""Data-scale runtime policy for PUHL-LUCK.

Model capacity is derived only from observed data state: events, features, rows,
vocabulary, and prompt length.  Runtime hardware information such as hardware parallelism,
worker count, or host architecture is intentionally excluded from
all model-capacity formulas.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple


def _ceil_pow2(x: int) -> int:
    x = max(1, int(x))
    return 1 << (x - 1).bit_length()


def _scale(*values: int) -> int:
    return max(1, int(sum(max(0, int(v)) for v in values)))


def dynamic_top_k(vocab_size: int, row_count: int, event_count: int, feature_count: int = 0, prompt_tokens: int = 0) -> int:
    s = _scale(vocab_size, row_count, event_count, feature_count, prompt_tokens)
    # Candidate width grows with actual ambiguity, not a fixed k.
    return max(1, min(max(1, int(vocab_size) + 1), int(math.ceil(math.sqrt(s))) + int(math.ceil(math.log2(s + 1)))))


def dynamic_rank_k(vocab_size: int, row_count: int, event_count: int, active_rows: int = 0) -> int:
    s = _scale(vocab_size, row_count, event_count, active_rows)
    return max(1, dynamic_top_k(vocab_size or s, row_count, event_count, active_rows) + int(math.ceil(math.log2(s + 1))))


def dynamic_max_tokens(prompt_tokens: int, learned_events: int, vocab_size: int, requested: Optional[int] = None) -> int:
    if requested is not None and int(requested) > 0:
        return int(requested)
    s = _scale(prompt_tokens, learned_events, vocab_size)
    return max(1, int(math.ceil(math.sqrt(max(1, prompt_tokens + 1)))) + int(math.ceil(math.log2(s + 1))))


def dynamic_context_window(event_count: int, prompt_tokens: int, vocab_size: int) -> int:
    s = _scale(event_count, prompt_tokens, vocab_size)
    return max(1, int(math.ceil(math.sqrt(s))) + int(math.ceil(math.log2(s + 1))))


def dynamic_clause_budget(event_count: int, feature_count: int, vocab_size: int, row_count: int) -> int:
    s = _scale(event_count, feature_count, vocab_size, row_count)
    return max(1, int(math.ceil(math.sqrt(s * max(1, feature_count + vocab_size)))) + int(math.ceil(math.log2(s + 1))))


def dynamic_cache_size(event_count: int, feature_count: int, vocab_size: int, row_count: int) -> int:
    s = _scale(event_count, feature_count, vocab_size, row_count)
    # Data-only cache budget.  No CPU/thread/worker count is allowed here.
    return _ceil_pow2(int(math.ceil(s + math.sqrt(s))))


def dynamic_hdc_bits(feature_count: int, event_count: int, row_count: int = 0, vocab_size: int = 0) -> int:
    s = _scale(feature_count, event_count, row_count, vocab_size)
    # Data-only dimensionality.  It deliberately does not align to CPU word size.
    return max(1, int(math.ceil(math.log2(s + 1))) * int(math.ceil(math.sqrt(s))))


def dynamic_hdc_band_bits(bits: int, row_count: int, event_count: int) -> int:
    # Bucket granularity follows dimensionality and row pressure only.
    s = _scale(bits, row_count, event_count)
    return max(1, min(int(bits), int(math.ceil(math.sqrt(s)))))


def dynamic_hdc_neighbors(row_count: int, event_count: int, vocab_size: int) -> int:
    s = _scale(row_count, event_count, vocab_size)
    return max(1, int(math.ceil(math.sqrt(max(1, row_count)))) + int(math.ceil(math.log2(s + 1))))


def dynamic_hdc_source_budget(feature_count: int, row_count: int, event_count: int, active_rows: int) -> int:
    s = _scale(feature_count, row_count, event_count, active_rows)
    return max(1, min(max(1, active_rows), int(math.ceil(math.sqrt(s))) + int(math.ceil(math.log2(s + 1)))))


def dynamic_readout_shape(feature_count: int, event_count: int, row_count: int = 0, vocab_size: int = 0, min_hidden: int = 0, max_hidden: int = 0, min_vocab_cap: int = 0, max_vocab_cap: int = 0) -> Tuple[int, int]:
    s = _scale(feature_count, event_count, row_count, vocab_size)
    hidden = int(math.ceil(math.sqrt(max(1, feature_count + row_count + vocab_size)))) * max(1, int(math.ceil(math.log2(s + 1))))
    hidden = _ceil_pow2(max(int(min_hidden or 0), hidden))
    if max_hidden and int(max_hidden) > 0:
        hidden = min(hidden, int(max_hidden))
    vocab_cap = _ceil_pow2(max(int(min_vocab_cap or 0), int(vocab_size) + dynamic_top_k(vocab_size, row_count, event_count, feature_count)))
    if max_vocab_cap and int(max_vocab_cap) > 0:
        vocab_cap = min(vocab_cap, int(max_vocab_cap))
    return max(1, hidden), max(1, vocab_cap)


def dynamic_readout_active_budget(hidden_dim: int, event_count: int, vocab_size: int, row_count: int) -> int:
    if hidden_dim <= 0:
        return 0
    s = _scale(hidden_dim, event_count, vocab_size, row_count)
    return max(1, min(int(hidden_dim), int(math.ceil(math.sqrt(s))) + int(math.ceil(math.log2(s + 1)))))


def dynamic_readout_score_gain(event_count: int, active_features: int, candidates: int) -> float:
    s = _scale(event_count, active_features, candidates)
    return 1.0 / math.sqrt(1.0 + 1.0 / float(s))


def dynamic_learning_rate(event_count: int, active_features: int) -> float:
    s = _scale(event_count, active_features)
    return 1.0 / math.sqrt(float(s))


def dynamic_hebbian_rows(active_rows: int, event_count: int, row_count: int) -> int:
    return max(1, min(max(1, active_rows), dynamic_hdc_source_budget(row_count, row_count, event_count, active_rows)))


def dynamic_pull_bits(hdc_bits: int, amount: int, row_weight: float = 1.0) -> int:
    if hdc_bits <= 0:
        return 0
    strength = max(1.0, float(amount)) * max(0.0, float(row_weight))
    return max(1, min(int(hdc_bits), int(math.ceil(math.log2(hdc_bits + 1) * math.sqrt(strength + 1.0)))))
