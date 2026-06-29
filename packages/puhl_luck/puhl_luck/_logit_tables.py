"""P73 dynamic hybrid evidence table: HDC binding plus adaptive readout.

The main memory remains sparse/HDC evidence, but P73 deliberately adds a
trainable margin readout on top of HDC bind/bundle context vectors. The readout
size is no longer a fixed 256/8192 choice: hidden width, token capacity, active
projection budget, and scoring work grow from observed data scale and can keep
growing until an optional user-supplied budget stops them. The objective is to
beat dense baselines with far fewer learned parameters, not to keep zero learned
weights.

    exact feature_id -> Counter(next_token)
    HDC(context bundle vector) -> Counter(next_token)
    HDC(bind(feature_i, feature_j, ...)) -> Counter(next_token)

P65 fixed the P64 mistake: HDC dimensionality is no longer a fixed constant.
It grows from observed data scale (feature count, event count, HDC row count,
vocabulary size). Existing rows keep the dimensionality they were indexed with;
new rows use the current larger dimensionality. KNN compares each row only over
its stored dimensionality, so growth does not invalidate old rows and has no hard
coded upper bound beyond machine memory/time.
"""
from __future__ import annotations

import math
import heapq
import pickle
import gzip
import hashlib
from ._dynamic_policy import (
    dynamic_hdc_bits as _policy_hdc_bits,
    dynamic_hdc_band_bits,
    dynamic_hdc_neighbors,
    dynamic_hdc_source_budget,
    dynamic_readout_shape as _policy_readout_shape,
    dynamic_readout_active_budget,
    dynamic_readout_score_gain,
    dynamic_learning_rate,
    dynamic_hebbian_rows,
    dynamic_pull_bits,
    dynamic_cache_size,
)
from collections import Counter, defaultdict, OrderedDict
from typing import Any, Dict, Iterable, List, Tuple, Set

# Capacity defaults are data-derived.  Zero means "derive from observed stream";
# nonzero values are treated only as explicit caller budgets, never as hidden model caps.
HDC_MIN_BITS = 0
HDC_BAND_BITS = 0
HEBB_MAX_ROWS = 0
HEBB_PULL_BITS_STRONG = 0
HEBB_PULL_BITS_WEAK = 0
HDC_COMPOSITE_MIN_SOURCES = 0
READOUT_DEFAULT_HIDDEN = 0
READOUT_DEFAULT_VOCAB_CAP = 0
READOUT_MIN_HIDDEN = 0
READOUT_MIN_VOCAB_CAP = 0
READOUT_DEFAULT_LR = 0.0
READOUT_SCORE_GAIN = 0.0
READOUT_ACTIVE_FEATURES = 0


# No hard max. This is only the rounding unit for fast word-aligned popcount.
def _round_up64(x: int) -> int:
    x = max(64, int(x))
    return ((x + 63) // 64) * 64


def dynamic_hdc_bits(feature_count: int, event_count: int, row_count: int = 0, vocab_size: int = 0) -> int:
    """Choose HDC dimensionality from data scale, not a fixed hand number."""
    return int(_policy_hdc_bits(feature_count, event_count, row_count, vocab_size))


def _ceil_pow2(x: int) -> int:
    x = max(1, int(x))
    return 1 << (x - 1).bit_length()


def dynamic_readout_shape(
    feature_count: int,
    event_count: int,
    row_count: int = 0,
    vocab_size: int = 0,
    min_hidden: int = READOUT_MIN_HIDDEN,
    max_hidden: int = 0,
    min_vocab_cap: int = READOUT_MIN_VOCAB_CAP,
    max_vocab_cap: int = 0,
) -> Tuple[int, int]:
    """Choose readout shape from observed data scale and explicit caller budgets."""
    return _policy_readout_shape(feature_count, event_count, row_count, vocab_size, min_hidden, max_hidden, min_vocab_cap, max_vocab_cap)


def dynamic_readout_config(
    vocab_size: int,
    feature_count: int,
    event_count: int
) -> Dict[str, Any]:
    """Auto-size readout based on learned data (Task 3.5).
    
    Implements dynamic adaptive readout configuration:
    - Hidden dimension: sqrt(vocab_size * feature_count), clamped to [64, 2048]
    - Vocab cap: sqrt(event_count) * log2(vocab_size), clamped appropriately
    - Learning rate decay: 0.01 / log2(event_count + 2)
    
    This provides adaptive capacity that prevents underfitting (too small) and
    overfitting (too large) by scaling with observed data patterns.
    
    Args:
        vocab_size: Number of unique tokens in vocabulary
        feature_count: Number of feature rows learned
        event_count: Number of training events/updates
        
    Returns:
        Dict with keys:
        - hidden_dim: Hidden layer dimension for adaptive readout
        - vocab_cap: Maximum vocabulary capacity for readout
        - learning_rate: Adaptive learning rate that decays with data
        
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 4.1
    """
    vocab_size = max(1, int(vocab_size))
    feature_count = max(1, int(feature_count))
    event_count = max(1, int(event_count))
    
    # Calculate hidden_dim as sqrt(vocab_size * feature_count), clamped to [64, 2048]
    hidden_dim = int(math.sqrt(vocab_size * feature_count))
    hidden_dim = max(64, min(2048, hidden_dim))  # Clamp to reasonable range
    
    # Calculate vocab_cap as sqrt(event_count) * log2(vocab_size), clamped appropriately
    # Add small constant to avoid log(0) or log(1)=0
    vocab_cap = int(math.sqrt(event_count) * math.log2(vocab_size + 2))
    # Clamp: at least 100, but never exceed vocab_size
    vocab_cap = min(vocab_size, max(100, vocab_cap))
    
    # Implement adaptive learning rate decay: 0.01 / log2(event_count + 2)
    learning_rate = 0.01 / math.log2(event_count + 2)
    
    return {
        'hidden_dim': hidden_dim,
        'vocab_cap': vocab_cap,
        'learning_rate': learning_rate
    }


def _fnv1a64(s: str) -> int:
    h = 0xcbf29ce484222325
    for b in str(s).encode('utf-8', 'ignore'):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF



def _rotl(x: int, r: int, bits: int) -> int:
    bits = max(1, int(bits))
    r = int(r) % bits
    x = int(x) & _mask(bits)
    if r == 0:
        return x
    return int(((x << r) | (x >> (bits - r))) & _mask(bits))


def _bind_hv(a: int, b: int, bits: int, salt: int = 0) -> int:
    """HDC binding for composite features.

    Binding keeps the result in the same D-dimensional space.  It creates no
    symbolic feature id and no trainable dense parameter.  XOR plus position
    rotation is used so A+B is not identical to a plain bundle, while sorting at
    the caller keeps pair identity deterministic for unordered row pairs.
    """
    bits = max(1, int(bits))
    salt = int(salt) & 0xFFFFFFFFFFFFFFFF
    r1 = 1 + int(_splitmix64(salt ^ 0xA24BAED4963EE407) % max(1, bits - 1))
    r2 = 1 + int(_splitmix64(salt ^ 0x9FB21C651E98DF25) % max(1, bits - 1))
    return int((_rotl(a, r1, bits) ^ _rotl(b, r2, bits) ^ _base_scramble(salt, bits)) & _mask(bits))


def _base_scramble(seed: int, bits: int) -> int:
    out = 0
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    for word in range((int(bits) + 63) // 64):
        out |= int(_splitmix64(seed + word * 0xD1B54A32D192ED03)) << (word * 64)
    return int(out) & _mask(bits)

def _tok_class(tok: str) -> str:
    t = str(tok)
    if t in {'[NL]', '[INDENT]', '[DEDENT]', '[EOS]', '[BOS]', '[SEP]'}:
        return t
    if t.isdigit():
        return 'NUM'
    if len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")):
        return 'STR'
    if t and (t[0].isalpha() or t[0] == '_'):
        return 'WORD'
    if t in {'(', ')', '[', ']', '{', '}'}:
        return 'BRACKET'
    if t in {':', ',', ';', '.'}:
        return 'PUNCT'
    if t in {'+', '-', '*', '/', '//', '%', '==', '!=', '<', '>', '<=', '>=', '=', '+=', '-=', '*=', '/='}:
        return 'OP'
    return 'SYM'


def _mask(bits: int) -> int:
    return (1 << int(bits)) - 1 if bits > 0 else 0


class SparseEvidenceTables:
    def __init__(
        self,
        use_rust: bool = True,
        readout_enabled: bool = True,
        readout_hidden_dim: int = READOUT_DEFAULT_HIDDEN,
        readout_vocab_cap: int = READOUT_DEFAULT_VOCAB_CAP,
        readout_lr: float = READOUT_DEFAULT_LR,
        readout_auto_resize: bool = True,
        readout_min_hidden: int = READOUT_MIN_HIDDEN,
        readout_max_hidden: int = 0,
        readout_min_vocab_cap: int = READOUT_MIN_VOCAB_CAP,
        readout_max_vocab_cap: int = 0,
        readout_active_features: int = READOUT_ACTIVE_FEATURES,
    ) -> None:
        self.readout_enabled = bool(readout_enabled)
        self.readout_auto_resize = bool(readout_auto_resize)
        self.readout_min_hidden = max(1, int(readout_min_hidden))
        self.readout_max_hidden = max(0, int(readout_max_hidden))
        self.readout_min_vocab_cap = max(1, int(readout_min_vocab_cap))
        self.readout_max_vocab_cap = max(0, int(readout_max_vocab_cap))
        self.readout_hidden_dim = max(0, int(readout_hidden_dim))
        self.readout_vocab_cap = max(0, int(readout_vocab_cap))
        self.readout_lr = float(readout_lr)
        self.readout_active_features = max(0, int(readout_active_features))
        self.readout_resize_count = 0
        # P74 speed path: generation/readout/HDC projections are pure between
        # updates, so cache them during inference and recursive lookahead.
        # Any learning update invalidates these caches.
        self.runtime_cache_enabled = True
        self.runtime_cache_max_entries = 0
        self._score_cache: OrderedDict[Any, Dict[str, float]] = OrderedDict()
        self._hdc_context_cache: OrderedDict[Any, List[Tuple[int, int, float, int]]] = OrderedDict()
        self._readout_projection_cache: OrderedDict[Any, Dict[int, float]] = OrderedDict()
        self.score_cache_hits = 0
        self.score_cache_misses = 0
        self.hdc_context_cache_hits = 0
        self.hdc_context_cache_misses = 0
        self.readout_projection_cache_hits = 0
        self.readout_projection_cache_misses = 0
        self.fast_score_calls = 0
        self.full_score_calls = 0
        self.rust_fast_path_calls = 0
        # Token -> sparse hidden-index weights. This is the only intentional
        # trainable dense-like state in P73. It is kept sparse/online so storage
        # can be measured exactly.  An inverted hidden-index -> token map makes
        # scoring O(active_hidden * posting_list) instead of O(vocab * hidden).
        self.readout_weights: Dict[str, Dict[int, float]] = defaultdict(dict)
        self.readout_index: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.readout_token_updates: Counter[str] = Counter()
        self.readout_updates = 0
        self.readout_last_active = 0
        self.readout_last_candidates = 0
        self.feature_next: Dict[str, Counter[str]] = defaultdict(Counter)
        self.feature_totals: Counter[str] = Counter()

        # P69 keeps discriminative, loss-routed evidence without adding dense W.
        # These are still addressable sparse counters, not trainable matrices:
        #   feature_next  = tokens that a row should support
        #   feature_wrong = tokens that outranked the gold token for that row
        self.feature_wrong: Dict[str, Counter[str]] = defaultdict(Counter)
        self.feature_wrong_totals: Counter[str] = Counter()
        self.vocab: Counter[str] = Counter()

        # HDC rows are keyed by (bits_used_when_stored, hv_int).
        self.hdc_next: Dict[Tuple[int, int], Counter[str]] = defaultdict(Counter)
        self.hdc_totals: Counter[Tuple[int, int]] = Counter()
        self.hdc_wrong: Dict[Tuple[int, int], Counter[str]] = defaultdict(Counter)
        self.hdc_wrong_totals: Counter[Tuple[int, int]] = Counter()
        self.hdc_buckets: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)

        self.feature_hv: Dict[str, int] = {}
        self.hdc_bits = dynamic_hdc_bits(0, 0, 0, 0)
        self.updates = 0
        self.resize_count = 0
        self.rust = None
        self.rust_available = False
        # P76 fast-only: always try to load Rust/native support.  The use_rust
        # argument is accepted for compatibility but no longer disables the fast
        # default path.
        use_rust = True
        if use_rust:
            try:
                from . import puhl_luck_core  # type: ignore
                cls = getattr(puhl_luck_core, 'RustHebbianHdcCountEvidence', None)
                if cls is not None:
                    self.rust = cls()
                    self.rust_available = True
                    self._sync_rust_policy()
            except Exception:
                self.rust = None
                self.rust_available = False
        self._maybe_resize_readout(force=True)

    def _sync_rust_policy(self) -> None:
        if self.rust is None:
            return
        try:
            if hasattr(self.rust, 'configure_dynamic'):
                self.rust.configure_dynamic(len(self.feature_next), int(self.updates), len(self.hdc_next), len(self.vocab))
        except Exception:
            self.rust = None
            self.rust_available = False

    def _cache_put(self, cache: OrderedDict, key: Any, value: Any) -> None:
        if not self.runtime_cache_enabled:
            return
        cache[key] = value
        cache.move_to_end(key)
        limit = int(self.runtime_cache_max_entries or dynamic_cache_size(int(self.updates), len(self.feature_next) + len(self.feature_hv), len(self.vocab), len(self.hdc_next)))
        while len(cache) > max(1, limit):
            cache.popitem(last=False)

    def _clear_runtime_caches(self) -> None:
        self._score_cache.clear()
        self._hdc_context_cache.clear()
        self._readout_projection_cache.clear()

    def _rows_cache_key(self, rows: List[Tuple[str, float]]) -> Tuple[Tuple[str, int], ...]:
        # Store quantized row weights so harmless float noise does not destroy
        # cache locality. Generation uses stable learned gates, not gradients.
        return tuple((str(f), int(round(float(w) * 1_000_000))) for f, w in rows)

    def _rank_top(self, scores: Dict[str, float], top_k: int) -> List[Tuple[str, float]]:
        if not scores:
            return []
        k = max(1, int(top_k))
        if len(scores) <= k * 4:
            items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        else:
            # Avoid full O(V log V) sort on large adaptive vocab/readout maps.
            items = heapq.nsmallest(k, scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(str(t), float(s)) for t, s in items]

    def _target_bits(self) -> int:
        return dynamic_hdc_bits(len(self.feature_next) + len(self.feature_hv), int(self.updates), len(self.hdc_next), len(self.vocab))

    def _maybe_resize_hdc(self) -> None:
        target = self._target_bits()
        if target <= self.hdc_bits:
            return
        # Avoid rebuilding buckets on every few events.  Grow geometrically; old
        # rows remain valid because each HDC row stores its own bit width.
        if target < int(self.hdc_bits * 1.25):
            return
        self.hdc_bits = int(target)
        self.resize_count += 1
        self._clear_runtime_caches()
        self._rebuild_buckets()
        self._sync_rust_policy()

    def _rows_from_features(self, features: Iterable[Tuple[str, float]] | Iterable[str]) -> List[Tuple[str, float]]:
        rows: List[Tuple[str, float]] = []
        seen = set()
        for item in features:
            if isinstance(item, tuple):
                f = str(item[0]); w = float(item[1])
            else:
                f = str(item); w = 1.0
            if not f or f in seen or w <= 0.0:
                continue
            seen.add(f)
            rows.append((f, w))
        return rows

    def _base_hv(self, f: str, bits: int | None = None) -> int:
        bits = int(bits or self.hdc_bits)
        seed = _fnv1a64(f)
        hv = 0
        for word in range((bits + 63) // 64):
            x = _splitmix64(seed + (word * 0xD1B54A32D192ED03))
            hv |= int(x) << (word * 64)
        return hv & _mask(bits)

    def _expand_hv(self, f: str, hv: int, bits: int | None = None) -> int:
        bits = int(bits or self.hdc_bits)
        cur_bits = max(1, int(hv).bit_length())
        # If the highest learned bit is lower than the current dimensionality,
        # append deterministic base bits for unseen dimensions. Learned lower bits
        # are preserved.
        if cur_bits >= bits:
            return int(hv) & _mask(bits)
        base = self._base_hv(f, bits)
        low_mask = _mask(cur_bits)
        return int((int(hv) & low_mask) | (base & ~low_mask)) & _mask(bits)

    def _get_hv(self, f: str, bits: int | None = None) -> int:
        bits = int(bits or self.hdc_bits)
        hv = self.feature_hv.get(f)
        if hv is None:
            return self._base_hv(f, bits)
        return self._expand_hv(f, int(hv), bits)

    def _ensure_hv(self, f: str) -> int:
        hv = self.feature_hv.get(f)
        if hv is None:
            hv = self._base_hv(f, self.hdc_bits)
        else:
            hv = self._expand_hv(f, int(hv), self.hdc_bits)
        self.feature_hv[f] = int(hv)
        return int(hv)

    def _pull_bits(self, src: int, dst: int, seed: int, bits_to_pull: int) -> int:
        bits = self.hdc_bits
        src = int(src) & _mask(bits)
        dst = int(dst) & _mask(bits)
        diff = int(src ^ dst)
        if diff == 0 or bits_to_pull <= 0:
            return int(src)
        want = min(int(bits_to_pull), int(diff.bit_count()))
        mask = 0
        x = int(seed) & 0xFFFFFFFFFFFFFFFF
        picked = 0
        tries = 0
        while picked < want and tries < bits * 4:
            x = _splitmix64(x + tries + 0x9E3779B97F4A7C15)
            pos = int(x % bits)
            bit = 1 << pos
            if (diff & bit) and not (mask & bit):
                mask |= bit
                picked += 1
            tries += 1
        if mask == 0:
            mask = diff & -diff
        return int((src & ~mask) | (dst & mask)) & _mask(bits)

    def _hdc_vector(self, rows: List[Tuple[str, float]], bits: int | None = None) -> Tuple[int, int]:
        bits = int(bits or self.hdc_bits)
        if not rows:
            return bits, 0
        acc = [0.0] * bits
        for f, w in rows:
            if w <= 0.0:
                continue
            hv = self._get_hv(f, bits)
            for i in range(bits):
                acc[i] += w if ((hv >> i) & 1) else -w
        out = 0
        for i, v in enumerate(acc):
            if v >= 0.0:
                out |= (1 << i)
        return bits, int(out) & _mask(bits)

    def _hdc_context_vectors(self, rows: List[Tuple[str, float]], bits: int | None = None) -> List[Tuple[int, int, float, int]]:
        """Return rowless HDC bundle/bind query vectors for the active context.

        The first vector is the ordinary superposed bundle.  Additional vectors
        are bound composites made directly in HDC space.  They are not emitted as
        feature ids, so unseen combinations can still land near related memories
        by Hamming similarity instead of requiring a new exact dictionary row.

        The closure is bounded by a data-scale vector budget, not by a fixed
        symbolic order like D2.  New composite vectors can be composed again until
        the per-forward vector budget is exhausted.
        """
        bits = int(bits or self.hdc_bits)
        if not rows:
            return []
        cache_key = None
        if self.runtime_cache_enabled:
            cache_key = ('hdcctx', int(bits), self._rows_cache_key(rows), int(self.updates), int(self.resize_count))
            cached = self._hdc_context_cache.get(cache_key)
            if cached is not None:
                self.hdc_context_cache_hits += 1
                self._hdc_context_cache.move_to_end(cache_key)
                return list(cached)
            self.hdc_context_cache_misses += 1
        bundle_bits, bundle_hv = self._hdc_vector(rows, bits)
        vectors: List[Tuple[int, int, float, int]] = []
        if bundle_hv:
            vectors.append((int(bundle_bits), int(bundle_hv), 1.0, 1))

        scored_sources: List[Tuple[float, str, float, int]] = []
        for f, w in rows:
            rw = max(0.0, float(w))
            if rw <= 0.0:
                continue
            # Evidence affects which atoms enter the runtime projection budget;
            # it does not create a manual template-weight ladder.
            support = float(self.feature_totals.get(str(f), 0) or 0)
            wrong = float(self.feature_wrong_totals.get(str(f), 0) or 0)
            confidence = (support + 1.0) / (support + wrong + 1.0) if (support + wrong) > 0 else 1.0
            score = rw * confidence * math.log1p(support + wrong + 1.0)
            scored_sources.append((score, str(f), rw, 1))
        scored_sources.sort(key=lambda x: (-x[0], x[1]))
        if len(scored_sources) < 2:
            if cache_key is not None:
                self._cache_put(self._hdc_context_cache, cache_key, list(vectors))
            return vectors

        # Vector budget grows with observed HDC rows.  It limits work per forward
        # pass but does not impose a D2/D3 semantic ceiling.
        source_budget = dynamic_hdc_source_budget(
            len(self.feature_next) + len(self.feature_hv),
            len(self.hdc_next),
            int(self.updates),
            len(scored_sources),
        )
        atom_rows = scored_sources[: min(len(scored_sources), source_budget)]
        pool: List[Tuple[int, int, float, int, str]] = []
        for _score, f, rw, depth in atom_rows:
            hv = self._get_hv(f, bits)
            pool.append((int(bits), int(hv), float(rw), int(depth), f))

        vector_budget = dynamic_hdc_source_budget(
            len(self.feature_next) + len(self.feature_hv),
            len(self.hdc_next),
            int(self.updates),
            len(pool),
        )
        seen = {(int(bundle_bits), int(bundle_hv))} if bundle_hv else set()
        made = 0
        start = 0
        while start < len(pool) and made < vector_budget:
            current_len = len(pool)
            added_this_round = 0
            round_budget = dynamic_hdc_source_budget(len(self.feature_next) + len(self.feature_hv), len(self.hdc_next), int(self.updates), max(1, vector_budget - made))
            for i in range(start, current_len):
                bi, hi, wi, di, ni = pool[i]
                for j in range(0, i):
                    bj, hj, wj, dj, nj = pool[j]
                    if bi != bj:
                        continue
                    # Deterministic unordered pair binding; the row names are
                    # used only as seeds, never as new feature table ids.
                    a_name, b_name = (ni, nj) if ni <= nj else (nj, ni)
                    salt = _fnv1a64(f'HDCBIND|{a_name}|{b_name}|{max(di, dj) + 1}')
                    hv = _bind_hv(hi if ni <= nj else hj, hj if ni <= nj else hi, bi, salt=salt)
                    key = (int(bi), int(hv))
                    if key in seen:
                        continue
                    seen.add(key)
                    depth = max(int(di), int(dj)) + 1
                    qweight = math.sqrt(max(0.0, wi) * max(0.0, wj))
                    vectors.append((int(bi), int(hv), float(qweight), int(depth)))
                    pool.append((int(bi), int(hv), float(qweight), int(depth), f'@B{depth}:{_fnv1a64(a_name + b_name):016x}'))
                    made += 1
                    added_this_round += 1
                    if made >= vector_budget or added_this_round >= round_budget:
                        break
                if made >= vector_budget or added_this_round >= round_budget:
                    break
            if added_this_round <= 0:
                break
            start = current_len
        if cache_key is not None:
            self._cache_put(self._hdc_context_cache, cache_key, list(vectors))
        return vectors

    def _bucket_keys_for(self, bits: int, hv: int) -> List[Tuple[int, int]]:
        band_bits = dynamic_hdc_band_bits(int(bits), len(self.hdc_next), int(self.updates))
        bands = (int(bits) + band_bits - 1) // band_bits
        out: List[Tuple[int, int]] = []
        for band in range(bands):
            shift = band * band_bits
            seg = int((hv >> shift) & ((1 << band_bits) - 1))
            out.append((band, seg))
        return out

    def _bucket_keys(self, hv_key: Tuple[int, int]) -> List[Tuple[int, int]]:
        return self._bucket_keys_for(int(hv_key[0]), int(hv_key[1]))

    def _rebuild_buckets(self) -> None:
        self.hdc_buckets = defaultdict(set)
        for hv_key in self.hdc_next.keys():
            for key in self._bucket_keys(hv_key):
                self.hdc_buckets[key].add(hv_key)

    def _dynamic_top_neighbors(self) -> int:
        return int(dynamic_hdc_neighbors(len(self.hdc_next), int(self.updates), len(self.vocab)))

    def _push_bits(self, src: int, away_from: int, seed: int, bits_to_push: int) -> int:
        bits = self.hdc_bits
        src = int(src) & _mask(bits)
        dst = int(away_from) & _mask(bits)
        same = int(~(src ^ dst)) & _mask(bits)
        if same == 0 or bits_to_push <= 0:
            return int(src)
        want = min(int(bits_to_push), int(same.bit_count()))
        mask = 0
        x = int(seed) & 0xFFFFFFFFFFFFFFFF
        picked = 0
        tries = 0
        while picked < want and tries < bits * 4:
            x = _splitmix64(x + tries + 0x517CC1B727220A95)
            pos = int(x % bits)
            bit = 1 << pos
            if (same & bit) and not (mask & bit):
                mask |= bit
                picked += 1
            tries += 1
        if mask == 0:
            mask = same & -same
        return int(src ^ mask) & _mask(bits)

    def _hebbian_update(self, rows: List[Tuple[str, float]], next_token: str, amount: int = 1, negative_tokens: List[str] | None = None) -> None:
        if not rows:
            return
        selected = sorted(rows, key=lambda x: -x[1])[:dynamic_hebbian_rows(len(rows), int(self.updates), len(self.feature_next))]
        if not selected:
            return
        _bits, ctx_hv = self._hdc_vector(selected, self.hdc_bits)
        y_key = 'Y|' + str(next_token)
        yc_key = 'YC|' + _tok_class(next_token)
        y_hv = self._ensure_hv(y_key)
        yc_hv = self._ensure_hv(yc_key)

        # Aggregate rank-loss magnitude into one vector update.  The old loop
        # repeated the same Hebbian pull amount times, which made CPU cost grow
        # with rank error instead of with data structure size.  Credit strength
        # still enters through dynamic_pull_bits(amount, ...).
        r = 0
        seed_y = _fnv1a64(f'{y_key}|CTX|{self.updates}|{r}')
        y_hv = self._pull_bits(y_hv, ctx_hv, seed_y, dynamic_pull_bits(self.hdc_bits, amount, 1.0))
        y_hv = self._pull_bits(y_hv, yc_hv, seed_y ^ 0xA5A5A5A5A5A5A5A5, dynamic_pull_bits(self.hdc_bits, amount, 0.5))
        self.feature_hv[y_key] = y_hv
        for f, w in selected:
            if w <= 0.04:
                continue
            fhv = self._ensure_hv(f)
            pull = dynamic_pull_bits(self.hdc_bits, amount, w)
            seed_f = _fnv1a64(f'{f}|{y_key}|{self.updates}|{r}')
            fhv = self._pull_bits(fhv, y_hv, seed_f, pull)
            if w >= 0.35:
                fhv = self._pull_bits(fhv, yc_hv, seed_f ^ 0xD1B54A32D192ED03, dynamic_pull_bits(self.hdc_bits, amount, 0.5 * w))
            self.feature_hv[f] = fhv

        # Error-routed anti-Hebbian credit: wrong tokens that beat the gold token
        # are pushed away from this context.  The number of losers comes from the
        # current loss/rank, not from a hand-set feature strength ladder.
        for bad_tok in (negative_tokens or []):
            bad_key = 'Y|' + str(bad_tok)
            bad_hv = self._ensure_hv(bad_key)
            r = 0
            seed_bad = _fnv1a64(f'{bad_key}|ANTI|{self.updates}|{r}')
            bad_hv = self._push_bits(bad_hv, ctx_hv, seed_bad, dynamic_pull_bits(self.hdc_bits, amount, 1.0))
            self.feature_hv[bad_key] = bad_hv

    def _training_hdc_context_vectors(self, rows: List[Tuple[str, float]]) -> List[Tuple[int, int, float, int]]:
        """HDC rows materialized during learning.

        The rowless HDC binding/composite path is still used by the adaptive
        readout, but learning no longer stores every transient composite vector
        as a permanent hdc_next/hdc_wrong row.  Permanent HDC evidence is the
        canonical bundle row plus any HDC row that already has learned evidence.
        This is a data/evidence rule, not a fixed D2/D3 cap.
        """
        vectors = self._hdc_context_vectors(rows, self.hdc_bits)
        out: List[Tuple[int, int, float, int]] = []
        seen = set()
        for bits, hv, qweight, depth in vectors:
            if not hv:
                continue
            key = (int(bits), int(hv))
            if int(depth) <= 1 or key in self.hdc_next or key in self.hdc_wrong:
                if key not in seen:
                    out.append((int(bits), int(hv), float(qweight), int(depth)))
                    seen.add(key)
        return out

    def _update_negative_python_rows(self, rows: List[Tuple[str, float]], wrong_tokens: List[str], amount: int = 1) -> None:
        if not rows or not wrong_tokens:
            return
        amount = max(1, int(amount))
        # Negative credit is routed only to rows that actually supported the
        # outranking wrong token.  The previous implementation wrote every wrong
        # token into every active row, causing rows * losers storage explosion.
        for f, _w in rows:
            row = self.feature_next.get(str(f))
            wrong = self.feature_wrong.get(str(f))
            local_bad = [tok for tok in wrong_tokens if (row and row.get(tok, 0) > 0) or (wrong and wrong.get(tok, 0) > 0)]
            for tok in local_bad:
                self.feature_wrong[f][tok] += amount
                self.feature_wrong_totals[f] += amount
        for bits, hv, _qweight, _depth in self._training_hdc_context_vectors(rows):
            key = (int(bits), int(hv))
            row = self.hdc_next.get(key)
            wrong = self.hdc_wrong.get(key)
            local_bad = [tok for tok in wrong_tokens if (row and row.get(tok, 0) > 0) or (wrong and wrong.get(tok, 0) > 0)]
            for tok in local_bad:
                self.hdc_wrong[key][tok] += amount
                self.hdc_wrong_totals[key] += amount
                for bkey in self._bucket_keys(key):
                    self.hdc_buckets[bkey].add(key)

    def _update_python_rows(self, rows: List[Tuple[str, float]], next_token: str, amount: int = 1, negative_tokens: List[str] | None = None) -> None:
        amount = max(1, int(amount))
        self._maybe_resize_hdc()
        negative_tokens = [str(t) for t in (negative_tokens or []) if str(t) != str(next_token)]
        self._hebbian_update(rows, next_token, amount=amount, negative_tokens=negative_tokens)
        if negative_tokens:
            self._update_negative_python_rows(rows, negative_tokens, amount=1)

        for f, _w in rows:
            self.feature_next[f][next_token] += amount
            self.feature_totals[f] += amount
        for bits, hv, _qweight, _depth in self._training_hdc_context_vectors(rows):
            key = (int(bits), int(hv))
            self.hdc_next[key][next_token] += amount
            self.hdc_totals[key] += amount
            for bkey in self._bucket_keys(key):
                self.hdc_buckets[bkey].add(key)
        self.vocab[next_token] += amount
        self.updates += amount
        self._clear_runtime_caches()
        self._maybe_resize_hdc()
        
        # Task 9.1: Periodically prune zero-count entries to maintain sparse storage (Requirement 14.2)
        # Prune every 1000 updates to balance memory efficiency with update performance
        if self.updates % 1000 == 0:
            self._prune_zero_counts()

    def update_features(self, features: Iterable[Tuple[str, float]] | Iterable[str], next_token: str, amount: int = 1) -> None:
        rows = self._rows_from_features(features)
        if not rows:
            return
        next_token = str(next_token)
        amount = max(1, int(amount))
        self._update_python_rows(rows, next_token, amount=amount)
        if self.rust is not None:
            try:
                self._sync_rust_policy()
                if hasattr(self.rust, 'update_features_weighted'):
                    self.rust.update_features_weighted(rows, next_token, amount)
                else:
                    raise AttributeError('missing RustHebbianHdcCountEvidence.update_features_weighted')
            except Exception:
                self.rust = None
                self.rust_available = False

    def update_many(self, batch: Iterable[Tuple[Any, str] | Tuple[Any, str, int]]) -> int:
        rust_batch = []
        n = 0
        for item in batch:
            if len(item) == 3:  # type: ignore[arg-type]
                features, tok, amount = item  # type: ignore[misc]
            else:
                features, tok = item  # type: ignore[misc]
                amount = 1
            rows = self._rows_from_features(features)
            if not rows:
                continue
            tok = str(tok)
            amount = max(1, int(amount))
            self._update_python_rows(rows, tok, amount=amount)
            rust_batch.append((rows, tok, amount))
            n += amount
        if self.rust is not None and rust_batch:
            try:
                self._sync_rust_policy()
                if hasattr(self.rust, 'update_many_weighted'):
                    self.rust.update_many_weighted(rust_batch)
                else:
                    raise AttributeError('missing RustHebbianHdcCountEvidence.update_many_weighted')
            except Exception:
                self.rust = None
                self.rust_available = False
        return n

    def _row_reliability(self, row: Counter[str], total: float) -> float:
        """Data-derived row confidence, not a hand tuned feature weight.

        A row becomes strong only when it has both support and a concentrated
        next-token distribution.  This replaces the old manual feature-weight
        ladder in the live generation path.
        """
        total = float(total or sum(row.values()) or 1.0)
        concentration = 0.0
        for cnt in row.values():
            p = float(cnt) / total
            concentration += p * p
        return math.log1p(total) * concentration

    def _target_readout_shape(self) -> Tuple[int, int]:
        return dynamic_readout_shape(
            len(self.feature_next) + len(self.feature_hv),
            int(self.updates + self.readout_updates),
            len(self.hdc_next),
            len(self.vocab) + len(self.readout_weights),
            min_hidden=self.readout_min_hidden,
            max_hidden=self.readout_max_hidden,
            min_vocab_cap=self.readout_min_vocab_cap,
            max_vocab_cap=self.readout_max_vocab_cap,
        )

    def _maybe_resize_readout(self, force: bool = False) -> None:
        if not self.readout_enabled:
            return
        if not self.readout_auto_resize and self.readout_hidden_dim > 0 and self.readout_vocab_cap > 0:
            return
        target_hidden, target_vocab = self._target_readout_shape()
        changed = False
        if self.readout_hidden_dim <= 0 or target_hidden > self.readout_hidden_dim or force:
            new_hidden = max(target_hidden, self.readout_hidden_dim, self.readout_min_hidden)
            if self.readout_max_hidden > 0:
                new_hidden = min(new_hidden, self.readout_max_hidden)
            if new_hidden != self.readout_hidden_dim:
                self.readout_hidden_dim = int(new_hidden)
                changed = True
        if self.readout_vocab_cap <= 0 or target_vocab > self.readout_vocab_cap or force:
            new_cap = max(target_vocab, self.readout_vocab_cap, self.readout_min_vocab_cap)
            if self.readout_max_vocab_cap > 0:
                new_cap = min(new_cap, self.readout_max_vocab_cap)
            if new_cap != self.readout_vocab_cap:
                self.readout_vocab_cap = int(new_cap)
                changed = True
        
        # Integrate dynamic_readout_config for adaptive learning rate (Task 3.5)
        # Update learning rate based on current data scale when readout_lr is not explicitly set (0)
        if self.readout_lr <= 0 and self.readout_auto_resize:
            config = dynamic_readout_config(
                vocab_size=len(self.vocab) + len(self.readout_weights),
                feature_count=len(self.feature_next) + len(self.feature_hv),
                event_count=int(self.updates + self.readout_updates)
            )
            # Use the adaptive learning rate from dynamic_readout_config
            self.readout_lr = config['learning_rate']
            changed = True
        
        if changed:
            self.readout_resize_count += 1
            self._clear_runtime_caches()

    def _readout_active_budget(self) -> int:
        if not self.readout_enabled or self.readout_hidden_dim <= 0:
            return 0
        if self.readout_active_features > 0:
            return min(int(self.readout_active_features), int(self.readout_hidden_dim))
        return int(dynamic_readout_active_budget(int(self.readout_hidden_dim), int(self.updates), len(self.vocab), len(self.hdc_next)))

    def _drop_readout_token(self, token: str) -> None:
        token = str(token)
        wdict = self.readout_weights.pop(token, None)
        if wdict:
            for i in list(wdict.keys()):
                posting = self.readout_index.get(int(i))
                if posting is not None:
                    posting.pop(token, None)
                    if not posting:
                        self.readout_index.pop(int(i), None)
        self.readout_token_updates.pop(token, None)

    def _set_readout_weight(self, token: str, idx: int, value: float) -> None:
        token = str(token)
        idx = int(idx)
        if abs(float(value)) <= 1e-10:
            if token in self.readout_weights:
                self.readout_weights[token].pop(idx, None)
            if idx in self.readout_index:
                self.readout_index[idx].pop(token, None)
                if not self.readout_index[idx]:
                    self.readout_index.pop(idx, None)
            return
        self.readout_weights[token][idx] = float(value)
        self.readout_index[idx][token] = float(value)

    def _rebuild_readout_index(self) -> None:
        self.readout_index = defaultdict(dict)
        for tok, wdict in self.readout_weights.items():
            for i, v in wdict.items():
                if abs(float(v)) > 1e-10:
                    self.readout_index[int(i)][str(tok)] = float(v)

    def _readout_accept_token(self, token: str) -> bool:
        """Track a bounded token set for the aggressive readout.

        The cap is a storage budget, not a grammar rule.  Once full, a new token
        replaces only the least-updated token.  This keeps the P73 readout small
        enough for honest dense-baseline comparisons.
        """
        token = str(token)
        if not self.readout_enabled:
            return False
        self._maybe_resize_readout()
        if self.readout_hidden_dim <= 0:
            return False
        if token in self.readout_weights:
            return True
        if self.readout_vocab_cap <= 0 or len(self.readout_weights) < self.readout_vocab_cap:
            self.readout_weights[token] = {}
            return True
        if not self.readout_weights:
            return False
        weakest = min(self.readout_weights.keys(), key=lambda t: (self.readout_token_updates.get(t, 0), len(self.readout_weights.get(t, {})), t))
        # Replacement is competitive: keep heavily trained tokens, but let new
        # data take over dead/rare slots when the stream changes.
        if self.readout_token_updates.get(weakest, 0) <= dynamic_hdc_neighbors(len(self.readout_weights), int(self.readout_updates), len(self.vocab)):
            self._drop_readout_token(weakest)
            self.readout_weights[token] = {}
            return True
        return False

    def _readout_features_from_vectors(self, q_vectors: List[Tuple[int, int, float, int]]) -> Dict[int, float]:
        """Project HDC bind/bundle vectors into a small fixed random hidden layer.

        The hidden projection itself is deterministic and not trained.  Only the
        output token margin weights are trained online.  Features are sparse and
        signed so update cost is O(active_hidden * outranking_wrong_tokens), not
        O(D * V).
        """
        if not self.readout_enabled or not q_vectors:
            return {}
        self._maybe_resize_readout()
        if self.readout_hidden_dim <= 0:
            return {}
        hidden = self.readout_hidden_dim
        active_budget = self._readout_active_budget()
        cache_key = None
        if self.runtime_cache_enabled:
            # Readout projection depends only on hidden size, active budget, and
            # query vectors. It does not depend on token output weights.
            cache_key = (
                'roproj', int(hidden), int(active_budget),
                tuple((int(b), int(h), int(round(float(w) * 1_000_000)), int(d)) for b, h, w, d in q_vectors),
            )
            cached = self._readout_projection_cache.get(cache_key)
            if cached is not None:
                self.readout_projection_cache_hits += 1
                self._readout_projection_cache.move_to_end(cache_key)
                self.readout_last_active = len(cached)
                return dict(cached)
            self.readout_projection_cache_misses += 1
        scores: Dict[int, float] = defaultdict(float)
        # Use strongest context/composite vectors first. Runtime budget grows
        # with data/readout size, not a fixed 12-vector wall.
        vector_budget = min(len(q_vectors), dynamic_hdc_source_budget(len(self.feature_next) + len(self.feature_hv), len(self.hdc_next), int(self.updates), max(1, active_budget)))
        ordered = sorted(q_vectors, key=lambda x: (-float(x[2]) / math.sqrt(max(1.0, float(x[3]))), int(x[3])))[:vector_budget]
        for bits, hv, qweight, depth in ordered:
            bits = int(bits)
            hv = int(hv) & _mask(bits)
            if bits <= 0 or hv == 0:
                continue
            depth_gain = 1.0 / math.sqrt(max(1.0, float(depth)))
            base_gain = max(0.0, float(qweight)) * depth_gain
            if base_gain <= 0.0:
                continue
            seed = _fnv1a64(f'P73READOUT|{bits}|{hv & 0xFFFFFFFFFFFF}|{depth}')
            per_vector = max(1, active_budget // max(1, len(ordered)))
            for k in range(per_vector):
                x = _splitmix64(seed + k * 0x9E3779B97F4A7C15)
                idx = int(x % hidden)
                p1 = int((x >> 16) % bits)
                p2 = int((x >> 33) % bits)
                b = ((hv >> p1) ^ (hv >> p2) ^ (x >> 7)) & 1
                sign = 1.0 if b else -1.0
                scores[idx] += sign * base_gain
        if not scores:
            return {}
        norm = math.sqrt(sum(v * v for v in scores.values())) or 1.0
        items = sorted(scores.items(), key=lambda kv: -abs(kv[1]))[:active_budget]
        out = {int(i): float(v / norm) for i, v in items if abs(v) > 1e-12}
        self.readout_last_active = len(out)
        if cache_key is not None:
            self._cache_put(self._readout_projection_cache, cache_key, dict(out))
        return out

    def _readout_features(self, rows: List[Tuple[str, float]]) -> Dict[int, float]:
        return self._readout_features_from_vectors(self._hdc_context_vectors(rows, self.hdc_bits))

    def _score_readout_python(self, rows: List[Tuple[str, float]], scores: Dict[str, float]) -> int:
        if not self.readout_enabled or not self.readout_weights:
            self.readout_last_candidates = 0
            return 0
        x = self._readout_features(rows)
        if not x:
            self.readout_last_candidates = 0
            return 0
        acc: Dict[str, float] = defaultdict(float)
        for i, xv in x.items():
            posting = self.readout_index.get(int(i))
            if not posting:
                continue
            for tok, wv in posting.items():
                acc[str(tok)] += float(wv) * float(xv)
        used = 0
        for tok, dot in acc.items():
            if abs(float(dot)) > 1e-12:
                scores[str(tok)] += dynamic_readout_score_gain(int(self.updates), len(x), len(acc)) * float(dot)
                used += 1
        self.readout_last_candidates = used
        return used

    def _update_readout(self, rows: List[Tuple[str, float]], target_token: str, negative_tokens: List[str] | None = None, amount: int = 1) -> Dict[str, Any]:
        if not self.readout_enabled:
            return {'readout_update_used': False, 'reason': 'disabled'}
        self._maybe_resize_readout()
        if self.readout_hidden_dim <= 0:
            return {'readout_update_used': False, 'reason': 'disabled'}
        target_token = str(target_token)
        accepted = self._readout_accept_token(target_token)
        if not accepted:
            return {'readout_update_used': False, 'reason': 'vocab_budget'}
        x = self._readout_features(rows)
        if not x:
            return {'readout_update_used': False, 'reason': 'no_features'}
        negs: List[str] = []
        for tok in (negative_tokens or []):
            tok = str(tok)
            if tok != target_token and self._readout_accept_token(tok):
                negs.append(tok)
        lr = (float(self.readout_lr) if self.readout_lr > 0 else dynamic_learning_rate(int(self.readout_updates), len(x))) / math.sqrt(1.0 + self.readout_updates / max(1.0, float(len(self.readout_weights) + len(x))))
        pos_step = float(lr) * max(1.0, math.sqrt(float(max(1, int(amount)))))
        target_w = self.readout_weights[target_token]
        for i, xv in x.items():
            nv = float(target_w.get(i, 0.0)) + pos_step * float(xv)
            self._set_readout_weight(target_token, int(i), nv)
        neg_budget = dynamic_hdc_neighbors(len(negs), int(self.readout_updates), len(self.readout_weights))
        neg_step = float(lr) / math.sqrt(max(1.0, float(min(max(1, neg_budget), len(negs)))))
        for bad in negs[:neg_budget]:
            bad_w = self.readout_weights[bad]
            for i, xv in x.items():
                nv = float(bad_w.get(i, 0.0)) - neg_step * float(xv)
                self._set_readout_weight(bad, int(i), nv)
        self.readout_token_updates[target_token] += 1
        for bad in negs[:neg_budget]:
            self.readout_token_updates[bad] += 1
        self.readout_updates += 1
        if self.readout_updates % 512 == 0:
            self._prune_readout()
        self._score_cache.clear()
        return {
            'readout_update_used': True,
            'readout_active_features': int(len(x)),
            'readout_negative_tokens': int(min(neg_budget, len(negs))),
            'readout_lr': float(lr),
        }

    def _prune_readout(self) -> None:
        keep_base = dynamic_readout_active_budget(int(self.readout_hidden_dim), int(self.readout_updates), len(self.readout_weights), len(self.hdc_next))
        for tok, wdict in list(self.readout_weights.items()):
            if not wdict:
                continue
            max_keep = max(1, min(self.readout_hidden_dim, keep_base))
            if len(wdict) > max_keep:
                self.readout_weights[tok] = dict(sorted(wdict.items(), key=lambda kv: -abs(kv[1]))[:max_keep])
            else:
                for i in [i for i, v in wdict.items() if abs(float(v)) < 1e-7]:
                    wdict.pop(i, None)
        self._rebuild_readout_index()

    def _prune_zero_counts(self) -> None:
        """Prune zero-count entries from sparse tables to maintain memory efficiency.
        
        Task 9.1: Implements pruning of zero-count entries during updates (Requirement 14.2).
        This method removes tokens with zero counts from feature_next, hdc_next, feature_wrong,
        and hdc_wrong dictionaries, ensuring we only store non-zero counts as required by
        Requirement 14.2: "WHEN storing token distributions, THE Sparse_Table SHALL only 
        store non-zero counts".
        
        The pruning also removes empty feature rows and HDC rows to prevent memory bloat.
        
        Requirements: 14.1, 14.2
        """
        # Prune feature_next: remove zero-count tokens and empty features
        for f in list(self.feature_next.keys()):
            counter = self.feature_next[f]
            # Remove tokens with zero counts
            zero_tokens = [tok for tok, cnt in counter.items() if cnt <= 0]
            for tok in zero_tokens:
                del counter[tok]
            # Remove the feature entirely if it has no tokens
            if not counter:
                del self.feature_next[f]
                # Also clean up feature_totals if the feature is gone
                if f in self.feature_totals:
                    del self.feature_totals[f]
        
        # Prune feature_wrong: remove zero-count tokens and empty features
        for f in list(self.feature_wrong.keys()):
            counter = self.feature_wrong[f]
            zero_tokens = [tok for tok, cnt in counter.items() if cnt <= 0]
            for tok in zero_tokens:
                del counter[tok]
            if not counter:
                del self.feature_wrong[f]
                if f in self.feature_wrong_totals:
                    del self.feature_wrong_totals[f]
        
        # Prune hdc_next: remove zero-count tokens and empty HDC rows
        for key in list(self.hdc_next.keys()):
            counter = self.hdc_next[key]
            zero_tokens = [tok for tok, cnt in counter.items() if cnt <= 0]
            for tok in zero_tokens:
                del counter[tok]
            if not counter:
                del self.hdc_next[key]
                # Clean up HDC buckets and totals
                if key in self.hdc_totals:
                    del self.hdc_totals[key]
                # Remove from buckets
                for bkey in self._bucket_keys(key):
                    if bkey in self.hdc_buckets:
                        self.hdc_buckets[bkey].discard(key)
                        # Remove empty bucket sets
                        if not self.hdc_buckets[bkey]:
                            del self.hdc_buckets[bkey]
        
        # Prune hdc_wrong: remove zero-count tokens and empty HDC rows
        for key in list(self.hdc_wrong.keys()):
            counter = self.hdc_wrong[key]
            zero_tokens = [tok for tok, cnt in counter.items() if cnt <= 0]
            for tok in zero_tokens:
                del counter[tok]
            if not counter:
                del self.hdc_wrong[key]
                if key in self.hdc_wrong_totals:
                    del self.hdc_wrong_totals[key]
        
        # Prune vocab: remove zero-count tokens
        zero_vocab_tokens = [tok for tok, cnt in self.vocab.items() if cnt <= 0]
        for tok in zero_vocab_tokens:
            del self.vocab[tok]

    def get_memory_footprint(self) -> Dict[str, Any]:
        """Track memory footprint of sparse evidence tables.
        
        Task 9.1: Implements memory footprint tracking method (Requirement 14.4).
        Returns a comprehensive breakdown of memory usage across all sparse storage
        structures to help monitor and enforce the requirement that "FOR training sets 
        with 10,000+ pairs, THE memory footprint SHALL NOT exceed 500MB".
        
        The method estimates memory usage in bytes for all major data structures:
        - feature_next: sparse feature → token counter mappings
        - feature_wrong: negative evidence counters
        - hdc_next: HDC hypervector → token counter mappings
        - hdc_wrong: negative HDC evidence counters
        - vocab: token frequency counter
        - feature_hv: learned HDC hypervectors
        - hdc_buckets: LSH bucket index for HDC similarity search
        - readout_weights: adaptive readout sparse weights
        
        Returns:
            Dict with keys:
                - feature_next_bytes: memory for feature_next Counter dicts
                - feature_wrong_bytes: memory for feature_wrong Counter dicts
                - hdc_next_bytes: memory for hdc_next Counter dicts
                - hdc_wrong_bytes: memory for hdc_wrong Counter dicts
                - vocab_bytes: memory for vocab Counter
                - feature_hv_bytes: memory for feature hypervector storage
                - hdc_buckets_bytes: memory for HDC bucket index
                - readout_weights_bytes: memory for adaptive readout weights
                - total_bytes: total memory footprint
                - total_mb: total memory in megabytes
                - feature_count: number of features
                - hdc_row_count: number of HDC rows
                - vocab_size: vocabulary size
                - readout_parameter_count: sparse readout parameters
        
        Requirements: 14.1, 14.2, 14.4
        """
        import sys
        
        # Helper to estimate Counter memory (dict overhead + key/value pairs)
        def counter_memory(counter: Counter) -> int:
            # Dict object overhead: ~232 bytes
            # Each entry: key (str ~50 bytes avg) + value (int ~28 bytes) + dict entry overhead (~50 bytes)
            if not counter:
                return 0
            dict_overhead = 232
            avg_key_size = 50  # Average string key size estimate
            int_size = 28
            entry_overhead = 50
            return dict_overhead + len(counter) * (avg_key_size + int_size + entry_overhead)
        
        # Calculate feature_next memory (Dict[str, Counter[str]])
        feature_next_bytes = 0
        for f, counter in self.feature_next.items():
            # Feature key storage
            feature_next_bytes += sys.getsizeof(f)
            # Counter storage
            feature_next_bytes += counter_memory(counter)
        feature_next_bytes += 232  # Dict overhead for feature_next itself
        
        # Calculate feature_wrong memory
        feature_wrong_bytes = 0
        for f, counter in self.feature_wrong.items():
            feature_wrong_bytes += sys.getsizeof(f)
            feature_wrong_bytes += counter_memory(counter)
        feature_wrong_bytes += 232
        
        # Calculate hdc_next memory (Dict[Tuple[int, int], Counter[str]])
        hdc_next_bytes = 0
        for key, counter in self.hdc_next.items():
            # Tuple key: (bits, hv) - two ints
            hdc_next_bytes += sys.getsizeof(key)  # Tuple overhead
            hdc_next_bytes += 28 * 2  # Two integers
            hdc_next_bytes += counter_memory(counter)
        hdc_next_bytes += 232  # Dict overhead
        
        # Calculate hdc_wrong memory
        hdc_wrong_bytes = 0
        for key, counter in self.hdc_wrong.items():
            hdc_wrong_bytes += sys.getsizeof(key)
            hdc_wrong_bytes += 28 * 2
            hdc_wrong_bytes += counter_memory(counter)
        hdc_wrong_bytes += 232
        
        # Calculate vocab memory
        vocab_bytes = counter_memory(self.vocab)
        
        # Calculate feature_hv memory (Dict[str, int])
        feature_hv_bytes = 232  # Dict overhead
        for f, hv in self.feature_hv.items():
            feature_hv_bytes += sys.getsizeof(f)  # Feature key
            feature_hv_bytes += 28  # Integer hypervector
        
        # Calculate hdc_buckets memory (Dict[Tuple[int, int], Set[Tuple[int, int]]])
        hdc_buckets_bytes = 232  # Dict overhead
        for bkey, bucket_set in self.hdc_buckets.items():
            hdc_buckets_bytes += sys.getsizeof(bkey) + 28 * 2  # Tuple key
            hdc_buckets_bytes += 232  # Set overhead
            hdc_buckets_bytes += len(bucket_set) * (sys.getsizeof((0, 0)) + 28 * 2)  # Set entries
        
        # Calculate readout_weights memory (Dict[str, Dict[int, float]])
        readout_weights_bytes = 232  # Dict overhead
        for tok, wdict in self.readout_weights.items():
            readout_weights_bytes += sys.getsizeof(tok)  # Token key
            readout_weights_bytes += 232  # Inner dict overhead
            readout_weights_bytes += len(wdict) * (28 + 24 + 50)  # int key + float value + entry overhead
        
        # Calculate totals
        total_bytes = (
            feature_next_bytes +
            feature_wrong_bytes +
            hdc_next_bytes +
            hdc_wrong_bytes +
            vocab_bytes +
            feature_hv_bytes +
            hdc_buckets_bytes +
            readout_weights_bytes
        )
        total_mb = total_bytes / (1024 * 1024)
        
        return {
            'feature_next_bytes': int(feature_next_bytes),
            'feature_wrong_bytes': int(feature_wrong_bytes),
            'hdc_next_bytes': int(hdc_next_bytes),
            'hdc_wrong_bytes': int(hdc_wrong_bytes),
            'vocab_bytes': int(vocab_bytes),
            'feature_hv_bytes': int(feature_hv_bytes),
            'hdc_buckets_bytes': int(hdc_buckets_bytes),
            'readout_weights_bytes': int(readout_weights_bytes),
            'total_bytes': int(total_bytes),
            'total_mb': float(total_mb),
            'feature_count': len(self.feature_next),
            'hdc_row_count': len(self.hdc_next),
            'vocab_size': len(self.vocab),
            'readout_parameter_count': self.readout_parameter_count(),
        }

    def readout_parameter_count(self) -> int:
        if not self.readout_enabled:
            return 0
        return int(sum(len(w) for w in self.readout_weights.values()))

    def readout_dense_capacity(self) -> int:
        if not self.readout_enabled:
            return 0
        tracked = max(len(self.readout_weights), min(self.readout_vocab_cap, len(self.vocab)))
        if self.readout_vocab_cap > 0:
            tracked = min(self.readout_vocab_cap, max(tracked, len(self.readout_weights)))
        return int(self.readout_hidden_dim * max(0, tracked))

    def _score_exact_python(self, rows: List[Tuple[str, float]], scores: Dict[str, float]) -> None:
        for f, weight in rows:
            # P69: row weight is no longer a hand-written feature ladder.  It is
            # a data-derived template gate or discovered-feature gate supplied by
            # SparseLogitGenerator.  Exact sparse evidence must respect it so
            # credit can affect the feature generator itself.
            rw = max(0.0, float(weight))
            if rw <= 0.0:
                continue
            row = self.feature_next.get(f)
            if row:
                total = float(self.feature_totals.get(f, 0) or sum(row.values()) or 1)
                reliability = self._row_reliability(row, total)
                for tok, cnt in row.items():
                    scores[tok] += rw * reliability * (float(cnt) / total)
            wrong = self.feature_wrong.get(f)
            if wrong:
                wtotal = float(self.feature_wrong_totals.get(f, 0) or sum(wrong.values()) or 1)
                wrel = self._row_reliability(wrong, wtotal)
                for tok, cnt in wrong.items():
                    scores[tok] -= rw * wrel * (float(cnt) / wtotal)

    def _score_hdc_python(self, rows: List[Tuple[str, float]], scores: Dict[str, float]) -> int:
        if not rows or not self.hdc_next:
            return 0
        q_vectors = self._hdc_context_vectors(rows, self.hdc_bits)
        if not q_vectors:
            return 0
        used_total = 0
        for q_bits, q, qweight, qdepth in q_vectors:
            if not q:
                continue
            candidates: Set[Tuple[int, int]] = set()
            for key in self._bucket_keys_for(q_bits, q):
                candidates.update(self.hdc_buckets.get(key, ()))
            # Dynamic fallback only for small row sets. It is row-count based,
            # not a fixed HDC dimension or fixed symbolic order.
            if len(candidates) < dynamic_hdc_neighbors(len(self.hdc_next), int(self.updates), len(self.vocab)) and len(self.hdc_next) <= dynamic_cache_size(int(self.updates), len(self.feature_next), len(self.vocab), len(self.hdc_next)):
                candidates.update(self.hdc_next.keys())
            if not candidates:
                continue

            sims: List[Tuple[float, Tuple[int, int]]] = []
            for row_bits, hv in candidates:
                common_bits = max(1, min(int(row_bits), int(q_bits)))
                common_mask = _mask(common_bits)
                dist = ((int(q) & common_mask) ^ (int(hv) & common_mask)).bit_count()
                sim = 1.0 - (dist / float(common_bits))
                sims.append((sim, (int(row_bits), int(hv))))
            if not sims:
                continue

            sims.sort(key=lambda x: -x[0])
            vals = [x[0] for x in sims]
            # Query-local similarity floor: no fixed global HDC threshold.
            floor = vals[len(vals) // 2]
            top = vals[0]
            span = max(1e-9, top - floor)
            used = 0
            for sim, hv_key in sims[: self._dynamic_top_neighbors()]:
                if sim < floor:
                    continue
                row = self.hdc_next.get(hv_key)
                if not row:
                    continue
                total = float(self.hdc_totals.get(hv_key, 0) or sum(row.values()) or 1)
                reliability = self._row_reliability(row, total)
                sim_gain = 1.0 if span <= 1e-9 else ((sim - floor) / span)
                if sim_gain <= 0.0:
                    continue
                # Composite query vectors may be deeper bound/bundled contexts.
                # qweight is derived from learned row gates, not hand-written
                # context-length constants.
                depth_norm = 1.0 / math.sqrt(max(1.0, float(qdepth)))
                qgain = max(0.0, float(qweight)) * depth_norm
                if qgain <= 0.0:
                    continue
                for tok, cnt in row.items():
                    scores[tok] += qgain * reliability * sim_gain * (float(cnt) / total)
                wrong = self.hdc_wrong.get(hv_key)
                if wrong:
                    wtotal = float(self.hdc_wrong_totals.get(hv_key, 0) or sum(wrong.values()) or 1)
                    wrel = self._row_reliability(wrong, wtotal)
                    for tok, cnt in wrong.items():
                        scores[tok] -= qgain * wrel * sim_gain * (float(cnt) / wtotal)
                used += 1
            used_total += used
        return used_total

    def score_map_from_features(self, features: List[Tuple[str, float]]) -> Dict[str, float]:
        """P76 fast-only score map.

        Public scoring no longer runs the slow full HDC-KNN pass.  HDC context
        still enters through readout projection, but token ranking uses the fast
        exact-sparse + adaptive-readout path.
        """
        return self.score_map_from_features_fast(features)

    def credit_assign(self, features: Iterable[Tuple[str, float]] | Iterable[str], target_token: str, top_k: int = 0) -> Dict[str, Any]:
        """Loss/rank-routed credit assignment for a weightless sparse model.

        This is the P69 replacement for the previous fixed mistake boost.
        It does a forward pass, measures whether the gold token is supported, and
        sends credit back to every active symbolic row and the matching HDC row:

          * gold token gets positive count mass proportional to its current rank (logarithmic)
          * wrong tokens that outrank gold get negative sparse evidence
          * Hebbian vectors pull gold toward the context and push losers away

        No dense parameter matrix is created.  The trainable state is still only
        addressable counters plus HDC bit vectors.
        
        Algorithm (Task 3.1 - Enhanced credit assignment with rank-loss):
        1. Score all tokens given features
        2. Rank tokens by score
        3. Identify wrong-above tokens (ranked higher than target)
        4. Compute positive evidence amount using logarithmic formula: log2(target_rank + 2)
        5. Apply positive evidence to target token
        6. Apply negative evidence to wrong-above tokens
        7. Return diagnostic dict with positive_amount, negative_tokens, wrong_above_count, token_loss
        """
        rows = self._rows_from_features(features)
        if not rows:
            return {'credit_used': False, 'reason': 'no_rows'}
        target_token = str(target_token)
        
        # Step 1: Score all tokens given features
        # P76 learning default: use the same fast score path as generation so
        # training speed matches the CPU-fast runtime.
        scores = self.score_map_from_features_fast(rows)
        
        # Step 2: Rank all tokens by score to find target token's position
        # We need the full ranking to compute target_rank, not just top_k
        all_ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        target_score = float(scores.get(target_token, 0.0))
        
        # Find target token's rank in full ranking (0-indexed)
        target_rank = next((i for i, (tok, _) in enumerate(all_ranked) if tok == target_token), len(all_ranked))
        
        # Step 3: Identify wrong-above tokens (tokens ranked higher than target)
        wrong_above = [tok for tok, _ in all_ranked[:target_rank]]
        
        # Step 4: Compute positive evidence amount using logarithmic formula
        # amount = max(1, ceil(log2(target_rank + 2)))
        # This provides diminishing credit as rank improves, preventing overshooting
        positive_amount = max(1, int(math.ceil(math.log2(target_rank + 2))))
        
        # Compute token loss: rank(target) / total_tokens
        token_loss = float(target_rank) / max(1, len(all_ranked))
        
        # Compute probability and cross-entropy loss for top_k candidates
        rank_k = int(top_k) if int(top_k or 0) > 0 else dynamic_hdc_neighbors(len(scores), int(self.updates), len(self.vocab))
        ranked_topk = all_ranked[: max(1, rank_k)]
        vals = [float(s) for _t, s in ranked_topk]
        if target_token not in dict(ranked_topk):
            vals.append(target_score)
        if vals:
            mx = max(vals)
            denom = sum(math.exp(v - mx) for v in vals) or 1.0
            p_gold = math.exp(target_score - mx) / denom
            loss = -math.log(max(p_gold, 1e-12))
        else:
            p_gold = 1.0
            loss = 0.0

        # Step 5 & 6: Apply positive evidence to target and negative evidence to wrong-above tokens
        self._update_python_rows(rows, target_token, amount=positive_amount, negative_tokens=wrong_above)
        readout_diag = self._update_readout(rows, target_token, negative_tokens=wrong_above, amount=positive_amount)
        
        # Step 7: Return diagnostic dict
        return {
            'credit_used': True,
            'readout_credit_used': bool(readout_diag.get('readout_update_used', False)),
            'readout_active_features': int(readout_diag.get('readout_active_features', 0)),
            'target_probability': float(p_gold),
            'token_loss': float(token_loss),
            'wrong_above_count': int(len(wrong_above)),
            'positive_amount': int(positive_amount),
            'negative_tokens': wrong_above,
        }

    def score_map_from_features_fast(self, features: List[Tuple[str, float]]) -> Dict[str, float]:
        """Cheap lookahead scorer for recursive stabilization.

        P76 live scoring uses exact sparse evidence plus the adaptive
        readout/inverted index and skips the slow full HDC-KNN pass.  HDC
        bind/bundle still contributes through readout projection.
        """
        rows = self._rows_from_features(features)
        if not rows:
            return {}
        self.fast_score_calls += 1
        self._maybe_resize_readout()
        key = None
        if self.runtime_cache_enabled:
            key = ('fast', self._rows_cache_key(rows), int(self.hdc_bits), int(self.updates), int(self.readout_updates), int(self.readout_resize_count))
            cached = self._score_cache.get(key)
            if cached is not None:
                self.score_cache_hits += 1
                self._score_cache.move_to_end(key)
                return dict(cached)
            self.score_cache_misses += 1
        scores: Dict[str, float] = defaultdict(float)
        self._score_exact_python(rows, scores)
        self._score_readout_python(rows, scores)
        out = dict(scores)
        if key is not None:
            self._cache_put(self._score_cache, key, out)
        return out

    def score_from_features_fast(self, features: List[Tuple[str, float]], top_k: int = 0) -> List[Tuple[str, float]]:
        scores = self.score_map_from_features_fast(features)
        if not scores:
            return []
        rank_k = int(top_k) if int(top_k or 0) > 0 else dynamic_hdc_neighbors(len(scores), int(self.updates), len(self.vocab))
        return self._rank_top(scores, rank_k)

    def score_from_features(self, features: List[Tuple[str, float]], top_k: int = 0) -> List[Tuple[str, float]]:
        """P76 fast-only public scorer.

        The old full HDC-KNN scorer is no longer the default live path.  HDC
        structure still enters through bind/bundle context vectors used by the
        adaptive readout, but token ranking avoids the slow all-neighbor HDC pass.
        """
        return self.score_from_features_fast(features, top_k=top_k)

    def peek_top_token(self, features: List[Tuple[str, float]]) -> str | None:
        out = self.score_from_features(features, top_k=1)
        return out[0][0] if out else None

    def lookup_with_backoff(
        self,
        context_features: List[Tuple[str, float]],
        field_features: Optional[List[Tuple[str, float]]] = None
    ) -> Tuple[Dict[str, float], int]:
        """Progressive backoff with HDC similarity search (Task 3.3).
        
        Implements progressive backoff sequence:
        - Level 0: Full context (K features)
        - Level 1: Half context (K/2 features)
        - Level 2: Quarter context (K/4 features)
        - Level 3: HDC similarity search (approximate matching)
        - Level 4: Unigram distribution
        - Level 5: Field-only features (if provided)
        - Level 6: Empty (failure)
        
        Args:
            context_features: List of (feature_id, weight) tuples for context
            field_features: Optional list of field-level features for fallback
            
        Returns:
            Tuple of (token_distribution, backoff_level) where:
            - token_distribution: Dict mapping tokens to scores
            - backoff_level: Integer indicating which backoff level succeeded (0-6)
        
        Requirements: 4.1, 4.2, 4.3, 10.1, 10.2, 10.3, 10.4
        """
        if not context_features:
            # No context, fall through to field or empty
            if field_features:
                scores = self.score_map_from_features_fast(field_features)
                return (scores, 5)
            return ({}, 6)
        
        K = len(context_features)
        
        # Level 0: Full context (K features)
        scores = self.score_map_from_features_fast(context_features)
        if scores:
            return (scores, 0)
        
        # Level 1: Half context (K/2 features)
        if K >= 2:
            half_idx = K - K // 2
            half_features = context_features[half_idx:]
            scores = self.score_map_from_features_fast(half_features)
            if scores:
                return (scores, 1)
        
        # Level 2: Quarter context (K/4 features)
        if K >= 4:
            quarter_idx = K - K // 4
            quarter_features = context_features[quarter_idx:]
            scores = self.score_map_from_features_fast(quarter_features)
            if scores:
                return (scores, 2)
        
        # Level 3: HDC similarity search
        # Generate HDC bundle vector for context and find similar stored vectors
        if context_features and self.hdc_next:
            rows = self._rows_from_features(context_features)
            if rows:
                # Get HDC query vector
                q_bits, q_hv = self._hdc_vector(rows, self.hdc_bits)
                if q_hv:
                    # Find candidate HDC rows using bucketing
                    candidates: Set[Tuple[int, int]] = set()
                    for key in self._bucket_keys_for(q_bits, q_hv):
                        candidates.update(self.hdc_buckets.get(key, ()))
                    
                    # If too few candidates, expand search
                    min_neighbors = dynamic_hdc_neighbors(len(self.hdc_next), int(self.updates), len(self.vocab))
                    if len(candidates) < min_neighbors:
                        # Add more candidates from all HDC rows
                        all_candidates = list(self.hdc_next.keys())
                        if len(all_candidates) <= 1000:  # Only scan if reasonable size
                            candidates.update(all_candidates)
                    
                    if candidates:
                        # Compute similarities for all candidates
                        sims: List[Tuple[float, Tuple[int, int]]] = []
                        for row_bits, hv in candidates:
                            common_bits = max(1, min(int(row_bits), int(q_bits)))
                            common_mask = _mask(common_bits)
                            dist = ((int(q_hv) & common_mask) ^ (int(hv) & common_mask)).bit_count()
                            sim = 1.0 - (dist / float(common_bits))
                            sims.append((sim, (int(row_bits), int(hv))))
                        
                        if sims:
                            # Sort by similarity
                            sims.sort(key=lambda x: -x[0])
                            
                            # Use top-K most similar vectors
                            top_k = min(len(sims), self._dynamic_top_neighbors())
                            scores_hdc: Dict[str, float] = defaultdict(float)
                            
                            for sim, hv_key in sims[:top_k]:
                                if sim < 0.3:  # Minimum similarity threshold
                                    continue
                                row = self.hdc_next.get(hv_key)
                                if row:
                                    total = float(self.hdc_totals.get(hv_key, 0) or sum(row.values()) or 1)
                                    for tok, cnt in row.items():
                                        # Weight by similarity and frequency
                                        scores_hdc[tok] += sim * (float(cnt) / total)
                            
                            if scores_hdc:
                                return (dict(scores_hdc), 3)
        
        # Level 4: Unigram distribution (global token frequencies)
        if self.vocab:
            total_vocab = sum(self.vocab.values())
            if total_vocab > 0:
                unigram_scores = {tok: float(cnt) / total_vocab for tok, cnt in self.vocab.items()}
                return (unigram_scores, 4)
        
        # Level 5: Field-only features (if provided)
        if field_features:
            scores = self.score_map_from_features_fast(field_features)
            if scores:
                return (scores, 5)
        
        # Level 6: Failure - no candidates found
        return ({}, 6)

    def _serial_hdc_key(self, key: Tuple[int, int]) -> str:
        return f'{int(key[0])}:{int(key[1])}'

    def _parse_hdc_key(self, k: Any) -> Tuple[int, int]:
        s = str(k)
        if ':' in s:
            b, v = s.split(':', 1)
            return (int(b), int(v))
        # P64 compatibility: old state had bare 128-bit integer keys.
        return (128, int(s))

    def __getstate__(self):
        return {
            'feature_next': {k: dict(v) for k, v in self.feature_next.items()},
            'feature_totals': dict(self.feature_totals),
            'feature_wrong': {k: dict(v) for k, v in self.feature_wrong.items()},
            'feature_wrong_totals': dict(self.feature_wrong_totals),
            'vocab': dict(self.vocab),
            'hdc_next': {self._serial_hdc_key(k): dict(v) for k, v in self.hdc_next.items()},
            'hdc_totals': {self._serial_hdc_key(k): int(v) for k, v in self.hdc_totals.items()},
            'hdc_wrong': {self._serial_hdc_key(k): dict(v) for k, v in self.hdc_wrong.items()},
            'hdc_wrong_totals': {self._serial_hdc_key(k): int(v) for k, v in self.hdc_wrong_totals.items()},
            'feature_hv': {k: str(int(v)) for k, v in self.feature_hv.items()},
            'hdc_bits': int(self.hdc_bits),
            'resize_count': int(self.resize_count),
            'updates': self.updates,
            'readout_enabled': bool(self.readout_enabled),
            'readout_hidden_dim': int(self.readout_hidden_dim),
            'readout_vocab_cap': int(self.readout_vocab_cap),
            'readout_lr': float(self.readout_lr),
            'readout_auto_resize': bool(self.readout_auto_resize),
            'readout_min_hidden': int(self.readout_min_hidden),
            'readout_max_hidden': int(self.readout_max_hidden),
            'readout_min_vocab_cap': int(self.readout_min_vocab_cap),
            'readout_max_vocab_cap': int(self.readout_max_vocab_cap),
            'readout_active_features': int(self.readout_active_features),
            'readout_resize_count': int(self.readout_resize_count),
            'readout_weights': {k: {str(i): float(v) for i, v in w.items()} for k, w in self.readout_weights.items()},
            'readout_token_updates': dict(self.readout_token_updates),
            'readout_updates': int(self.readout_updates),
        }

    def __setstate__(self, state):
        self.feature_next = defaultdict(Counter)
        for k, v in state.get('feature_next', {}).items():
            self.feature_next[k] = Counter(v)
        self.feature_totals = Counter(state.get('feature_totals', {}))
        self.feature_wrong = defaultdict(Counter)
        for k, v in state.get('feature_wrong', {}).items():
            self.feature_wrong[k] = Counter(v)
        self.feature_wrong_totals = Counter(state.get('feature_wrong_totals', {}))
        self.vocab = Counter(state.get('vocab', {}))
        self.hdc_next = defaultdict(Counter)
        for k, v in state.get('hdc_next', {}).items():
            self.hdc_next[self._parse_hdc_key(k)] = Counter(v)
        self.hdc_totals = Counter({self._parse_hdc_key(k): int(v) for k, v in state.get('hdc_totals', {}).items()})
        self.hdc_wrong = defaultdict(Counter)
        for k, v in state.get('hdc_wrong', {}).items():
            self.hdc_wrong[self._parse_hdc_key(k)] = Counter(v)
        self.hdc_wrong_totals = Counter({self._parse_hdc_key(k): int(v) for k, v in state.get('hdc_wrong_totals', {}).items()})
        self.feature_hv = {str(k): int(v) for k, v in state.get('feature_hv', {}).items()}
        self.updates = int(state.get('updates', 0))
        self.hdc_bits = max(int(state.get('hdc_bits', 0) or 0), self._target_bits())
        self.resize_count = int(state.get('resize_count', 0))
        self.readout_enabled = bool(state.get('readout_enabled', True))
        self.readout_auto_resize = bool(state.get('readout_auto_resize', True))
        self.readout_min_hidden = int(state.get('readout_min_hidden', 0))
        self.readout_max_hidden = int(state.get('readout_max_hidden', 0))
        self.readout_min_vocab_cap = int(state.get('readout_min_vocab_cap', 0))
        self.readout_max_vocab_cap = int(state.get('readout_max_vocab_cap', 0))
        self.readout_hidden_dim = int(state.get('readout_hidden_dim', READOUT_DEFAULT_HIDDEN))
        self.readout_vocab_cap = int(state.get('readout_vocab_cap', READOUT_DEFAULT_VOCAB_CAP))
        self.readout_lr = float(state.get('readout_lr', READOUT_DEFAULT_LR))
        self.readout_active_features = int(state.get('readout_active_features', READOUT_ACTIVE_FEATURES))
        self.readout_resize_count = int(state.get('readout_resize_count', 0))
        self.readout_weights = defaultdict(dict)
        for tok, wdict in state.get('readout_weights', {}).items():
            self.readout_weights[str(tok)] = {int(i): float(v) for i, v in dict(wdict).items()}
        self.readout_index = defaultdict(dict)
        self._rebuild_readout_index()
        self.readout_token_updates = Counter(state.get('readout_token_updates', {}))
        self.readout_updates = int(state.get('readout_updates', 0))
        self.readout_last_active = 0
        self.readout_last_candidates = 0
        self.runtime_cache_enabled = True
        self.runtime_cache_max_entries = 0
        self._score_cache = OrderedDict()
        self._hdc_context_cache = OrderedDict()
        self._readout_projection_cache = OrderedDict()
        self.score_cache_hits = 0
        self.score_cache_misses = 0
        self.hdc_context_cache_hits = 0
        self.hdc_context_cache_misses = 0
        self.readout_projection_cache_hits = 0
        self.readout_projection_cache_misses = 0
        self.fast_score_calls = 0
        self.full_score_calls = 0
        self.rust_fast_path_calls = 0
        self._maybe_resize_readout(force=True)
        self.hdc_buckets = defaultdict(set)
        self._rebuild_buckets()
        self.rust = None
        self.rust_available = False
        try:
            from . import puhl_luck_core  # type: ignore
            cls = getattr(puhl_luck_core, 'RustHebbianHdcCountEvidence', None)
            if cls is not None:
                self.rust = cls()
                self.rust_available = True
                self._sync_rust_policy()
                if hasattr(self.rust, 'set_feature_hv_decimal') and self.feature_hv:
                    self.rust.set_feature_hv_decimal([(k, str(v)) for k, v in self.feature_hv.items()])
                batch = []
                for f, row in self.feature_next.items():
                    for tok, cnt in row.items():
                        batch.append(([(f, 1.0)], tok, int(cnt)))
                if batch and hasattr(self.rust, 'update_many_weighted'):
                    self.rust.update_many_weighted(batch)
        except Exception:
            self.rust = None
            self.rust_available = False

    def learned_state(self) -> Dict[str, Any]:
        """Return only learned model state, excluding runtime caches/Rust objects."""
        return self.__getstate__()

    def storage_summary(self) -> Dict[str, Any]:
        """Measured serialized learned-state size, not process RSS."""
        components = self.learned_state()
        comp_bytes: Dict[str, int] = {}
        for k, v in components.items():
            try:
                comp_bytes[str(k)] = len(pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                comp_bytes[str(k)] = -1
        payload = pickle.dumps(components, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = gzip.compress(payload)
        return {
            'learned_pickle_bytes': int(len(payload)),
            'learned_gzip_bytes': int(len(compressed)),
            'component_pickle_bytes': comp_bytes,
            'runtime_cache_excluded': True,
            'process_rss_excluded': True,
        }

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'feature_count': len(self.feature_next),
            'feature_next_entries': sum(len(v) for v in self.feature_next.values()),
            'feature_wrong_entries': sum(len(v) for v in self.feature_wrong.values()),
            'hdc_row_count': len(self.hdc_next),
            'hdc_next_entries': sum(len(v) for v in self.hdc_next.values()),
            'hdc_wrong_entries': sum(len(v) for v in self.hdc_wrong.values()),
            'hebbian_feature_vectors': len(self.feature_hv),
            'vocab_size': len(self.vocab),
            'updates': self.updates,
            'storage': self.storage_summary(),
            'rust_available': bool(self.rust_available),
            'rust_scoring_used': False,
            'p76_fast_only_default': True,
            'p76_live_score_path': 'score_from_features_fast',
            'hdc_bits': int(self.hdc_bits),
            'hdc_bits_policy': 'dynamic_by_feature_event_row_vocab_no_code_cap',
            'hdc_resize_count': int(self.resize_count),
            'hdc_threshold': 'query_local_median_floor',
            'hdc_top_neighbors': self._dynamic_top_neighbors(),
            'hebbian_hdc': True,
            'dynamic_hdc': True,
            'hdc_binding_composition': True,
            'hdc_composite_vector_budget': 'dynamic_by_rows_events_memory_no_D2_symbolic_cap',
            'p73_adaptive_readout_enabled': bool(self.readout_enabled),
            'p73_readout_hidden_dim': int(self.readout_hidden_dim),
            'p73_readout_vocab_cap': int(self.readout_vocab_cap),
            'p73_readout_tracked_tokens': len(self.readout_weights),
            'p73_readout_materialized_weights': self.readout_parameter_count(),
            'p73_readout_dense_capacity_weights': self.readout_dense_capacity(),
            'p73_readout_fp16_mb_materialized': self.readout_parameter_count() * 2 / (1024 * 1024),
            'p73_readout_fp16_mb_dense_capacity': self.readout_dense_capacity() * 2 / (1024 * 1024),
            'p73_readout_auto_resize': bool(self.readout_auto_resize),
            'p73_readout_resize_count': int(self.readout_resize_count),
            'p73_readout_min_hidden': int(self.readout_min_hidden),
            'p73_readout_max_hidden': int(self.readout_max_hidden),
            'p73_readout_min_vocab_cap': int(self.readout_min_vocab_cap),
            'p73_readout_max_vocab_cap': int(self.readout_max_vocab_cap),
            'p73_readout_active_budget': int(self._readout_active_budget()),
            'p73_readout_inverted_index_entries': sum(len(v) for v in self.readout_index.values()),
            'p74_runtime_cache_enabled': True,
            'p74_score_cache_entries': len(getattr(self, '_score_cache', {})),
            'p74_score_cache_hits': int(getattr(self, 'score_cache_hits', 0)),
            'p74_score_cache_misses': int(getattr(self, 'score_cache_misses', 0)),
            'p74_hdc_context_cache_entries': len(getattr(self, '_hdc_context_cache', {})),
            'p74_hdc_context_cache_hits': int(getattr(self, 'hdc_context_cache_hits', 0)),
            'p74_hdc_context_cache_misses': int(getattr(self, 'hdc_context_cache_misses', 0)),
            'p74_readout_projection_cache_entries': len(getattr(self, '_readout_projection_cache', {})),
            'p74_readout_projection_cache_hits': int(getattr(self, 'readout_projection_cache_hits', 0)),
            'p74_readout_projection_cache_misses': int(getattr(self, 'readout_projection_cache_misses', 0)),
            'p74_fast_score_calls': int(getattr(self, 'fast_score_calls', 0)),
            'p74_full_score_calls': int(getattr(self, 'full_score_calls', 0)),
            'p74_rust_fast_path_calls': int(getattr(self, 'rust_fast_path_calls', 0)),
            'credit_assignment': 'rank_loss_signed_sparse_hdc_template_gates_plus_dynamic_adaptive_margin_readout',
        }

    def clear(self) -> None:
        self.feature_next.clear()
        self.feature_totals.clear()
        self.feature_wrong.clear()
        self.feature_wrong_totals.clear()
        self.vocab.clear()
        self.hdc_next.clear()
        self.hdc_totals.clear()
        self.hdc_wrong.clear()
        self.hdc_wrong_totals.clear()
        self.hdc_buckets.clear()
        self.feature_hv.clear()
        self.readout_weights.clear()
        self.readout_index.clear()
        self.readout_token_updates.clear()
        self.readout_updates = 0
        self.readout_last_active = 0
        self.readout_last_candidates = 0
        self.readout_resize_count = 0
        if hasattr(self, '_clear_runtime_caches'):
            self._clear_runtime_caches()
        self._maybe_resize_readout(force=True)
        self.updates = 0
        self.resize_count = 0
        self.hdc_bits = dynamic_hdc_bits(0, 0, 0, 0)
        if self.rust is not None:
            try:
                self.rust.clear()
                self._sync_rust_policy()
            except Exception:
                pass


SparseLogitTables = SparseEvidenceTables


class SparseTableWithCache:
    """Wrapper for sparse table with LRU caching of context sketches.
    
    Implements efficient context sketch computation with LRU caching to reduce
    repeated hash computation during generation. Each token extension of context
    requires a new hash, so caching provides significant speedup.
    
    Requirements: 2.1, 2.2, 9.1, 9.2, 9.3, 9.4
    """
    
    def __init__(self, cache_size: int = 1000):
        """Initialize sparse table with configurable cache size.
        
        Args:
            cache_size: Maximum number of context sketches to cache (default: 1000)
        """
        self.context_table: Dict[bytes, Counter[str]] = {}
        self._sketch_cache: OrderedDict[Tuple[str, ...], bytes] = OrderedDict()
        self.cache_size = max(1, int(cache_size))
        
        # Track cache hits/misses for diagnostics
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _context_sketch(self, tokens: Tuple[str, ...]) -> bytes:
        """Compute or retrieve cached BLAKE2b sketch for context tokens.
        
        Uses LRU caching to avoid repeated hash computation. When a token is
        appended to context, the previous context may already be cached.
        
        Args:
            tokens: Tuple of context tokens
            
        Returns:
            16-byte BLAKE2b digest serving as context sketch
        """
        # Check cache first (LRU)
        if tokens in self._sketch_cache:
            self.cache_hits += 1
            self._sketch_cache.move_to_end(tokens)  # Mark as recently used
            return self._sketch_cache[tokens]
        
        # Cache miss - compute new sketch
        self.cache_misses += 1
        
        # Compute BLAKE2b hash of concatenated tokens
        h = hashlib.blake2b(digest_size=16)
        for tok in tokens:
            h.update(str(tok).encode('utf-8', 'ignore'))
        sketch = h.digest()
        
        # Add to cache with LRU eviction
        self._sketch_cache[tokens] = sketch
        if len(self._sketch_cache) > self.cache_size:
            # Remove least recently used (oldest) entry
            self._sketch_cache.popitem(last=False)
        
        return sketch
    
    def lookup(self, tokens: Tuple[str, ...]) -> Counter[str]:
        """Lookup token distribution for context tokens.
        
        Computes context sketch (with caching) and retrieves associated
        token distribution from sparse table.
        
        Args:
            tokens: Tuple of context tokens
            
        Returns:
            Counter mapping tokens to occurrence counts, or empty Counter if not found
        """
        sketch = self._context_sketch(tokens)
        return self.context_table.get(sketch, Counter())
    
    def update(self, tokens: Tuple[str, ...], next_token: str, amount: int = 1) -> None:
        """Update sparse table with new context -> token association.
        
        Args:
            tokens: Tuple of context tokens
            next_token: Token that follows the context
            amount: Count to add (default: 1)
        """
        sketch = self._context_sketch(tokens)
        if sketch not in self.context_table:
            self.context_table[sketch] = Counter()
        self.context_table[sketch][str(next_token)] += max(1, int(amount))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics for diagnostics.
        
        Returns:
            Dict with cache_size, cache_entries, cache_hits, cache_misses,
            hit_rate, and table_entries
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': self.cache_size,
            'cache_entries': len(self._sketch_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'table_entries': len(self.context_table),
        }
    
    def clear_cache(self) -> None:
        """Clear the context sketch cache (preserves table data)."""
        self._sketch_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
