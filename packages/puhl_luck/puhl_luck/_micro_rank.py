from __future__ import annotations

from ._brain_common import *

class MicroRankModel:
    """Tiny inference-only ranker backed by the PMR1 shared feature hash space."""

    def __init__(
        self,
        feature_rows: Dict[int, List[Tuple[int, int]]],
        event_features: Dict[int, Tuple[int, ...]],
        hash_bits: int = 16,
    ):
        self.feature_rows = feature_rows
        self.event_features = event_features
        self.hash_bits = 32 if int(hash_bits) == 32 else 16
        self.event_feature_sets = {eid: set(features) for eid, features in event_features.items()}
        self.feature_event_freq = {h: max(1, len(rows)) for h, rows in feature_rows.items()}
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        self._choice_cache: Dict[str, Dict[str, Any]] = {}
        self._rank_cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[int, List[float]]] = {}
        self._predict_cache: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self._gpu_checked = False
        self._gpu_module = None

    def hash_weight(self, feature_hash: int) -> float:
        return 1.0 / math.sqrt(float(self.feature_event_freq.get(feature_hash, 1)))

    def set_weight(self, feature_set: set[int]) -> float:
        return sum(self.hash_weight(h) for h in feature_set)

    def shared_weight(self, left: set[int], right: set[int]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(self.hash_weight(h) for h in left if h in right)

    @classmethod
    def load(cls, path: str | Path) -> "MicroRankModel":
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_bytes(cls, payload: bytes) -> "MicroRankModel":
        raw = zlib.decompress(payload)
        if raw[:4] == b"PMR2":
            offset = 4
            hash_bits, offset = read_varuint(raw, offset)
            top_count, offset = read_varuint(raw, offset)
            feature_rows: Dict[int, List[Tuple[int, int]]] = {}
            for _ in range(top_count):
                h, offset = read_varuint(raw, offset)
                row_count, offset = read_varuint(raw, offset)
                rows = []
                for _ in range(row_count):
                    eid, offset = read_varuint(raw, offset)
                    count, offset = read_varuint(raw, offset)
                    rows.append((eid, count))
                if rows:
                    feature_rows[h] = rows
            event_count, offset = read_varuint(raw, offset)
            event_features: Dict[int, Tuple[int, ...]] = {}
            for _ in range(event_count):
                eid, offset = read_varuint(raw, offset)
                feature_count, offset = read_varuint(raw, offset)
                features = []
                for _ in range(feature_count):
                    h, offset = read_varuint(raw, offset)
                    features.append(h)
                if features:
                    event_features[eid] = tuple(dict.fromkeys(features))
            return cls(feature_rows, event_features, hash_bits=hash_bits)
        if raw[:4] != b"PMR1":
            raise ValueError("micro rank file must start with PMR1")
        offset = 4
        top_count, event_count = struct.unpack_from("<HH", raw, offset)
        offset += 4
        feature_rows: Dict[int, List[Tuple[int, int]]] = {}
        for _ in range(top_count):
            h, row_count = struct.unpack_from("<HB", raw, offset)
            offset += 3
            rows = []
            for _ in range(row_count):
                eid, count = struct.unpack_from("<HB", raw, offset)
                offset += 3
                rows.append((eid, count))
            if rows:
                feature_rows[h] = rows
        event_features: Dict[int, Tuple[int, ...]] = {}
        for _ in range(event_count):
            eid, feature_count = struct.unpack_from("<HB", raw, offset)
            offset += 3
            features = []
            for _ in range(feature_count):
                (h,) = struct.unpack_from("<H", raw, offset)
                offset += 2
                features.append(h)
            if features:
                event_features[eid] = tuple(dict.fromkeys(features))
        return cls(feature_rows, event_features, hash_bits=16)

    def query_state(self, query: str) -> Dict[str, Any]:
        key = str(query)
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        hashes = micro_hashes_for_value(key, bits=self.hash_bits)
        query_set = set(hashes)
        event_scores: Dict[int, float] = {}
        for h in hashes[:96]:
            for eid, count in self.feature_rows.get(h, ()):
                event_scores[eid] = event_scores.get(eid, 0.0) + float(count) * self.hash_weight(h)
        top_events = sorted(event_scores.items(), key=lambda item: item[1], reverse=True)[:8]
        state = {"hashes": hashes, "set": query_set, "weight": self.set_weight(query_set), "top_events": top_events}
        if len(self._query_cache) > 4096:
            self._query_cache.clear()
            self._rank_cache.clear()
            self._predict_cache.clear()
        self._query_cache[key] = state
        return state

    def choice_state(self, choice: str) -> Dict[str, Any]:
        key = str(choice)
        cached = self._choice_cache.get(key)
        if cached is not None:
            return cached
        hashes = micro_hashes_for_value(key, bits=self.hash_bits)
        choice_set = set(hashes)
        row = {
            "hashes": hashes,
            "set": choice_set,
            "weight": self.set_weight(choice_set),
            "length_penalty": 1.0 + 0.06 * math.log1p(len(hashes)),
        }
        if len(self._choice_cache) > 4096:
            self._choice_cache.clear()
            self._rank_cache.clear()
            self._predict_cache.clear()
        self._choice_cache[key] = row
        return row

    def score_state(self, query_state: Dict[str, Any], choice_state: Dict[str, Any]) -> float:
        choice_set = choice_state["set"]
        query_set = query_state["set"]
        if not choice_set:
            return 0.0
        direct = 0.0
        if query_set:
            direct_shared = self.shared_weight(query_set, choice_set)
            direct = direct_shared / math.sqrt(max(1e-9, float(query_state.get("weight", 0.0) * choice_state.get("weight", 0.0))))
        support = 0.0
        alignment_num = 0.0
        alignment_den = 0.0
        for eid, event_score in query_state["top_events"]:
            event_set = self.event_feature_sets.get(eid)
            if not event_set:
                continue
            shared = self.shared_weight(event_set, choice_set)
            if shared <= 0.0:
                continue
            event_weight = self.set_weight(event_set)
            choice_weight = float(choice_state.get("weight", 0.0))
            weight = math.log1p(event_score)
            support = max(support, (shared / math.sqrt(max(1e-9, event_weight * choice_weight))) * weight)
            alignment_num += (shared / max(1e-9, event_weight + choice_weight - shared)) * weight
            alignment_den += weight
        alignment = alignment_num / max(1e-9, alignment_den)
        return (support + 0.6 * alignment + 0.25 * direct) / max(1e-9, float(choice_state["length_penalty"]))

    def gpu_module(self, required: bool = False) -> Any:
        if not self._gpu_checked:
            self._gpu_checked = True
            try:
                import cupy as cp  # type: ignore

                if cp.cuda.runtime.getDeviceCount() > 0:
                    self._gpu_module = cp
            except Exception:
                self._gpu_module = None
        if required and self._gpu_module is None:
            raise RuntimeError("GPU backend requires CuPy with a visible CUDA device")
        return self._gpu_module

    def resolve_device(self, device: str = "auto") -> str:
        selected = str(device or "auto").lower()
        if selected == "gpu":
            if self.hash_bits != 16:
                raise RuntimeError("GPU backend currently requires a 16-bit micro model")
            self.gpu_module(required=True)
            return "gpu"
        if selected == "auto":
            return "cpu"
        return "cpu"

    def score_states_gpu(self, query_state: Dict[str, Any], choice_states: List[Dict[str, Any]]) -> List[float]:
        cp = self.gpu_module(required=True)
        if not choice_states:
            return []
        choice_sets = [row["set"] for row in choice_states]
        choice_lens = cp.asarray([max(1, len(row["set"])) for row in choice_states], dtype=cp.float32)
        penalties = cp.asarray([float(row["length_penalty"]) for row in choice_states], dtype=cp.float32)
        feature_width = 65536

        choice_mask = cp.zeros((len(choice_states), feature_width), dtype=cp.bool_)
        for row_idx, row in enumerate(choice_states):
            hashes = row["hashes"]
            if hashes:
                choice_mask[row_idx, cp.asarray(hashes, dtype=cp.int32)] = True

        query_hashes = query_state["hashes"]
        direct = cp.zeros(len(choice_states), dtype=cp.float32)
        if query_hashes:
            shared_query = choice_mask[:, cp.asarray(query_hashes, dtype=cp.int32)].sum(axis=1, dtype=cp.float32)
            direct = shared_query / cp.sqrt(cp.asarray(float(len(query_hashes)), dtype=cp.float32) * choice_lens)

        top_events = query_state["top_events"]
        if not top_events:
            scores = (0.25 * direct) / penalties
            return [float(x) for x in cp.asnumpy(scores)]

        event_sets = [self.event_feature_sets.get(eid, set()) for eid, _ in top_events]
        event_lens = cp.asarray([max(1, len(row)) for row in event_sets], dtype=cp.float32)
        weights = cp.asarray([math.log1p(score) for _, score in top_events], dtype=cp.float32)
        event_mask = cp.zeros((len(event_sets), feature_width), dtype=cp.bool_)
        for row_idx, hashes in enumerate(event_sets):
            if hashes:
                event_mask[row_idx, cp.asarray(list(hashes), dtype=cp.int32)] = True

        shared = cp.matmul(event_mask.astype(cp.float32), choice_mask.T.astype(cp.float32))
        support = (shared / cp.sqrt(event_lens[:, None] * choice_lens[None, :]) * weights[:, None]).max(axis=0)
        union = event_lens[:, None] + choice_lens[None, :] - shared
        alignment = ((shared / cp.maximum(union, 1.0)) * weights[:, None]).sum(axis=0) / cp.maximum(weights.sum(), 1e-9)
        scores = (support + 0.6 * alignment + 0.25 * direct) / penalties
        return [float(x) for x in cp.asnumpy(scores)]

    def score_choices(self, query_state: Dict[str, Any], choice_states: List[Dict[str, Any]], device: str = "cpu") -> List[float]:
        selected = self.resolve_device(device)
        if selected == "gpu":
            return self.score_states_gpu(query_state, choice_states)
        return [self.score_state(query_state, row) for row in choice_states]

    def rank(self, query: str, choices: List[str], device: str = "cpu") -> Tuple[int, List[float]]:
        key = (str(query), tuple(str(choice) for choice in choices))
        use_cache = str(device or "cpu").lower() in {"cpu", "auto"}
        if use_cache:
            cached = self._rank_cache.get(key)
            if cached is not None:
                return cached
        state = self.query_state(key[0])
        raw_scores = self.score_choices(state, [self.choice_state(choice) for choice in key[1]], device=device)
        scores = self.normalize_scores(raw_scores)
        if not scores:
            return 0, []
        result = (max(range(len(scores)), key=lambda i: scores[i]), scores)
        if use_cache:
            if len(self._rank_cache) > 4096:
                self._rank_cache.clear()
            self._rank_cache[key] = result
        return result

    def predict(self, query: str, choices: List[str], device: str = "cpu") -> int:
        key = (str(query), tuple(str(choice) for choice in choices))
        use_cache = str(device or "cpu").lower() in {"cpu", "auto"}
        if use_cache:
            cached = self._predict_cache.get(key)
            if cached is not None:
                return cached
        if not key[1]:
            return 0
        state = self.query_state(key[0])
        choice_states = [self.choice_state(choice) for choice in key[1]]
        if self.resolve_device(device) == "gpu":
            raw_scores = self.score_states_gpu(state, choice_states)
            return max(range(len(raw_scores)), key=lambda i: raw_scores[i]) if raw_scores else 0
        best_idx = 0
        best_score = -float("inf")
        for idx, choice_state in enumerate(choice_states):
            score = self.score_state(state, choice_state)
            if score > best_score:
                best_idx = idx
                best_score = score
        if use_cache:
            if len(self._predict_cache) > 4096:
                self._predict_cache.clear()
            self._predict_cache[key] = best_idx
        return best_idx

    @staticmethod
    def normalize_scores(raw_scores: List[float]) -> List[float]:
        if not raw_scores:
            return []
        if len(raw_scores) == 1:
            return [1.0]
        max_score = max(raw_scores)
        min_score = min(raw_scores)
        if max_score - min_score < 1e-12:
            return [1.0 / len(raw_scores) for _ in raw_scores]
        mean = sum(raw_scores) / len(raw_scores)
        var = sum((score - mean) ** 2 for score in raw_scores) / len(raw_scores)
        scale = math.sqrt(var) if var > 1e-12 else (max_score - min_score)
        z_scores = [(score - mean) / max(1e-9, scale) for score in raw_scores]
        peak = max(z_scores)
        exps = [math.exp(max(-20.0, min(20.0, score - peak))) for score in z_scores]
        total = sum(exps) + 1e-12
        return [value / total for value in exps]
