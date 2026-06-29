
"""P85 compact frozen inference artifact.

Training can keep rich Python dictionaries.  Inference must not.  This module
freezes learned sparse evidence into integer-id CSR arrays and uses a Rust
scorer when available.  Hardware/thread counts never enter capacity formulas;
Rust is only an execution backend.
"""
from __future__ import annotations

import gzip
import pickle
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def fnv1a64(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in str(s).encode('utf-8', 'ignore'):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return int(h)


def _row_reliability(row: Dict[str, int], total: float) -> float:
    import math
    total = float(total or sum(row.values()) or 1.0)
    conc = 0.0
    for cnt in row.values():
        p = float(cnt) / total
        conc += p * p
    return float(math.log1p(total) * conc)


@dataclass
class FrozenStorageSummary:
    rows: int
    entries: int
    tokens: int
    artifact_pickle_bytes: int
    artifact_gzip_bytes: int
    rust_available: bool


class FrozenSparseLogitModel:
    """Compact inference-only model.

    The artifact stores no Python training counters, HDC feature vectors, runtime
    caches, or process state.  It is primarily exact sparse evidence frozen as:

        row_hash -> CSR postings [(token_id, signed_score)]

    Signed score already includes row reliability and positive/negative sparse
    evidence.  Active row weights are supplied at inference time by the existing
    feature generator.  Scoring uses RustFrozenExactScorer when the extension is
    available, otherwise a Python CSR fallback.
    """

    VERSION = 'P89_frozen_rust_full_prompt_budget_parity_v1'

    def __init__(self, payload: Dict[str, Any]):
        self.version = payload.get('version', self.VERSION)
        self.tokens: List[str] = list(payload['tokens'])
        self.row_ids = array('Q', payload['row_ids'])
        self.row_ptr = array('I', payload['row_ptr'])
        self.token_ids = array('I', payload['token_ids'])
        self.values = array('f', payload['values'])
        self.template_gates: Dict[str, float] = {str(k): float(v) for k, v in dict(payload.get('template_gates', {})).items()}
        self.meta = dict(payload.get('meta', {}))
        self.row_index: Dict[int, int] = {int(r): i for i, r in enumerate(self.row_ids)}
        self.rust = None
        self.rust_next = None
        self.rust_available = False
        self.rust_next_available = False
        self._try_rust()

    @classmethod
    def from_generator(cls, gen: Any) -> 'FrozenSparseLogitModel':
        tbl = getattr(gen, 'tables', gen)
        vocab = set()
        for dname in ('feature_next', 'feature_wrong'):
            for row in getattr(tbl, dname, {}).values():
                vocab.update(str(t) for t in row.keys())
        for t in getattr(tbl, 'vocab', {}).keys():
            vocab.add(str(t))
        tokens = sorted(vocab)
        tok_to_id = {t: i for i, t in enumerate(tokens)}

        row_scores: Dict[int, Dict[int, float]] = {}
        row_names: Dict[int, str] = {}

        template_gates: Dict[str, float] = {}
        for tid in set(getattr(gen, 'template_positive', {}).keys()) | set(getattr(gen, 'template_negative', {}).keys()):
            pos = float(getattr(gen, 'template_positive', {}).get(tid, 0))
            neg = float(getattr(gen, 'template_negative', {}).get(tid, 0))
            template_gates[str(tid)] = max(0.0, (pos + 1.0) / (pos + neg + 1.0)) if (pos + neg) > 0 else 1.0

        def add_row(fname: str, row: Dict[str, int], total: float, sign: float) -> None:
            if not row:
                return
            rid = fnv1a64(fname)
            row_names.setdefault(rid, str(fname))
            rel = _row_reliability(row, total)
            denom = float(total or sum(row.values()) or 1.0)
            dst = row_scores.setdefault(rid, {})
            for tok, cnt in row.items():
                tid = tok_to_id.get(str(tok))
                if tid is None:
                    continue
                dst[tid] = float(dst.get(tid, 0.0)) + float(sign) * rel * (float(cnt) / denom)

        for fname, row in getattr(tbl, 'feature_next', {}).items():
            total = float(getattr(tbl, 'feature_totals', {}).get(fname, 0) or sum(row.values()) or 1.0)
            add_row(str(fname), dict(row), total, +1.0)
        for fname, row in getattr(tbl, 'feature_wrong', {}).items():
            total = float(getattr(tbl, 'feature_wrong_totals', {}).get(fname, 0) or sum(row.values()) or 1.0)
            add_row(str(fname), dict(row), total, -1.0)

        # Build CSR sorted by row id.  Zero-valued cancellations are omitted.
        row_ids = []
        row_ptr = [0]
        token_ids = []
        values = []
        for rid in sorted(row_scores):
            items = [(tid, val) for tid, val in row_scores[rid].items() if abs(float(val)) > 1e-12]
            if not items:
                continue
            row_ids.append(int(rid))
            for tid, val in sorted(items):
                token_ids.append(int(tid))
                values.append(float(val))
            row_ptr.append(len(token_ids))

        try:
            data_scale = dict(gen._data_scale()) if hasattr(gen, '_data_scale') else {}
        except Exception:
            data_scale = {}
        payload = {
            'version': cls.VERSION,
            'tokens': tokens,
            'row_ids': row_ids,
            'row_ptr': row_ptr,
            'token_ids': token_ids,
            'values': values,
            'template_gates': template_gates,
            'meta': {
                'source': 'SparseLogitGenerator.freeze',
                'rows': len(row_ids),
                'entries': len(token_ids),
                'tokens': len(tokens),
                'pairs_learned': int(getattr(gen, 'pairs_learned', 0)),
                'tokens_learned': int(getattr(gen, 'tokens_learned', 0)),
                'data_scale_events': int(data_scale.get('events', 0) or 0),
                'data_scale_features': int(data_scale.get('features', 0) or 0),
                'data_scale_rows': int(data_scale.get('rows', 0) or 0),
                'data_scale_vocab': int(data_scale.get('vocab', len(tokens)) or len(tokens)),
                'exact_sparse_only': True,
                'readout_not_serialized': True,
                'hdc_vectors_not_serialized': True,
                'python_training_dicts_excluded': True,
                'runtime_caches_excluded': True,
                'p87_dense_touched_rust_scorer': True,
                'p87_batch_generate_rust': True,
                'p89_rust_prompt_and_active_feature_budget_parity': True,
            },
        }
        return cls(payload)

    def _payload(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'tokens': self.tokens,
            'row_ids': list(self.row_ids),
            'row_ptr': list(self.row_ptr),
            'token_ids': list(self.token_ids),
            'values': list(self.values),
            'template_gates': dict(self.template_gates),
            'meta': dict(self.meta),
        }

    def _try_rust(self) -> None:
        try:
            from puhl_luck_core import RustFrozenExactScorer  # type: ignore
            self.rust = RustFrozenExactScorer(
                self.tokens,
                list(self.row_ids),
                list(self.row_ptr),
                list(self.token_ids),
                [float(x) for x in self.values],
            )
            self.rust_available = True
        except Exception:
            self.rust = None
            self.rust_available = False
        try:
            from puhl_luck_core import RustFrozenNextEngine  # type: ignore
            gate_items = [(str(k), float(v)) for k, v in self.template_gates.items()]
            # P89: pass data-scale metadata through the existing gate_items
            # channel so the Rust tokenizer/feature extractor can use the same
            # dynamic budgets as the Python frozen path.  These are not model
            # gates and are filtered out by RustFrozenNextEngine::new.
            for k in ('data_scale_events', 'data_scale_features', 'data_scale_rows', 'data_scale_vocab'):
                if k in self.meta:
                    gate_items.append((f'__p88_{k}', float(self.meta.get(k) or 0)))
            self.rust_next = RustFrozenNextEngine(
                self.tokens,
                list(self.row_ids),
                list(self.row_ptr),
                list(self.token_ids),
                [float(x) for x in self.values],
                gate_items,
            )
            self.rust_next_available = True
        except Exception:
            self.rust_next = None
            self.rust_next_available = False

    def save(self, path: str | Path, compress: bool = True) -> str:
        path = Path(path)
        payload = pickle.dumps(self._payload(), protocol=pickle.HIGHEST_PROTOCOL)
        if compress or str(path).endswith('.gz'):
            if not str(path).endswith('.gz'):
                path = path.with_suffix(path.suffix + '.gz')
            path.write_bytes(gzip.compress(payload))
        else:
            path.write_bytes(payload)
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> 'FrozenSparseLogitModel':
        raw = Path(path).read_bytes()
        if str(path).endswith('.gz'):
            raw = gzip.decompress(raw)
        return cls(pickle.loads(raw))

    def storage_summary(self) -> Dict[str, Any]:
        payload = pickle.dumps(self._payload(), protocol=pickle.HIGHEST_PROTOCOL)
        gz = gzip.compress(payload)
        return {
            'version': self.version,
            'rows': int(len(self.row_ids)),
            'entries': int(len(self.token_ids)),
            'tokens': int(len(self.tokens)),
            'artifact_pickle_bytes': int(len(payload)),
            'artifact_gzip_bytes': int(len(gz)),
            'row_ids_bytes': int(len(self.row_ids) * self.row_ids.itemsize),
            'row_ptr_bytes': int(len(self.row_ptr) * self.row_ptr.itemsize),
            'token_ids_bytes': int(len(self.token_ids) * self.token_ids.itemsize),
            'values_bytes': int(len(self.values) * self.values.itemsize),
            'rust_available': bool(self.rust_available),
            'rust_next_available': bool(self.rust_next_available),
            'template_gates': int(len(self.template_gates)),
            'meta': dict(self.meta),
        }

    def _score_py(self, rows: Sequence[Tuple[str, float]], top_k: int = 0) -> List[Tuple[str, float]]:
        scores: Dict[int, float] = {}
        for fname, weight in rows:
            rid = fnv1a64(str(fname))
            idx = self.row_index.get(rid)
            if idx is None:
                continue
            w = float(weight)
            if w <= 0.0:
                continue
            start = self.row_ptr[idx]
            end = self.row_ptr[idx + 1]
            for j in range(start, end):
                tid = int(self.token_ids[j])
                scores[tid] = scores.get(tid, 0.0) + w * float(self.values[j])
        ranked = sorted(((self.tokens[tid], score) for tid, score in scores.items()), key=lambda kv: (-kv[1], kv[0]))
        if top_k and int(top_k) > 0:
            ranked = ranked[: int(top_k)]
        return ranked

    def score_features(self, rows: Sequence[Tuple[str, float]], top_k: int = 0) -> List[Tuple[str, float]]:
        if self.rust is not None:
            try:
                row_ids = [fnv1a64(str(f)) for f, _w in rows]
                weights = [float(w) for _f, w in rows]
                return [(str(t), float(s)) for t, s in self.rust.score_row_ids(row_ids, weights, int(top_k or 0))]
            except Exception:
                pass
        return self._score_py(rows, top_k=top_k)

    def predict_text_rust(self, input_text: str, top_k: int = 1) -> Tuple[str, List[Tuple[str, float]], Dict[str, Any]]:
        if self.rust_next is None:
            return '', [], {'rust_next_available': False, 'fallback_required': True}
        tok, ranked, diag = self.rust_next.predict_text(str(input_text), int(top_k or 1))
        diag = dict(diag)
        diag['rust_next_available'] = True
        diag['python_feature_extraction_used'] = False
        return str(tok), [(str(t), float(s)) for t, s in ranked], diag

    def batch_predict_text_rust(self, inputs: Sequence[str], top_k: int = 1) -> List[Tuple[str, List[Tuple[str, float]]]]:
        if self.rust_next is None:
            return [('', []) for _ in inputs]
        return [(str(tok), [(str(t), float(s)) for t, s in ranked]) for tok, ranked in self.rust_next.batch_predict_text([str(x) for x in inputs], int(top_k or 1))]

    def generate_text_rust(self, input_text: str, max_tokens: int = 64, temperature: float = 0.0) -> Tuple[str, List[str], Dict[str, Any]]:
        if self.rust_next is None:
            return '', [], {'rust_next_available': False, 'fallback_required': True}
        text, toks, diag = self.rust_next.generate_text(str(input_text), int(max_tokens or 0), float(temperature or 0.0))
        diag = dict(diag)
        diag['rust_next_available'] = True
        diag['python_generation_loop_used'] = False
        return str(text), [str(t) for t in toks], diag


    def batch_generate_text_rust(self, inputs: Sequence[str], max_tokens: int = 64, temperature: float = 0.0) -> List[Tuple[str, List[str]]]:
        if self.rust_next is None:
            return [('', []) for _ in inputs]
        try:
            return [(str(text), [str(t) for t in toks]) for text, toks in self.rust_next.batch_generate_text([str(x) for x in inputs], int(max_tokens or 0), float(temperature or 0.0))]
        except Exception:
            return [('', []) for _ in inputs]

    def predict_token(self, rows: Sequence[Tuple[str, float]]) -> str:
        ranked = self.score_features(rows, top_k=1)
        return ranked[0][0] if ranked else ''
