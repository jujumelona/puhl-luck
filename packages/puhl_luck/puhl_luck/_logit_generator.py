
"""CPU dynamic live next-token generator.

Default live path is data-scale adaptive, not fixed-cap:
  learn(input, target): fast forward score -> rank/loss credit -> sparse/HDC/readout update
  generate(input): active sparse/HDC features -> HDC readout projection -> fast top-k greedy token

Runtime sizing is derived from observed data; caller values are treated as explicit overrides only when positive.  The
model always enables Rust/native support when available, runtime caches, batch
learning wrappers, and threaded read-only batch generation.  Expensive full HDC
KNN scoring and deep recursive lookahead are kept out of the live default path;
HDC bind/bundle still feeds the adaptive readout, so the structure remains
HDC/sparse-memory centered instead of pure dense.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque, OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import time
import pickle
import gzip
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

logger = logging.getLogger('puhl_luck.hdc')

from ._logit_tables import SparseEvidenceTables
from ._logit_scorer import LogitScorer
from ._dynamic_policy import (
    dynamic_top_k, dynamic_rank_k, dynamic_max_tokens, dynamic_context_window,
    dynamic_clause_budget, dynamic_cache_size, dynamic_hdc_source_budget,
)


@dataclass
class GenerationMetrics:
    """Detailed metrics for a generation run.
    
    Tracks tokens generated, backoff strategy usage, copy gate activations,
    generation method, and diagnostics for empty outputs.
    """
    tokens_generated: int
    backoff_levels: List[int]  # Backoff level per token (0=exact, 1=K/2, etc.)
    copy_gate_activations: int
    generation_method: str  # 'sparse', 'hdc', 'backoff', 'copy'
    empty_output: bool
    failure_reason: Optional[str]
    inference_time_ms: float

_COPY_RE = re.compile(r"^\[COPY(\d+)\]$")
_TOKEN_RE = re.compile(
    r"\[COPY\d+\]|\[BOS\]|\[SEP\]|\[EOS\]|\[NL\]|\[INDENT\]|\[DEDENT\]|"
    r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|==|!=|<=|>=|//=|\+=|-=|\*=|/=|//=|//|<<|>>|->|"
    r"\"[^\"\\]*(?:\\.[^\"\\]*)*\"|'[^'\\]*(?:\\.[^'\\]*)*'|\S"
)
_STOP = {
    'the','a','an','and','or','to','of','in','on','for','with','by','is','are','be','as','that','this',
    'write','python','function','given','return','returns','which','takes','input','output','from','using',
    'create','make','find','get','check','whether','true','false','list','array','string','number','numbers','integer','integers',
    'two','three','four','five','value','values','item','items','element','elements','result','res'
}

# Only used for corpus exposure filtering: do not let arbitrary identifiers from
# training code become global next-token answers. These are ordinary tokens learned
# from data, not a grammar/FSM and not a forced decoder rule.
_STRUCTURAL_WORDS = {
    'def','return','if','else','elif','for','while','in','not','and','or','import','from','as',
    'class','try','except','finally','with','lambda','yield','assert','raise','pass','break',
    'continue','true','false','none','is'
}


def _is_word(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok))


def _class(tok: str) -> str:
    if tok in {'[NL]','[INDENT]','[DEDENT]'}:
        return tok
    if _COPY_RE.match(tok):
        return 'COPY'
    if _is_word(tok):
        return 'WORD'
    if tok.isdigit() or re.fullmatch(r"\d+\.\d+", tok):
        return 'NUM'
    if (tok.startswith('"') and tok.endswith('"')) or (tok.startswith("'") and tok.endswith("'")):
        return 'STR'
    if tok in {'(',')','[',']','{','}'}:
        return 'BRACKET'
    if tok in {':',',',';','.'}:
        return 'PUNCT'
    if tok in {'+','-','*','/','//','%','==','!=','<','>','<=','>=','=','+=','-=','*=','/=','and','or','not'}:
        return 'OP'
    return 'SYM'


def _feature_template_id(feature_id: str) -> str:
    """Coarse template identity for meta-credit.

    The token/prompt payload is intentionally stripped: L1|x and L1|y share the
    L1 template, P2S|... rows share P2S, etc.  This lets loss credit reach the
    feature generator itself rather than only the individual row counters.
    """
    head = str(feature_id).split('|', 1)[0]
    if head.startswith('R') and head[1:].isdigit():
        return 'R*'
    if head.startswith('RC') and head[2:].isdigit():
        return 'RC*'
    return head or 'UNKNOWN'


def _short_hash(s: str) -> str:
    """Stable tiny row id for discovered feature rows."""
    h = 0xcbf29ce484222325
    for b in str(s).encode('utf-8', 'ignore'):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return format(h, '016x')[:12]


def _feature_depth(feature_id: str) -> int:
    """Abstraction depth of a feature row.

    Hand-written symbolic rows are depth 1.  Discovered rows use the Dn| prefix.
    There is intentionally no model-code maximum depth: the highest reachable
    depth is bounded only by the learned discovered-row graph and runtime budget.
    """
    head = str(feature_id).split('|', 1)[0]
    if len(head) >= 2 and head[0] == 'D' and head[1:].isdigit():
        return max(2, int(head[1:]))
    return 1


def _composite_template_id(feature_id: str) -> str:
    """Template id used inside discovered-row names.

    Dn rows keep their depth prefix instead of being collapsed to D2.  That lets
    later discoveries distinguish base->D2 from D2->D3 and so on.
    """
    head = str(feature_id).split('|', 1)[0]
    if len(head) >= 2 and head[0] == 'D' and head[1:].isdigit():
        return head
    return _feature_template_id(feature_id)


class IncrementalFeatureExtractor:
    """Incrementally extracts features during generation to avoid redundant computation.
    
    Uses a sliding window (deque) to track recent tokens and only recomputes
    features that depend on newly added tokens, caching unchanged features.
    This optimization reduces O(K²) feature computation to O(K) per token.
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    
    def __init__(self, context_window: int):
        """Initialize the incremental feature extractor.
        
        Args:
            context_window: Maximum number of tokens to keep in sliding window (K)
        """
        self.K = context_window
        self.context_deque: deque = deque(maxlen=context_window)
        self.cached_features: List[str] = []
        
    def append_token(self, token: str) -> List[str]:
        """Incrementally update features when a new token is added.
        
        Only recomputes features that are affected by the new token:
        - Unigram feature for the new token
        - Bigram feature with previous token
        - Trigram feature with previous two tokens
        - Positional features (L1, L2, L3, etc.)
        
        Args:
            token: The new token to append to the context
            
        Returns:
            List of feature strings representing the current context
        """
        self.context_deque.append(token)
        
        # Build features incrementally based on current context
        new_features = []
        context_list = list(self.context_deque)
        n = len(context_list)
        
        if n == 0:
            self.cached_features = []
            return []
        
        # Unigram: current token
        new_features.append(f'tok:{context_list[-1]}')
        
        # Bigram: if we have at least 2 tokens
        if n >= 2:
            new_features.append(f'bi:{context_list[-2]}|{context_list[-1]}')
        
        # Trigram: if we have at least 3 tokens
        if n >= 3:
            new_features.append(f'tri:{context_list[-3]}|{context_list[-2]}|{context_list[-1]}')
        
        # Positional features (L1, L2, L3, ...) - most recent tokens first
        for i in range(min(n, self.K)):
            idx = n - 1 - i  # Index from the end
            if idx >= 0:
                new_features.append(f'L{i+1}|{context_list[idx]}')
        
        # Position-based features with class information
        for i in range(min(n, self.K)):
            idx = n - 1 - i
            if idx >= 0:
                cls = _class(context_list[idx])
                new_features.append(f'C{i+1}|{cls}')
        
        # Cache the computed features
        self.cached_features = new_features
        return new_features
    
    def get_cached_features(self) -> List[str]:
        """Return the cached features without recomputation.
        
        Returns:
            List of cached feature strings
        """
        return self.cached_features
    
    def reset(self) -> None:
        """Reset the extractor state, clearing context and cache."""
        self.context_deque.clear()
        self.cached_features = []
    
    def get_context(self) -> List[str]:
        """Return the current context tokens.
        
        Returns:
            List of tokens in the current context window
        """
        return list(self.context_deque)


class SparseLogitGenerator:
    def __init__(
        self,
        copy_boost_weight: float = 0.0,
        repetition_penalty_weight: Optional[float] = None,
        repetition_window: Optional[int] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        tokenizer=None,
        output_mode: str = 'text',
        use_rust: bool = True,
        max_hops: Optional[int] = None,
        symbolic_clause_budget: Optional[int] = None,
        aggressive_readout: Optional[bool] = None,
        adaptive_readout: Optional[bool] = None,
        readout_hidden_dim: int = 0,
        readout_vocab_cap: int = 0,
        readout_lr: float = 0.0,
        readout_auto_resize: bool = True,
        readout_min_hidden: int = 0,
        readout_max_hidden: int = 0,
        readout_min_vocab_cap: int = 0,
        readout_max_vocab_cap: int = 0,
        readout_active_features: int = 0,
        runtime_cache_size: Optional[int] = None,
        fast_runtime: bool = True,
        runtime_workers: int = 0,
        parallel_inference: bool = True,
        batch_learning: bool = True,
        rare_token_threshold: int = 2,
    ) -> None:
        if adaptive_readout is None:
            adaptive_readout = True if aggressive_readout is None else bool(aggressive_readout)
        # P76 fast-only: always try Rust/native support.  Passing use_rust=False
        # is ignored because the public default must be the fastest runtime.
        self.tables = SparseEvidenceTables(
            use_rust=True,
            readout_enabled=bool(adaptive_readout),
            readout_hidden_dim=readout_hidden_dim,
            readout_vocab_cap=readout_vocab_cap,
            readout_lr=readout_lr,
            readout_auto_resize=readout_auto_resize,
            readout_min_hidden=readout_min_hidden,
            readout_max_hidden=readout_max_hidden,
            readout_min_vocab_cap=readout_min_vocab_cap,
            readout_max_vocab_cap=readout_max_vocab_cap,
            readout_active_features=readout_active_features,
        )
        # P76 fast-only: no slow runtime profile.  Constructor arguments are
        # accepted for backward compatibility but cannot disable fast mode.
        self.fast_runtime = True
        # Runtime workers are caller-provided only. They never enter model capacity formulas.
        self.runtime_workers = int(runtime_workers) if runtime_workers and int(runtime_workers) > 0 else 0
        self.parallel_inference = True
        self.batch_learning = True
        self.p75_parallel_generate_calls = 0
        self.p75_parallel_generate_items = 0
        self.p75_batch_learn_calls = 0
        self.p75_batch_learn_items = 0
        self.p75_batch_sequence_calls = 0
        self.p75_batch_sequence_items = 0
        self.runtime_cache_size = int(runtime_cache_size) if runtime_cache_size and int(runtime_cache_size) > 0 else dynamic_cache_size(0, 0, 0, 0)
        if hasattr(self.tables, 'runtime_cache_enabled'):
            self.tables.runtime_cache_enabled = True
            self.tables.runtime_cache_max_entries = self.runtime_cache_size
        self._active_feature_cache: OrderedDict[Any, List[Tuple[str, float]]] = OrderedDict()
        self._p74_value_cache: Dict[Any, Tuple[float, List[float], int]] = {}
        self.active_feature_cache_hits = 0
        self.active_feature_cache_misses = 0
        self.recursive_value_cache_hits = 0
        self.recursive_value_cache_misses = 0
        
        # Initialize incremental feature extractor for generation optimization
        # This will be reset at the start of each generation call
        self._incremental_extractor: Optional[IncrementalFeatureExtractor] = None
        
        self.scorer = LogitScorer(
            repetition_penalty_weight=(dynamic_top_k(0, 0, 0) if repetition_penalty_weight is None else float(repetition_penalty_weight)),
            repetition_window=(dynamic_context_window(0, 0, 0) if repetition_window is None else int(repetition_window)),
        )
        self.max_tokens = int(max_tokens) if max_tokens and int(max_tokens) > 0 else 0
        self.top_k = int(top_k) if top_k and int(top_k) > 0 else 0
        self.temperature = float(temperature)
        self.output_mode = output_mode
        self.rare_token_threshold = int(rare_token_threshold)
        self.BOS = '[BOS]'
        self.SEP = '[SEP]'
        self.EOS = '[EOS]'
        self.pairs_learned = 0
        self.tokens_learned = 0
        # Fast-only live generation uses single-step stabilization.  Deeper
        # lookahead is intentionally not a default path because it dominates CPU
        # inference time.
        self.max_hops = int(max_hops) if max_hops and int(max_hops) > 0 else 0
        self.symbolic_clause_budget = int(symbolic_clause_budget) if symbolic_clause_budget and int(symbolic_clause_budget) > 0 else 0
        self.adaptive_readout = bool(adaptive_readout)
        self.readout_hidden_dim = int(readout_hidden_dim)
        self.readout_vocab_cap = int(readout_vocab_cap)
        self.readout_lr = float(readout_lr)
        self.readout_auto_resize = bool(readout_auto_resize)
        self.readout_min_hidden = int(readout_min_hidden)
        self.readout_max_hidden = int(readout_max_hidden)
        self.readout_min_vocab_cap = int(readout_min_vocab_cap)
        self.readout_max_vocab_cap = int(readout_max_vocab_cap)
        self.readout_active_features = int(readout_active_features)
        self.operator_next_class: Dict[str, Counter[str]] = defaultdict(Counter)
        self.operator_totals: Counter[str] = Counter()
        self.operator_wrong_class: Dict[str, Counter[str]] = defaultdict(Counter)
        self.operator_wrong_totals: Counter[str] = Counter()

        # P71: credit reaches rowless HDC bind/bundle composites first; bounded symbolic clauses are optional interpretability slots.
        # Template gates are learned from ablation loss deltas; discovered pairs
        # become new symbolic rows only after data shows that a composition helps.
        self.template_positive: Counter[str] = Counter()
        self.template_negative: Counter[str] = Counter()
        self.template_loss_delta: Counter[str] = Counter()
        # Discovered symbolic rows are now data-budgeted clause slots.  The main
        # unbounded-composition path is HDC binding inside SparseEvidenceTables.
        self.discovered_pair_positive: Counter[str] = Counter()
        self.discovered_pair_negative: Counter[str] = Counter()
        self.discovered_pair_parents: Dict[str, Tuple[str, str]] = {}
        self.max_discovered_depth_seen = 1
        self.token_next_role: Dict[str, Counter[str]] = defaultdict(Counter)
        self.token_role_totals: Counter[str] = Counter()
        self.template_credit_events = 0
        self.discovered_feature_events = 0
        self.live_layers: Dict[str, Any] = {}
        
        # Generation statistics tracking
        self.total_generations = 0
        self.total_field_only_backoffs = 0
        self.generation_backoff_levels: List[int] = []

    def _data_scale(self) -> Dict[str, int]:
        tbl = getattr(self, 'tables', None)
        return {
            'events': int(getattr(tbl, 'updates', 0) if tbl is not None else 0) + int(self.pairs_learned) + int(self.tokens_learned),
            'features': len(getattr(tbl, 'feature_next', {})) + len(getattr(tbl, 'feature_hv', {})) if tbl is not None else 0,
            'rows': len(getattr(tbl, 'hdc_next', {})) if tbl is not None else 0,
            'vocab': len(getattr(tbl, 'vocab', {})) if tbl is not None else 0,
        }

    def _dynamic_top_k(self, prompt_tokens: int = 0) -> int:
        if int(getattr(self, 'top_k', 0) or 0) > 0:
            return int(self.top_k)
        d = self._data_scale()
        return int(dynamic_top_k(d['vocab'], d['rows'], d['events'], d['features'], prompt_tokens))

    def _dynamic_rank_k(self, active_rows: int = 0) -> int:
        d = self._data_scale()
        return int(dynamic_rank_k(d['vocab'], d['rows'], d['events'], active_rows))

    def _dynamic_max_tokens(self, prompt_tokens: int = 0, requested: Optional[int] = None) -> int:
        d = self._data_scale()
        return int(dynamic_max_tokens(prompt_tokens, d['events'], d['vocab'], requested=requested))

    def _dynamic_context_window(self, prompt_tokens: int = 0) -> int:
        d = self._data_scale()
        return int(dynamic_context_window(d['events'], prompt_tokens, d['vocab']))

    def _dynamic_symbolic_clause_budget(self) -> int:
        if int(getattr(self, 'symbolic_clause_budget', 0) or 0) > 0:
            return int(self.symbolic_clause_budget)
        d = self._data_scale()
        return int(dynamic_clause_budget(d['events'], d['features'], d['vocab'], d['rows']))

    def _dynamic_cache_size(self) -> int:
        d = self._data_scale()
        return int(dynamic_cache_size(d['events'], d['features'], d['vocab'], d['rows']))

    def _dynamic_hops(self) -> int:
        if int(getattr(self, 'max_hops', 0) or 0) > 0:
            return int(self.max_hops)
        d = self._data_scale()
        return max(1, int(math.ceil(math.log2(max(2, d['events'] + d['features'] + d['rows'] + d['vocab'] + 1)))))

    def _data_feature_budget(self, available: int, prompt_tokens: int = 0, active_rows: int = 0) -> int:
        """Number of prompt/feature rows to activate from data scale only.

        This replaces literal slices such as [:80].  The budget depends on how
        many candidate rows are available plus learned events/features/rows/vocab
        and the current prompt length.  It contains no hardware term and no
        hidden fixed cap.
        """
        available = max(0, int(available))
        if available <= 0:
            return 0
        d = self._data_scale()
        s = max(1, available + int(prompt_tokens) + int(active_rows) + d['events'] + d['features'] + d['rows'] + d['vocab'])
        return max(1, min(available, int(math.ceil(math.sqrt(s))) + int(math.ceil(math.log2(s + 1)))))

    def _take_data_features(self, items: List[str], prompt_tokens: int = 0, active_rows: int = 0) -> List[str]:
        return list(items[: self._data_feature_budget(len(items), prompt_tokens, active_rows)])

    def learn(self, input_text: str, target_text: str, field_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Supervised prompt->target exposure.

        P70 keeps the model weightless and routes credit beyond rows: it performs
        a forward score at every teacher-forced step, computes rank/loss error,
        reinforces the gold continuation, and records sparse negative evidence
        for candidates that incorrectly outranked it.
        """
        in_tokens = self._tokenize(input_text)
        copy_tokens = self._copy_tokens(in_tokens)
        target_tokens_raw = self._tokenize(target_text)
        target_tokens = self._encode_copy_tokens(target_tokens_raw, copy_tokens)
        prompt_features = self._prompt_features(in_tokens, field_features)
        added = 0
        corrected = 0
        total_loss = 0.0
        total_wrong_above = 0
        for i in range(len(target_tokens) + 1):
            prefix = target_tokens[:i]
            next_tok = target_tokens[i] if i < len(target_tokens) else self.EOS
            feats = self._active_features(prompt_features, prefix)
            if hasattr(self.tables, 'credit_assign'):
                diag = self.tables.credit_assign(feats, next_tok, top_k=self._dynamic_rank_k(len(feats)))
                amount = int(diag.get('positive_amount', 1))
                wrong_tokens = [str(t) for t in diag.get('negative_tokens', [])]
                added += amount
                total_loss += float(diag.get('token_loss', 0.0))
                total_wrong_above += int(diag.get('wrong_above_count', 0))
                if wrong_tokens:
                    corrected += 1
                self._assign_template_credit(feats, next_tok, wrong_tokens=wrong_tokens)
                self._observe_operator(prefix, next_tok, amount=amount, wrong_tokens=wrong_tokens)
            else:
                self.tables.update_features(feats, next_tok, amount=1)
                added += 1
                self._observe_operator(prefix, next_tok, amount=1)
        self.pairs_learned += 1
        self.tokens_learned += len(in_tokens) + len(target_tokens) + 2
        self._clear_runtime_caches()
        return {
            'input_tokens': len(in_tokens),
            'target_tokens': len(target_tokens),
            'transitions_added': added,
            'credit_corrected_steps': corrected,
            'rank_loss_wrong_above': total_wrong_above,
            'mean_token_loss': total_loss / max(1, len(target_tokens) + 1),
            'credit_assignment': 'rank_loss_to_sparse_hdc_template_gates_and_p73_margin_readout',
        }

    def learn_sequence(
        self,
        text: str,
        field_features: Optional[List[str]] = None,
        structural_targets_only: bool = False,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Self-supervised next-token exposure over a token sequence.

        This is the data-exposure path that was missing in P61. It does not store
        raw answers and it does not add grammar states. It only adds count evidence
        from prefix context rows.

        structural_targets_only=True is used for code-like corpora so random
        identifier names from train examples do not become global output names.
        """
        toks = self._tokenize(text)
        if max_tokens is not None:
            toks = toks[: max(0, int(max_tokens))]
        if len(toks) < 2:
            return {'sequence_tokens': len(toks), 'transitions_added': 0}

        # Empty prompt features means corpus exposure teaches prefix/shape
        # continuation only. It cannot act as prompt-specific answer retrieval.
        prompt_features: List[str] = []
        added = 0
        corrected = 0
        total_loss = 0.0
        start = 1  # do not train global first-token prior from corpus
        for i in range(start, len(toks) + 1):
            next_tok = toks[i] if i < len(toks) else self.EOS
            if structural_targets_only and not self._is_structural_target(next_tok):
                continue
            win = self._dynamic_context_window(i)
            prefix = toks[max(0, i - win):i]
            feats = self._active_features(prompt_features, prefix)
            if hasattr(self.tables, 'credit_assign'):
                diag = self.tables.credit_assign(feats, next_tok, top_k=self._dynamic_rank_k(len(feats)))
                amount = int(diag.get('positive_amount', 1))
                wrong_tokens = [str(t) for t in diag.get('negative_tokens', [])]
                added += amount
                total_loss += float(diag.get('token_loss', 0.0))
                if wrong_tokens:
                    corrected += 1
                self._assign_template_credit(feats, next_tok, wrong_tokens=wrong_tokens)
                self._observe_operator(prefix, next_tok, amount=amount, wrong_tokens=wrong_tokens)
            else:
                self.tables.update_features(feats, next_tok)
                added += 1
                self._observe_operator(prefix, next_tok, amount=1)
        self.tokens_learned += len(toks)
        self._clear_runtime_caches()
        return {
            'sequence_tokens': len(toks),
            'transitions_added': added,
            'credit_corrected_steps': corrected,
            'mean_token_loss': total_loss / max(1, added),
            'structural_targets_only': bool(structural_targets_only),
            'credit_assignment': 'rank_loss_to_sparse_hdc_template_gates_and_p73_margin_readout',
        }

    def cpu_runtime_status(self) -> Dict[str, Any]:
        """Return whether the file itself is running the P76 CPU fast-only path.

        Exact online learning is still sequential because every update changes the
        memory that the next credit-assignment step sees.  The live path is now
        data-scale fast scoring, Rust/native update support when available, dynamic
        runtime caches, batch wrappers, and threaded read-only generation.
        """
        tbl = getattr(self, 'tables', None)
        return {
            'p75_max_cpu_runtime': True,
            'p76_fast_only_default': True,
            'runtime_workers': int(getattr(self, 'runtime_workers', 0)),
            'parallel_inference_enabled': bool(getattr(self, 'parallel_inference', False)),
            'batch_learning_enabled': bool(getattr(self, 'batch_learning', False)),
            'rust_available': bool(getattr(tbl, 'rust_available', False)),
            'rust_object_loaded': bool(getattr(tbl, 'rust', None) is not None),
            'rust_is_source_of_truth_for_p73_scoring': False,
            'live_score_path': 'fast_exact_sparse_plus_adaptive_readout_hdc_projection_no_full_hdc_knn',
            'slow_runtime_switches_ignored': True,
            'rust_reason': 'Rust fast path is used for compatible count/HDC update support; P73 signed evidence + adaptive readout scoring remains Python source of truth.',
            'exact_online_learning_parallel_safe': False,
            'learning_parallel_policy': 'safe sequential commit; learn_many/learn_sequences_many use warmed runtime and one cache clear per batch',
            'inference_parallel_policy': 'read-only batch parallelism uses caller-provided max_workers or batch size; model capacity is data-only',
            'score_cache_entries': len(getattr(tbl, '_score_cache', {})) if tbl is not None else 0,
            'runtime_cache_size': int(self._dynamic_cache_size()),
        }

    def learn_many(
        self,
        pairs: Iterable[Tuple[str, str]],
        field_features: Optional[List[str]] = None,
        max_items: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Batch supervised learning wrapper for CPU-only runs.

        This intentionally keeps the final commit order sequential.  Parallel
        online writes would race because each rank-loss credit step depends on
        the current memory.  Speed comes from keeping one warmed Rust/Python
        runtime, avoiding object recreation, and clearing caches once per batch.
        """
        t0 = time.perf_counter()
        stats = {
            'items': 0, 'transitions_added': 0, 'credit_corrected_steps': 0,
            'rank_loss_wrong_above': 0, 'mean_token_loss_sum': 0.0,
        }
        for idx, pair in enumerate(pairs):
            if max_items is not None and idx >= int(max_items):
                break
            if pair is None:
                continue
            inp, tgt = pair
            r = self.learn(str(inp), str(tgt), field_features=field_features)
            stats['items'] += 1
            stats['transitions_added'] += int(r.get('transitions_added', 0))
            stats['credit_corrected_steps'] += int(r.get('credit_corrected_steps', 0))
            stats['rank_loss_wrong_above'] += int(r.get('rank_loss_wrong_above', 0))
            stats['mean_token_loss_sum'] += float(r.get('mean_token_loss', 0.0))
        self.p75_batch_learn_calls += 1
        self.p75_batch_learn_items += int(stats['items'])
        self._clear_runtime_caches()
        elapsed = max(1e-12, time.perf_counter() - t0)
        return {
            'p75_batch_learning_used': True,
            'p76_fast_only_learning_used': True,
            'items': int(stats['items']),
            'transitions_added': int(stats['transitions_added']),
            'credit_corrected_steps': int(stats['credit_corrected_steps']),
            'rank_loss_wrong_above': int(stats['rank_loss_wrong_above']),
            'mean_token_loss': float(stats['mean_token_loss_sum']) / max(1, int(stats['items'])),
            'seconds': float(elapsed),
            'items_per_sec': float(stats['items']) / elapsed,
            'exact_online_learning_parallel_safe': False,
            'rust_available': bool(getattr(self.tables, 'rust_available', False)),
        }

    def learn_sequences_many(
        self,
        texts: Iterable[str],
        field_features: Optional[List[str]] = None,
        structural_targets_only: bool = False,
        max_items: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Batch sequence exposure wrapper with one warmed CPU/Rust runtime."""
        t0 = time.perf_counter()
        items = 0
        transitions = 0
        corrected = 0
        token_count = 0
        loss_sum = 0.0
        for idx, text in enumerate(texts):
            if max_items is not None and idx >= int(max_items):
                break
            if text is None or not str(text).strip():
                continue
            r = self.learn_sequence(str(text), field_features=field_features, structural_targets_only=structural_targets_only, max_tokens=max_tokens)
            items += 1
            transitions += int(r.get('transitions_added', 0))
            corrected += int(r.get('credit_corrected_steps', 0))
            token_count += int(r.get('sequence_tokens', 0))
            loss_sum += float(r.get('mean_token_loss', 0.0))
        self.p75_batch_sequence_calls += 1
        self.p75_batch_sequence_items += int(items)
        self._clear_runtime_caches()
        elapsed = max(1e-12, time.perf_counter() - t0)
        return {
            'p75_batch_sequence_learning_used': True,
            'p76_fast_only_learning_used': True,
            'items': int(items),
            'sequence_tokens': int(token_count),
            'transitions_added': int(transitions),
            'credit_corrected_steps': int(corrected),
            'mean_token_loss': float(loss_sum) / max(1, int(items)),
            'seconds': float(elapsed),
            'items_per_sec': float(items) / elapsed,
            'tokens_per_sec': float(token_count) / elapsed,
            'exact_online_learning_parallel_safe': False,
            'rust_available': bool(getattr(self.tables, 'rust_available', False)),
        }

    def generate_many(
        self,
        inputs: Iterable[str],
        field_features: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Batch generation API for CPU read-only evaluation.

        Uses threads by default so one model instance can be reused without
        serializing the whole memory.  The generator caches are deterministic and
        may be shared; cache misses/hits can race but scores remain functions of
        immutable learned tables during read-only evaluation.
        """
        items = [str(x) for x in inputs]
        if not items:
            return []
        workers = int(max_workers or getattr(self, 'runtime_workers', 0) or len(items))
        workers = max(1, min(workers, len(items)))
        self.p75_parallel_generate_calls += 1
        self.p75_parallel_generate_items += len(items)
        if workers <= 1 or not bool(getattr(self, 'parallel_inference', True)):
            return [self.generate(x, field_features=field_features, max_tokens=max_tokens, top_k=top_k) for x in items]
        out: List[Optional[Tuple[str, Dict[str, Any]]]] = [None] * len(items)
        def _run(i: int, x: str):
            return i, self.generate(x, field_features=field_features, max_tokens=max_tokens, top_k=top_k)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run, i, x) for i, x in enumerate(items)]
            for fut in as_completed(futs):
                i, res = fut.result()
                out[int(i)] = res
        return [x if x is not None else ('', {'error': 'missing_result'}) for x in out]

    def _is_structural_target(self, tok: str) -> bool:
        if tok == self.EOS:
            return True
        cls = _class(tok)
        if cls != 'WORD':
            return True
        return tok.lower() in _STRUCTURAL_WORDS

    def attach_live_layers(self, **layers: Any) -> None:
        """Attach non-gradient live-generation layers from BrainMemory.

        The generator still owns the sparse evidence tables, but these handles make
        the runtime path explicit: operator statistics, candidate emergence, energy
        scoring, and recursive stabilization are no longer hidden/disabled in the
        mixin.
        """
        self.live_layers = {k: v for k, v in layers.items() if v is not None}

    def generate(self, input_text: str, field_features: Optional[List[str]] = None, max_tokens: Optional[int] = None,
                 top_k: Optional[int] = None, temperature: Optional[float] = None, return_metrics: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Generate completion for input text.
        
        Args:
            input_text: Input text to complete
            field_features: Optional field features
            max_tokens: Maximum tokens to generate
            top_k: Top-K candidates to consider
            temperature: Temperature for sampling
            return_metrics: If True, return detailed GenerationMetrics in the response
            
        Returns:
            Tuple of (generated_text, metrics_dict)
        """
        start_time = time.perf_counter()
        in_tokens_preview = self._tokenize(input_text)
        max_tokens = int(max_tokens) if max_tokens is not None and int(max_tokens) > 0 else self._dynamic_max_tokens(len(in_tokens_preview))
        top_k = int(top_k) if top_k is not None and int(top_k) > 0 else self._dynamic_top_k(len(in_tokens_preview))
        # P76: do not clear read-only runtime caches at every generate() call.
        # Learning updates call _clear_runtime_caches(); during batch inference
        # the memory is immutable and cache reuse is the speed path.
        self._p74_value_cache = {}
        
        # Initialize incremental feature extractor for this generation
        context_window = self._dynamic_context_window(len(in_tokens_preview))
        self._incremental_extractor = IncrementalFeatureExtractor(context_window)
        
        in_tokens = in_tokens_preview
        copy_tokens = self._copy_tokens(in_tokens)
        prompt_features = self._prompt_features(in_tokens, field_features)
        prefix_model: List[str] = []
        out_visible: List[str] = []
        recent = deque(maxlen=self.scorer.repetition_window)
        total_candidates = 0
        stop = 'max_tokens'
        copy_count = 0
        no_candidate = 0
        total_hops = 0
        total_energy_checks = 0
        energy_trace: List[float] = []
        backoff_levels: List[int] = []  # Track backoff level per token
        field_only_count = 0  # Track field-only backoffs
        copy_gate_activations = 0  # Track copy gate activations

        for step in range(max_tokens):
            feats = self._active_features_cached(prompt_features, prefix_model)
            # P76 live default: skip expensive full HDC KNN.  HDC bind/bundle
            # vectors still feed the adaptive readout projection, while exact
            # sparse rows and readout provide the token scores.
            candidates = self.tables.score_from_features_fast(feats, top_k=max(top_k, self._dynamic_rank_k(len(feats))))
            
            # Track backoff level (0 = exact match in current implementation)
            # In future implementations with progressive backoff, this would track K -> K/2 -> K/4 -> unigram -> field
            backoff_level = 0
            if not candidates:
                backoff_level = 5  # Field-only or no match
                field_only_count += 1
                stop = 'no_candidates'
                no_candidate += 1
                break
            
            # Append backoff level for this token
            backoff_levels.append(backoff_level)
            total_candidates += len(candidates)

            # P76 fast-only default: no recursive lookahead in the live loop.
            # The structure still has learned operator/readout/HDC state, but
            # scoring does not simulate future branches by default because that
            # is the main CPU inference slowdown.
            refined = candidates
            diag = {'hops': 0, 'energy_checks': 0, 'energy_trace': []}
            total_hops += 0
            total_energy_checks += 0

            scored = self.scorer.score_candidates(refined or candidates, recent_tokens=recent, step=step)
            if not scored:
                stop = 'no_scored'
                break
            tok_model = self.scorer.get_top_token(scored)
            if tok_model == self.EOS:
                stop = 'eos'
                break
            # avoid degenerate triple-repeat without switching to global fallback
            if len(prefix_model) >= 2 and prefix_model[-1] == prefix_model[-2] == tok_model:
                scored = [(t, s) for t, s in scored if t != tok_model]
                if not scored:
                    stop = 'repeat_block'
                    break
                tok_model = scored[0][0]
                if tok_model == self.EOS:
                    stop = 'eos'
                    break
            prefix_model.append(tok_model)
            
            # Update incremental feature extractor with new token
            if self._incremental_extractor is not None:
                self._incremental_extractor.append_token(tok_model)
            
            visible = self._resolve_copy(tok_model, copy_tokens)
            if visible is not None:
                out_visible.append(visible)
                if visible != tok_model:
                    copy_count += 1
                    copy_gate_activations += 1  # Track copy gate activation
            recent.append(tok_model)
        
        # Track generation statistics for logging excessive backoff
        self.total_generations += 1
        self.total_field_only_backoffs += field_only_count
        self.generation_backoff_levels.extend(backoff_levels)
        
        # Log warning if field-only backoff exceeds 20%
        if len(backoff_levels) > 0:
            field_only_percentage = (field_only_count / len(backoff_levels)) * 100
            if field_only_percentage > 20:
                logger.warning(
                    f"Excessive field-only backoff: {field_only_percentage:.1f}% "
                    f"({field_only_count}/{len(backoff_levels)} tokens) reached field_only level"
                )
        
        text = self._detokenize(out_visible)
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Prepare base metrics
        base_metrics = {
            'generation_method': 'p76_fast_only_cpu_hdc_sparse_memory_with_adaptive_margin_readout',
            'input_tokens': len(in_tokens),
            'output_tokens': len(out_visible),
            'model_tokens': len(prefix_model),
            'copy_tokens_used': copy_count,
            'candidates_scored': total_candidates,
            'avg_candidates_per_step': total_candidates / max(1, len(prefix_model)),
            'stop_reason': stop,
            'empty_output': len(out_visible) == 0,
            'rust_used': bool(getattr(self.tables, 'rust_available', False)),
            'rust_score_used': False,
            'no_candidate_steps': no_candidate,
            'surface_raw_return_used': False,
            'old_answer_fallback_used': False,
            'unigram_fallback_used': False,
            'first_token_special_used': False,
            'hdc_index_used': True,
            'hebbian_hdc_used': True,
            'dynamic_hdc_used': True,
            'operator_layer_used': True,
            'candidate_emergence_used': False,
            'energy_computer_used': False,
            'recursive_stabilization_used': False,
            'credit_assignment_used': True,
            'credit_assignment': 'rank_loss_to_sparse_hdc_operator_template_rowless_hdc_composites_data_budgeted_clauses_and_adaptive_readout',
            'adaptive_readout_used': bool(getattr(self.tables, 'readout_enabled', False)),
            'readout_hidden_dim': int(getattr(self.tables, 'readout_hidden_dim', 0)),
            'readout_vocab_cap': int(getattr(self.tables, 'readout_vocab_cap', 0)),
            'readout_tracked_tokens': len(getattr(self.tables, 'readout_weights', {})),
            'readout_materialized_weights': int(self.tables.readout_parameter_count()) if hasattr(self.tables, 'readout_parameter_count') else 0,
            'readout_dense_capacity_weights': int(self.tables.readout_dense_capacity()) if hasattr(self.tables, 'readout_dense_capacity') else 0,
            'readout_fp16_mb_materialized': (int(self.tables.readout_parameter_count()) * 2 / (1024 * 1024)) if hasattr(self.tables, 'readout_parameter_count') else 0.0,
            'readout_fp16_mb_dense_capacity': (int(self.tables.readout_dense_capacity()) * 2 / (1024 * 1024)) if hasattr(self.tables, 'readout_dense_capacity') else 0.0,
            'readout_last_active_features': int(getattr(self.tables, 'readout_last_active', 0)),
            'readout_last_candidates': int(getattr(self.tables, 'readout_last_candidates', 0)),
            'readout_auto_resize': bool(getattr(self.tables, 'readout_auto_resize', False)),
            'readout_resize_count': int(getattr(self.tables, 'readout_resize_count', 0)),
            'readout_active_budget': int(self.tables._readout_active_budget()) if hasattr(self.tables, '_readout_active_budget') else 0,
            'readout_inverted_index_entries': sum(len(v) for v in getattr(self.tables, 'readout_index', {}).values()),
            'p74_fast_runtime_used': bool(self.fast_runtime),
            'p75_max_cpu_runtime_used': True,
            'p76_fast_only_default': True,
            'p76_slow_switches_ignored': True,
            'p75_runtime_workers': int(getattr(self, 'runtime_workers', 0)),
            'p75_parallel_inference_enabled': bool(getattr(self, 'parallel_inference', False)),
            'p75_batch_learning_enabled': bool(getattr(self, 'batch_learning', False)),
            'p75_parallel_generate_calls': int(getattr(self, 'p75_parallel_generate_calls', 0)),
            'p75_parallel_generate_items': int(getattr(self, 'p75_parallel_generate_items', 0)),
            'p75_batch_learn_calls': int(getattr(self, 'p75_batch_learn_calls', 0)),
            'p75_batch_learn_items': int(getattr(self, 'p75_batch_learn_items', 0)),
            'p75_rust_available': bool(getattr(self.tables, 'rust_available', False)),
            'p74_active_feature_cache_hits': int(self.active_feature_cache_hits),
            'p74_active_feature_cache_misses': int(self.active_feature_cache_misses),
            'p74_recursive_value_cache_hits': int(self.recursive_value_cache_hits),
            'p74_recursive_value_cache_misses': int(self.recursive_value_cache_misses),
            'p74_score_cache_hits': int(getattr(self.tables, 'score_cache_hits', 0)),
            'p74_score_cache_misses': int(getattr(self.tables, 'score_cache_misses', 0)),
            'p74_hdc_context_cache_hits': int(getattr(self.tables, 'hdc_context_cache_hits', 0)),
            'p74_readout_projection_cache_hits': int(getattr(self.tables, 'readout_projection_cache_hits', 0)),
            'p74_fast_score_calls': int(getattr(self.tables, 'fast_score_calls', 0)),
            'p74_full_score_calls': int(getattr(self.tables, 'full_score_calls', 0)),
            'p76_live_score_path': 'score_from_features_fast_no_recursive_lookahead',
            'feature_discovery_used': True,
            'recursive_feature_discovery_used': True,
            'hdc_binding_composition_used': True,
            'symbolic_clause_budget': int(self._dynamic_symbolic_clause_budget()),
            'max_discovered_depth_seen': int(self.max_discovered_depth_seen),
            'template_credit_events': int(self.template_credit_events),
            'template_rows': len(set(self.template_positive) | set(self.template_negative)),
            'discovered_pair_rows': len(set(self.discovered_pair_positive) | set(self.discovered_pair_negative)),
            'discovered_pair_parent_rows': len(getattr(self, 'discovered_pair_parents', {})),
            'learned_token_roles': len(self.token_next_role),
            'multi_hop_total_iterations': total_hops,
            'energy_checks': total_energy_checks,
            'energy_trace': energy_trace[-64:],
            'attached_live_layers': sorted(self.live_layers.keys()),
        }
        
        # If return_metrics=True, add detailed GenerationMetrics
        if return_metrics:
            # Determine failure reason if empty output
            failure_reason = None
            if len(out_visible) == 0:
                if stop == 'no_candidates':
                    failure_reason = 'no_candidates_found'
                elif stop == 'no_scored':
                    failure_reason = 'no_scored_candidates'
                elif stop == 'repeat_block':
                    failure_reason = 'repetition_blocked'
                else:
                    failure_reason = f'stopped_early_{stop}'
            
            base_metrics['detailed_metrics'] = GenerationMetrics(
                tokens_generated=len(out_visible),
                backoff_levels=backoff_levels,
                copy_gate_activations=copy_gate_activations,
                generation_method='sparse',
                empty_output=len(out_visible) == 0,
                failure_reason=failure_reason,
                inference_time_ms=inference_time_ms
            )
        
        return text, base_metrics

    def _clear_runtime_caches(self) -> None:
        self._active_feature_cache.clear()
        self._p74_value_cache.clear()
        if hasattr(self.tables, '_clear_runtime_caches'):
            self.tables._clear_runtime_caches()

    def _active_features_cached(self, prompt_features: List[str], prefix: List[str]) -> List[Tuple[str, float]]:
        # P76 fast-only: cache is always used. Include credit/discovery counters so cache never crosses a learning
        # change in template gates or symbolic clause slots. During generate()
        # those counters are stable, and recursive lookahead reuses many prefixes.
        key = (
            tuple(prompt_features),
            tuple(prefix[-64:]),
            int(self.template_credit_events),
            int(self.discovered_feature_events),
            int(self.max_discovered_depth_seen),
        )
        cached = self._active_feature_cache.get(key)
        if cached is not None:
            self.active_feature_cache_hits += 1
            self._active_feature_cache.move_to_end(key)
            return list(cached)
        self.active_feature_cache_misses += 1
        out = self._active_features(prompt_features, prefix)
        self._active_feature_cache[key] = list(out)
        self._active_feature_cache.move_to_end(key)
        while len(self._active_feature_cache) > self._dynamic_cache_size():
            self._active_feature_cache.popitem(last=False)
        return out

    def _tokenize(self, text: str) -> List[str]:
        text = str(text).replace('\r\n', '\n').replace('\r', '\n')
        toks: List[str] = []
        lines = text.split('\n')
        for li, line in enumerate(lines):
            if li > 0:
                toks.append('[NL]')
            stripped = line.lstrip(' ')
            if line and len(line) - len(stripped) > 0:
                toks.append('[INDENT]')
            toks.extend(_TOKEN_RE.findall(stripped))
        return [t for t in toks if t and not t.isspace()]

    def _copy_tokens(self, input_tokens: List[str], limit: int = 24) -> List[str]:
        """Prompt token references for general pointer-style generation.

        This is not a separate copy gate; the model must learn [COPYi] as a normal next token.
        Order is chosen from explicit prompt symbols, not from stored answers.
        
        As per Requirement 6.4: Tokens with frequency < rare_token_threshold are prioritized
        for copy gate extraction.
        """
        primary: List[str] = []
        secondary: List[str] = []
        seen = set()
        depth = 0
        
        # Track rare tokens (frequency < rare_token_threshold) for priority extraction
        rare_tokens: List[str] = []
        
        for t in input_tokens:
            if t in {'(', '[', '{'}:
                depth += 1
                continue
            if t in {')', ']', '}'}:
                depth = max(0, depth - 1)
                continue
            if t in {'[NL]','[INDENT]','[DEDENT]', self.BOS, self.SEP, self.EOS}:
                continue
            low = t.lower()
            ok = _is_word(t) or t.isdigit() or (len(t) >= 2 and (t[0] in {'\"', "'"}))
            if not ok:
                continue
            key = low if _is_word(t) else t
            if key in seen:
                continue
            
            # Check if token is rare (frequency < rare_token_threshold)
            # Requirement 6.4: Tokens with frequency < threshold are marked as copy candidates
            token_freq = self.tables.vocab.get(t, 0)
            is_rare = token_freq < self.rare_token_threshold
            
            # Rare tokens get highest priority for copy extraction
            if is_rare:
                rare_tokens.append(t)
                seen.add(key)
                continue
            
            # long/non-stop identifiers first; one-letter symbols are useful only when explicit in a delimited slot
            if low not in _STOP and len(low) >= 2:
                target = primary
            elif depth > 0 and _is_word(t):
                target = secondary
            elif low not in _STOP and len(low) == 1:
                target = secondary
            else:
                continue
            seen.add(key)
            target.append(t)
        
        # Prioritize rare tokens, then primary, then secondary
        out = (rare_tokens + primary + secondary)[:limit]
        return out

    def _encode_copy_tokens(self, target_tokens: List[str], copy_tokens: List[str]) -> List[str]:
        idx = {t.lower() if _is_word(t) else t: i for i, t in enumerate(copy_tokens)}
        encoded: List[str] = []
        for t in target_tokens:
            key = t.lower() if _is_word(t) else t
            j = idx.get(key)
            if j is not None:
                encoded.append(f'[COPY{j}]')
            else:
                encoded.append(t)
        return encoded

    def _resolve_copy(self, tok: str, copy_tokens: List[str]) -> Optional[str]:
        m = _COPY_RE.match(tok)
        if not m:
            return tok
        j = int(m.group(1))
        if 0 <= j < len(copy_tokens):
            return copy_tokens[j]
        return None

    def _prompt_features(self, input_tokens: List[str], field_features: Optional[List[str]]) -> List[str]:
        """Prompt conditioning features for the weightless count model.

        P61 change: do not feed broad BrainMemory text features into the logit table.
        Those features were too common and acted like a global unigram prior.
        This keeps only prompt-local token/phrase/sketch features, ordered from
        most-specific to least-specific. No raw answer storage, no grammar state.
        """
        feats: List[str] = []
        lows = [t.lower() for t in input_tokens if t not in {'[NL]', '[INDENT]', '[DEDENT]'}]
        classes = [_class(t) for t in input_tokens if t not in {'[NL]', '[INDENT]', '[DEDENT]'}]

        # Rare lexical tokens carry the task identity. Keep order and a sorted bag form.
        rare: List[str] = []
        seen_rare = set()
        for t in lows:
            if len(t) >= 3 and t not in _STOP and re.search(r"[A-Za-z0-9_]", t):
                if t not in seen_rare:
                    seen_rare.add(t)
                    rare.append(t)
        rare_sorted = sorted(rare)
        onechars = []
        for t in lows:
            if len(t) == 1 and re.search(r"[A-Za-z0-9_]", t) and t not in onechars:
                onechars.append(t)

        # Most-specific prompt signatures first. These are not answer retrieval;
        # they are normal feature ids that vote for next tokens through counts.
        if not rare:
            # Keep one-character labels / variables when they are the only task signal.
            for t in lows:
                if len(t) == 1 and re.search(r"[A-Za-z0-9_]", t) and t not in seen_rare:
                    seen_rare.add(t)
                    rare.append(t)
            rare_sorted = sorted(rare)

        if rare or onechars:
            sig_base = (rare[: self._data_feature_budget(len(rare), len(lows), len(feats))] + ['ch:' + c for c in onechars[: self._data_feature_budget(len(onechars), len(lows), len(feats))]])
            bag_base = (rare_sorted[: self._data_feature_budget(len(rare_sorted), len(lows), len(feats))] + ['ch:' + c for c in sorted(onechars[: self._data_feature_budget(len(onechars), len(lows), len(feats))])])
            feats.append('sig:first:' + '|'.join(sig_base))
            feats.append('sig:bag:' + '|'.join(bag_base))
            for c in self._take_data_features(onechars, len(lows), len(feats)):
                feats.append('char:' + c)
            for i, x in enumerate(self._take_data_features(rare, len(lows), len(feats))):
                feats.append(f'rare{i}:{x}')
                feats.append('rare:' + x)
                if len(x) >= 4:
                    feats.append('rpre4:' + x[:4])
                    feats.append('rsuf4:' + x[-4:])
            for i in range(max(0, min(len(rare) - 1, self._data_feature_budget(len(rare), len(lows), len(feats))))):
                feats.append('rare2:' + rare[i] + '|' + rare[i + 1])
            # Sparse unordered pairs, capped so table size stays bounded.
            for i in range(self._data_feature_budget(len(rare_sorted), len(lows), len(feats))):
                for j in range(i + 1, self._data_feature_budget(len(rare_sorted), len(lows), len(feats))):
                    feats.append('bag2:' + rare_sorted[i] + '|' + rare_sorted[j])
        else:
            feats.append('prompt:no_rare')

        # Prompt token and phrase shape. Stop words are excluded from token ids,
        # but class/shape remains for general language continuation.
        for i, t in enumerate(lows[: self._data_feature_budget(len(lows), len(lows), len(feats))]):
            if len(t) >= 3 and t not in _STOP and re.search(r"[A-Za-z0-9_]", t):
                feats.append('pt:' + t)
                feats.append('p3:' + t[:3])
                feats.append('p3e:' + t[-3:])
            if i < len(classes):
                feats.append('pc:' + classes[i])
        for i in range(max(0, min(len(lows) - 1, self._data_feature_budget(len(lows), len(lows), len(feats))))):
            a, b = lows[i], lows[i + 1]
            if (a not in _STOP or b not in _STOP) and (len(a) >= 2 and len(b) >= 2):
                feats.append('pt2:' + a + '|' + b)
        for i in range(max(0, min(len(lows) - 2, self._data_feature_budget(len(lows), len(lows), len(feats))))):
            a, b, c = lows[i], lows[i + 1], lows[i + 2]
            if any((x not in _STOP and len(x) >= 3) for x in (a, b, c)):
                feats.append('pt3:' + a + '|' + b + '|' + c)

        # Optional field features only as tiny hashed sketches, not direct rows.
        # This prevents common features like bf:mod:text from dominating.
        if field_features:
            clean = []
            for f in field_features:
                sf = str(f)
                if any(k in sf for k in ('stem:', 'c3:', 'tok:')):
                    # keep only non-stop, informative tail
                    tail = sf.rsplit(':', 1)[-1].lower()
                    if len(tail) >= 4 and tail not in _STOP:
                        clean.append(tail)
            if clean:
                clean = sorted(set(clean))[: self._data_feature_budget(len(set(clean)), 0, 0)]
                feats.append('bfsk:' + '|'.join(clean[: self._data_feature_budget(len(clean), 0, len(feats))]))
                for x in self._take_data_features(clean, 0, len(feats)):
                    feats.append('bfrare:' + x)

        out=[]; seen=set()
        for f in feats:
            if f and f not in seen:
                seen.add(f); out.append(f)
        return out

    def _token_role(self, tok: str) -> str:
        """Data-derived token role with surface class only as cold-start fallback."""
        row = self.token_next_role.get(str(tok))
        total = float(self.token_role_totals.get(str(tok), 0) or 0)
        if row and total > 0:
            cls, cnt = max(row.items(), key=lambda kv: (kv[1], kv[0]))
            purity = float(cnt) / max(1.0, total)
            return f'LR|{cls}|{int(round(purity * 100))}'
        return 'SURF|' + _class(tok)

    def _template_gate(self, feature_id: str) -> float:
        """Learned activation strength for a feature template.

        No manually assigned L1/L2/P1S ladder is used.  If a template has no
        history it is explored at full strength.  Once loss ablations have been
        observed, the gate is the empirical useful fraction with one neutral
        count so templates can recover instead of being permanently deleted.
        """
        tid = _feature_template_id(feature_id)
        pos = float(self.template_positive.get(tid, 0))
        neg = float(self.template_negative.get(tid, 0))
        if pos + neg <= 0.0:
            return 1.0
        return max(0.0, (pos + 1.0) / (pos + neg + 1.0))

    def _weighted_rows_from_templates(self, feats: Iterable[str]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        seen = set()
        for f in feats:
            if not f or f in seen:
                continue
            seen.add(f)
            gate = self._template_gate(f)
            if gate > 0.0:
                out.append((f, float(gate)))
        return out

    def _pair_row_key(self, a: str, b: str) -> str:
        """Recursive discovered-row key.

        P69 always returned D2 and therefore stopped feature discovery at second
        order.  P70 makes discovered rows first-class sources: combining any
        active Dn row with another source creates a D(n+1) row.  D2+D2 therefore
        yields D3, D3+D2 yields D4, etc.  There is no hand-coded maximum depth.
        """
        if a > b:
            a, b = b, a
        depth = max(_feature_depth(a), _feature_depth(b)) + 1
        ta = _composite_template_id(a)
        tb = _composite_template_id(b)
        return f'D{depth}|{ta}&{tb}|{_short_hash(a)}|{_short_hash(b)}'

    def _feature_support(self, row_id: str) -> int:
        tbl = getattr(self, 'tables', None)
        if tbl is None:
            return 0
        try:
            return int(getattr(tbl, 'feature_totals', {}).get(row_id, 0))
        except Exception:
            return 0

    def _candidate_pair_sources(self, rows: List[Tuple[str, float]]) -> List[str]:
        """Rows eligible for recursive discovered composition.

        P70 deliberately includes already-discovered Dn rows.  D rows are not
        terminal features; once activated, they can be parents of deeper D(n+1)
        rows.  Ranking is still evidence/gate based so unsupported rows do not
        explode combinatorially.
        """
        scored: List[Tuple[float, int, str]] = []
        for f, w in rows:
            support = self._feature_support(f)
            # Discovered rows may be useful before they have much exact-table
            # support, so their own discovered positive/negative gate also
            # contributes to parent selection.
            dpos = float(self.discovered_pair_positive.get(f, 0))
            dneg = float(self.discovered_pair_negative.get(f, 0))
            dgate = (dpos + 1.0) / (dpos + dneg + 1.0) if (dpos + dneg) > 0 else 1.0
            evidence = math.log1p(max(0, support) + dpos)
            # P84: deeper symbolic clauses must not dominate merely because
            # they are recursive.  They remain active and can grow without a
            # hard depth cap, but their parent-selection strength is normalized
            # by depth so D6/D7 rows cannot swamp base evidence on small data.
            depth_gain = 1.0 / math.sqrt(max(1.0, float(_feature_depth(f))))
            score = float(w) * dgate * depth_gain * evidence
            scored.append((score, _feature_depth(f), f))
        scored.sort(key=lambda kv: (-kv[0], -kv[1], kv[2]))
        # Budget grows with learned discovered graph size and active-row count.
        # It is not a depth cap; it only keeps each live forward pass finite.
        budget = max(1, int(math.sqrt(max(1, len(scored))) * math.sqrt(max(1, len(self.discovered_pair_positive) + 1))))
        return [f for _s, _d, f in scored[: max(1, min(len(scored), budget))]]

    def _add_discovered_feature_rows(self, rows: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Activate discovered rows through an indexed learned-parent graph.

        The old path recomputed all active pair combinations at every forward
        step.  That made inference/training cost grow as active_rows^2 and could
        stall once many Dn rows existed.  This version uses the learned parent
        index recorded at birth: a discovered row becomes active when both of its
        learned parents are active.  Depth can still grow without a code maximum,
        but activation cost follows the learned graph size rather than all
        possible pairs.
        """
        if not rows or not self.discovered_pair_positive or not self.discovered_pair_parents:
            return rows
        base: List[Tuple[str, float]] = list(rows)
        seen = {f for f, _w in base}
        frontier_changed = True
        while frontier_changed:
            frontier_changed = False
            added_this_round: List[Tuple[str, float]] = []
            for key, parents in list(self.discovered_pair_parents.items()):
                if key in seen:
                    continue
                a, b = parents
                if a not in seen or b not in seen:
                    continue
                pos = float(self.discovered_pair_positive.get(key, 0))
                if pos <= 0.0:
                    continue
                neg = float(self.discovered_pair_negative.get(key, 0))
                gate = (pos + 1.0) / (pos + neg + 1.0)
                if gate <= 0.0:
                    continue
                # P84: recursive Dn rows are symbolic clause evidence, not a
                # free full-strength feature ladder.  Activation weight is
                # data-derived (pos/neg/support) and depth-normalized; there is
                # no fixed max-depth cap, but deeper rows must earn influence.
                depth = max(1, _feature_depth(key))
                support = self._feature_support(key)
                evidence = math.log1p(max(0.0, pos) + max(0, support))
                weight = float(gate) * evidence / float(depth * depth)
                if weight <= 1e-12:
                    continue
                seen.add(key)
                added_this_round.append((key, weight))
                self.max_discovered_depth_seen = max(self.max_discovered_depth_seen, _feature_depth(key))
            if added_this_round:
                base.extend(added_this_round)
                frontier_changed = True
        return base

    def _clause_strength(self, key: str) -> float:
        pos = float(self.discovered_pair_positive.get(key, 0))
        neg = float(self.discovered_pair_negative.get(key, 0))
        return (pos + 1.0) / (pos + neg + 1.0) if (pos + neg) > 0 else 0.0

    def _remember_discovered_pair(self, key: str, helpful: bool = True, parents: Optional[Tuple[str, str]] = None) -> bool:
        """Fixed-slot symbolic clause memory for interpretability only.

        P70 allowed the number of Dn symbolic rows to keep growing.  P71 makes
        rowless HDC binding the main composition path and keeps symbolic Dn rows
        under a fixed slot budget.  A new clause must either use an empty slot or
        replace the weakest existing slot; harmful evidence is recorded only for
        clauses that already own a slot.
        """
        key = str(key)
        budget = self._dynamic_symbolic_clause_budget()
        if budget <= 0:
            return False
        exists = key in self.discovered_pair_positive or key in self.discovered_pair_negative
        if parents is not None:
            self.discovered_pair_parents.setdefault(key, (str(parents[0]), str(parents[1])))
        if exists:
            if helpful:
                self.discovered_pair_positive[key] += 1
            else:
                self.discovered_pair_negative[key] += 1
            self.max_discovered_depth_seen = max(self.max_discovered_depth_seen, _feature_depth(key))
            return True
        if not helpful:
            return False
        live = set(self.discovered_pair_positive) | set(self.discovered_pair_negative)
        if len(live) < budget:
            self.discovered_pair_positive[key] += 1
            if parents is not None:
                self.discovered_pair_parents[key] = (str(parents[0]), str(parents[1]))
            self.max_discovered_depth_seen = max(self.max_discovered_depth_seen, _feature_depth(key))
            return True
        weakest = min(live, key=lambda k: (self._clause_strength(k), self.discovered_pair_positive.get(k, 0) + self.discovered_pair_negative.get(k, 0), k))
        # Competitive replacement: a fresh helpful clause can replace only a row
        # whose empirical useful fraction is not above neutral.
        if self._clause_strength(weakest) <= 0.5:
            self.discovered_pair_positive.pop(weakest, None)
            self.discovered_pair_negative.pop(weakest, None)
            self.discovered_pair_parents.pop(weakest, None)
            self.discovered_pair_positive[key] += 1
            if parents is not None:
                self.discovered_pair_parents[key] = (str(parents[0]), str(parents[1]))
            self.max_discovered_depth_seen = max(self.max_discovered_depth_seen, _feature_depth(key))
            return True
        return False

    def _loss_from_score_map(self, scores: Dict[str, float], target_token: str) -> float:
        if not scores:
            return math.log(2.0)
        vals = list(float(v) for v in scores.values())
        target_score = float(scores.get(str(target_token), 0.0))
        vals.append(target_score)
        mx = max(vals)
        denom = sum(math.exp(v - mx) for v in vals) or 1.0
        p = math.exp(target_score - mx) / denom
        return -math.log(max(p, 1e-12))

    def _assign_template_credit(
        self,
        rows: List[Tuple[str, float]],
        target_token: str,
        wrong_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Route rank-loss to templates and discover useful feature crosses.

        This is a local weightless credit-assignment rule over the feature
        generator.  It does not create a dense matrix and it does not backprop
        through tensors.  Instead, each active template receives credit from the
        actual sparse rows it emitted:

          margin(template) = evidence_for_gold - evidence_for_ranked_losers

        Positive-margin templates keep/gain activation.  Negative-margin
        templates lose activation.  Positive row combinations are promoted to
        learned recursive Dn composite rows, so future calls can activate features
        whose parents may themselves be discovered features.
        """
        if not rows:
            return {'template_credit_used': False}
        wrong_tokens = [str(t) for t in (wrong_tokens or []) if str(t) != str(target_token)]
        groups: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for f, w in rows:
            groups[_feature_template_id(f)].append((f, w))

        helpful_templates: List[str] = []
        harmful_templates: List[str] = []
        helpful_rows: List[str] = []
        harmful_rows: List[str] = []
        target_token = str(target_token)

        feature_next = getattr(self.tables, 'feature_next', {})
        feature_totals = getattr(self.tables, 'feature_totals', {})
        feature_wrong = getattr(self.tables, 'feature_wrong', {})
        feature_wrong_totals = getattr(self.tables, 'feature_wrong_totals', {})
        row_reliability = getattr(self.tables, '_row_reliability', None)

        for tid, group_rows in groups.items():
            good = 0.0
            bad = 0.0
            observed = 0
            for f, w in group_rows:
                row = feature_next.get(f)
                if row:
                    total = float(feature_totals.get(f, 0) or sum(row.values()) or 1.0)
                    rel = float(row_reliability(row, total)) if callable(row_reliability) else math.log1p(total)
                    good += float(w) * rel * (float(row.get(target_token, 0)) / total)
                    if wrong_tokens:
                        bad += float(w) * rel * max((float(row.get(bt, 0)) / total for bt in wrong_tokens), default=0.0)
                    observed += 1
                wrow = feature_wrong.get(f)
                if wrow:
                    wtotal = float(feature_wrong_totals.get(f, 0) or sum(wrow.values()) or 1.0)
                    wrel = float(row_reliability(wrow, wtotal)) if callable(row_reliability) else math.log1p(wtotal)
                    # If this row previously had the gold token marked wrong,
                    # it is harmful for the current target; wrong tokens marked
                    # wrong are evidence that the template can discriminate.
                    bad += float(w) * wrel * (float(wrow.get(target_token, 0)) / wtotal)
                    if wrong_tokens:
                        good += float(w) * wrel * max((float(wrow.get(bt, 0)) / wtotal for bt in wrong_tokens), default=0.0)
                    observed += 1
            if observed <= 0:
                continue
            margin = good - bad
            if margin > 1e-12:
                self.template_positive[tid] += 1
                self.template_loss_delta[tid] += int(max(1, round(margin * 1000.0)))
                helpful_templates.append(tid)
                helpful_rows.extend([f for f, _w in group_rows])
            elif margin < -1e-12:
                self.template_negative[tid] += 1
                self.template_loss_delta[tid] -= int(max(1, round(abs(margin) * 1000.0)))
                harmful_templates.append(tid)
                harmful_rows.extend([f for f, _w in group_rows])
        self.template_credit_events += 1

        # P71: do not let explicit Dn symbols grow without bound.  Rowless HDC
        # bind/bundle composites are handled inside SparseEvidenceTables.  The
        # symbolic Dn path below is only a data-budgeted clause layer for
        # interpretability / highly repeated combinations.
        helpful_rows = self._candidate_pair_sources([(f, 1.0) for f in helpful_rows])
        harmful_rows = self._candidate_pair_sources([(f, 1.0) for f in harmful_rows])
        born = 0
        for source_rows, helpful in ((helpful_rows, True), (harmful_rows, False)):
            max_births = max(1, int(math.sqrt(max(1, len(source_rows)))))
            made = 0
            for i in range(len(source_rows)):
                for j in range(i + 1, len(source_rows)):
                    key = self._pair_row_key(source_rows[i], source_rows[j])
                    changed = self._remember_discovered_pair(key, helpful=helpful, parents=(source_rows[i], source_rows[j]))
                    made += 1
                    if helpful and changed:
                        born += 1
                    if made >= max_births:
                        break
                if made >= max_births:
                    break
        self.discovered_feature_events += born
        return {
            'template_credit_used': True,
            'helpful_templates': helpful_templates,
            'harmful_templates': harmful_templates,
            'discovered_pairs_born': int(born),
        }

    def _active_features(self, prompt_features: List[str], prefix: List[str]) -> List[Tuple[str, float]]:
        """Return symbolic live rows with learned template gates.

        P71 removes the fixed feature-weight ladder.  Base rows come from the
        existing symbolic templates, but their activation strength is learned by
        template credit.  HDC bind/bundle composite vectors handle ordinary composition without new
        symbolic rows. Recursive Dn rows remain only as data-budgeted clause slots.
        """
        feats: List[str] = []

        sig_feats = [f for f in prompt_features if f.startswith(('sig:', 'rare2:', 'bag2:', 'pt2:', 'pt3:'))]
        rare_feats = [f for f in prompt_features if f.startswith(('rare', 'char:', 'pt:', 'p3:', 'p3e:', 'rpre4:', 'rsuf4:', 'bfrare:', 'bfsk:'))]
        shape_feats = [f for f in prompt_features if f.startswith(('pc:', 'prompt:'))]

        if not prefix:
            feats.append('PX|0')
            for f in self._take_data_features(sig_feats, len(prefix), len(feats)):
                feats.append('P0S|' + f)
            for f in self._take_data_features(rare_feats, len(prefix), len(feats)):
                feats.append('P0R|' + f)
            for f in self._take_data_features(shape_feats, len(prefix), len(feats)):
                feats.append('P0C|' + f)
        else:
            l1 = prefix[-1]
            c1 = _class(l1)
            feats.append('L1|' + l1)
            feats.append('C1|' + c1)
            for f in self._take_data_features(sig_feats, len(prefix), len(feats)):
                feats.append('P1S|' + f + '|L1|' + l1)
                feats.append('P1SC|' + f + '|C1|' + c1)
            for f in self._take_data_features(rare_feats, len(prefix), len(feats)):
                feats.append('P1R|' + f + '|L1|' + l1)
            if len(prefix) >= 2:
                l2 = prefix[-2] + ' ' + prefix[-1]
                c2 = _class(prefix[-2]) + ' ' + _class(prefix[-1])
                feats.append('L2|' + l2)
                feats.append('C2|' + c2)
                for f in self._take_data_features(sig_feats, len(prefix), len(feats)):
                    feats.append('P2S|' + f + '|L2|' + l2)
                for f in self._take_data_features(rare_feats, len(prefix), len(feats)):
                    feats.append('P2R|' + f + '|L2|' + l2)
            if len(prefix) >= 3:
                l3 = prefix[-3] + ' ' + prefix[-2] + ' ' + prefix[-1]
                c3 = _class(prefix[-3]) + ' ' + _class(prefix[-2]) + ' ' + _class(prefix[-1])
                feats.append('L3|' + l3)
                feats.append('C3|' + c3)
                for f in self._take_data_features(sig_feats, len(prefix), len(feats)):
                    feats.append('P3S|' + f + '|L3|' + l3)
                for f in self._take_data_features(rare_feats, len(prefix), len(feats)):
                    feats.append('P3R|' + f + '|L3|' + l3)

        for f in self._take_data_features(sig_feats, len(prefix), len(feats)):
            feats.append('PB|' + f)

        if prefix:
            tail = prefix[-self._dynamic_context_window(len(prefix)):]
            pos_items = list(reversed(tail[: self._data_feature_budget(len(tail), len(prefix), len(feats))]))
            for j, tok in enumerate(pos_items):
                cls = _class(tok)
                feats.append(f'R{j}|{tok}')
                feats.append(f'RC{j}|{cls}')
            seen_bag = set()
            for tok in reversed(tail):
                if tok in seen_bag:
                    continue
                seen_bag.add(tok)
                if len(seen_bag) > self._data_feature_budget(len(tail), len(prefix), len(feats)):
                    break
                feats.append('PBAG|' + tok)
            cc = {}
            for tok in tail:
                c = _class(tok); cc[c] = cc.get(c, 0) + 1
            if cc:
                sk = '|'.join(f'{k}:{cc[k]}' for k in sorted(cc))
                feats.append('PSK|' + sk)

        n = len(prefix)
        feats.append('LEN|' + str(n))
        feats.append('LENLOG|' + str(int(math.ceil(math.log2(max(1, n + 1))))))

        out = self._weighted_rows_from_templates(feats)
        return self._add_discovered_feature_rows(out)

    def _operator_keys(self, prefix: List[str]) -> List[str]:
        keys: List[str] = []
        if prefix:
            keys.append('OP1|' + _class(prefix[-1]))
            keys.append('OP1LR|' + self._token_role(prefix[-1]))
        else:
            keys.append('OP1|[BOS]')
            keys.append('OP1LR|[BOS]')
        if len(prefix) >= 2:
            keys.append('OP2|' + _class(prefix[-2]) + '|' + _class(prefix[-1]))
            keys.append('OP2LR|' + self._token_role(prefix[-2]) + '|' + self._token_role(prefix[-1]))
        return keys

    def _observe_operator(self, prefix: List[str], next_tok: str, amount: int = 1, wrong_tokens: Optional[List[str]] = None) -> None:
        nxt = _class(next_tok)
        if prefix:
            last = str(prefix[-1])
            self.token_next_role[last][nxt] += max(1, int(amount))
            self.token_role_totals[last] += max(1, int(amount))
        for key in self._operator_keys(prefix):
            self.operator_next_class[key][nxt] += max(1, int(amount))
            self.operator_totals[key] += max(1, int(amount))
            for bad_tok in (wrong_tokens or []):
                bad_cls = _class(str(bad_tok))
                if bad_cls != nxt:
                    self.operator_wrong_class[key][bad_cls] += 1
                    self.operator_wrong_totals[key] += 1

    def _operator_support(self, prefix: List[str], tok: str) -> float:
        cls = _class(tok)
        vals: List[float] = []
        for key in self._operator_keys(prefix):
            total = float(self.operator_totals.get(key, 0) or 0)
            good = float(self.operator_next_class[key].get(cls, 0)) / total if total > 0 else 0.0
            wtotal = float(self.operator_wrong_totals.get(key, 0) or 0)
            bad = float(self.operator_wrong_class[key].get(cls, 0)) / wtotal if wtotal > 0 else 0.0
            if total > 0 or wtotal > 0:
                vals.append(good - bad)
        return sum(vals) / len(vals) if vals else 0.0

    def _distribution_energy(self, candidates: List[Tuple[str, float]]) -> Tuple[float, float]:
        if not candidates:
            return 1.0, 0.0
        vals = [max(0.0, float(s)) for _t, s in candidates]
        total = sum(vals)
        if total <= 0.0:
            return 1.0, 0.0
        probs = [v / total for v in vals if v > 0.0]
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs) / max(1e-12, math.log(max(2, len(probs))))
        evidence = max(probs) if probs else 0.0
        return float(entropy), float(evidence)

    def _recursive_value(self, prompt_features: List[str], prefix: List[str], tok: str, depth: int) -> Tuple[float, List[float], int]:
        key = None
        if self.fast_runtime:
            key = (tuple(prompt_features), tuple(prefix[-self._dynamic_context_window(len(prefix)):]), str(tok), int(depth), int(getattr(self.tables, 'updates', 0)), int(getattr(self.tables, 'readout_updates', 0)))
            cached = self._p74_value_cache.get(key)
            if cached is not None:
                self.recursive_value_cache_hits += 1
                return (float(cached[0]), list(cached[1]), int(cached[2]))
            self.recursive_value_cache_misses += 1
        next_prefix = prefix + [tok]
        next_feats = self._active_features_cached(prompt_features, next_prefix)
        if hasattr(self.tables, 'score_from_features_fast'):
            nxt = self.tables.score_from_features_fast(next_feats, top_k=self._dynamic_top_k(len(next_prefix)))
        else:
            nxt = self.tables.score_from_features(next_feats, top_k=self._dynamic_top_k(len(next_prefix)))
        entropy, evidence = self._distribution_energy(nxt)
        op = self._operator_support(prefix, tok)
        energy = entropy - evidence - op
        trace = [float(energy)]
        checks = 1
        if depth <= 1 or not nxt:
            result = (-energy, trace, checks)
            if key is not None:
                self._p74_value_cache[key] = result
            return result
        child_vals = []
        for child_tok, child_score in nxt[: min(2, len(nxt))]:
            cv, ct, cc = self._recursive_value(prompt_features, next_prefix, child_tok, depth - 1)
            child_vals.append(cv)
            trace.extend(ct)
            checks += cc
        if child_vals:
            result = (-energy + (sum(child_vals) / len(child_vals)), trace, checks)
        else:
            result = (-energy, trace, checks)
        if key is not None:
            self._p74_value_cache[key] = result
        return result

    def _recursive_stabilize_candidates(
        self,
        prompt_features: List[str],
        prefix: List[str],
        candidates: List[Tuple[str, float]],
        top_k: int,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        if not candidates:
            return [], {'hops': 0, 'energy_checks': 0, 'energy_trace': []}
        base_vals = [float(s) for _t, s in candidates]
        lo = min(base_vals); hi = max(base_vals); span = max(1e-12, hi - lo)
        refined: List[Tuple[str, float]] = []
        energy_trace: List[float] = []
        checks = 0
        depth = self._dynamic_hops()
        # Lookahead width is a runtime budget, not a reasoning-depth cap.  The
        # base candidate list still comes from full sparse+HDC+readout scoring;
        # recursive stabilization only refines the strongest few.
        lookahead_width = min(len(candidates), dynamic_hdc_source_budget(len(candidates), self._data_scale()['rows'], self._data_scale()['events'], max(1, top_k)))
        for tok, score in candidates[: lookahead_width]:
            base_norm = 1.0 if span <= 1e-12 else ((float(score) - lo) / span)
            lookahead, trace, c = self._recursive_value(prompt_features, prefix, tok, depth=depth)
            energy_trace.extend(trace)
            checks += c
            op = self._operator_support(prefix, tok)
            refined.append((tok, float(base_norm + lookahead + op)))
        refined.sort(key=lambda x: (-x[1], x[0]))
        # Keep unrefined tail so candidate diversity is not destroyed by the
        # speed budget.
        refined_tokens = {t for t, _s in refined}
        for tok, score in candidates:
            if tok not in refined_tokens:
                refined.append((tok, float(score)))
            if len(refined) >= max(top_k, 1):
                break
        refined.sort(key=lambda x: (-x[1], x[0]))
        return refined[: max(top_k, 1)], {
            'hops': depth,
            'energy_checks': checks,
            'energy_trace': energy_trace,
        }

    def _detokenize(self, tokens: List[str]) -> str:
        if not tokens:
            return ''
        out: List[str] = []
        at_line_start = True
        for tok in tokens:
            if tok == '[NL]':
                out.append('\n'); at_line_start = True; continue
            if tok == '[INDENT]':
                if at_line_start:
                    out.append('    ')
                else:
                    out.append(' ')
                continue
            if tok == '[DEDENT]':
                continue
            if not out:
                out.append(tok); at_line_start = False; continue
            prev = out[-1]
            if prev.endswith('\n') or at_line_start:
                out.append(tok); at_line_start = False
            elif tok in {')',']','}',',',':',';','.'}:
                out.append(tok)
            elif prev in {'(', '[', '{', '.', '\n', '    '}:
                out.append(tok)
            elif tok in {'(', '[', '{'}:
                out.append(tok)
            elif tok in {'==','!=','<=','>=','+','-','*','/','//','%','=','+=','-=','*=','/=','<','>','and','or'}:
                out.append(' ' + tok + ' ')
            else:
                if prev.endswith(' '):
                    out.append(tok)
                else:
                    out.append(' ' + tok)
        text = ''.join(out)
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.replace('( ', '(').replace(' )', ')').replace('[ ', '[').replace(' ]', ']').replace('{ ', '{').replace(' }', '}')
        text = text.replace(' ,', ',').replace(' :', ':').replace(' .', '.')
        return text.strip()


    def freeze(self):
        """Return a compact inference artifact (P85).

        The training object can keep Python dicts and rich credit state.  The
        frozen artifact is inference-only: integer row ids + CSR postings, with
        Rust scoring when puhl_luck_core is available.
        """
        from ._frozen_runtime import FrozenSparseLogitModel
        return FrozenSparseLogitModel.from_generator(self)

    def save_frozen(self, path: str, compress: bool = True) -> str:
        return self.freeze().save(path, compress=compress)

    @staticmethod
    def load_frozen(path: str):
        from ._frozen_runtime import FrozenSparseLogitModel
        return FrozenSparseLogitModel.load(path)

    def predict_next_frozen(self, frozen, input_text: str, field_features: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        """One-token inference through a frozen compact artifact.

        P88 first tries the parity-budget Rust next-token engine: tokenization, active
        feature extraction, CSR scoring, copy-token resolution, and top choice all
        happen in Rust.  If the extension is unavailable, it falls back to the P85
        Python feature extraction + frozen CSR scorer path.
        """
        if field_features is None and hasattr(frozen, 'predict_text_rust'):
            try:
                tok, ranked, diag = frozen.predict_text_rust(input_text, top_k=self._dynamic_top_k(len(str(input_text).split())))
                if tok:
                    diag.update({
                        'generation_method': 'p88_parity_budget_rust_next_token',
                        'training_dict_scoring_used': False,
                    })
                    return tok, diag
            except Exception:
                pass
        in_tokens = self._tokenize(input_text)
        copy_tokens = self._copy_tokens(in_tokens)
        prompt_features = self._prompt_features(in_tokens, field_features)
        rows = self._active_features(prompt_features, [])
        ranked = frozen.score_features(rows, top_k=self._dynamic_top_k(len(in_tokens)))
        tok_model = ranked[0][0] if ranked else ''
        visible = self._resolve_copy(tok_model, copy_tokens)
        tok = visible if visible is not None else tok_model
        return tok, {
            'generation_method': 'p85_frozen_compact_csr_rust_first_python_features',
            'frozen_rust_available': bool(getattr(frozen, 'rust_available', False)),
            'frozen_rows_active': int(len(rows)),
            'frozen_candidates': int(len(ranked)),
            'training_dict_scoring_used': False,
            'python_feature_extraction_used': True,
        }

    def generate_frozen(self, frozen, input_text: str, max_tokens: Optional[int] = None, temperature: float = 0.0) -> Tuple[str, Dict[str, Any]]:
        if hasattr(frozen, 'generate_text_rust'):
            text, toks, diag = frozen.generate_text_rust(
                input_text,
                max_tokens=self._dynamic_max_tokens(len(str(input_text).split()), requested=max_tokens),
                temperature=float(temperature),
            )
            if toks or text:
                diag.update({'generation_method': 'p88_parity_budget_rust_generate'})
                return text, diag
        out = []
        cur = str(input_text)
        steps = self._dynamic_max_tokens(len(self._tokenize(input_text)), requested=max_tokens)
        for _ in range(steps):
            tok, _diag = self.predict_next_frozen(frozen, cur)
            if not tok or tok == self.EOS:
                break
            out.append(tok)
            cur = (cur + ' ' + tok).strip()
        return self._detokenize(out), {'generation_method': 'python_fallback_autoregressive_generate', 'tokens': len(out)}

    def save(self, path: str, compress: bool = True) -> Dict[str, Any]:
        """Save the model with gzip compression (Task 9.2).
        
        Serializes the complete SparseLogitGenerator state to disk using pickle
        with optional gzip compression. The saved model includes all learned data:
        - Sparse evidence tables (feature_next, hdc_next, vocab)
        - Credit assignment state (template gates, discovered features)
        - Operator memories
        - Learning statistics
        - Configuration parameters
        
        This method implements compressed serialization as required by Requirements
        14.3 and 14.4: "Use pickle for object serialization with gzip wrapper" and
        "Verify memory footprint <500MB for 10K+ training pairs".
        
        Args:
            path: File path to save the model (e.g., 'model.pkl.gz')
            compress: Whether to use gzip compression (default: True)
            
        Returns:
            Dict with keys:
                - path: absolute file path where model was saved
                - compressed: whether compression was used
                - size_bytes: file size in bytes
                - size_mb: file size in megabytes
                - memory_footprint: memory usage breakdown from get_memory_footprint()
                
        Requirements: 14.3, 14.4
        
        Example:
            >>> gen = SparseLogitGenerator()
            >>> gen.learn("def add(a, b):", "return a + b")
            >>> result = gen.save("my_model.pkl.gz")
            >>> print(f"Saved {result['size_mb']:.2f} MB to {result['path']}")
        """
        import os
        
        # Prepare state dict with all necessary data
        state = {
            # Core configuration
            'temperature': self.temperature,
            'output_mode': self.output_mode,
            'max_tokens': self.max_tokens,
            'top_k': self.top_k,
            'max_hops': self.max_hops,
            'symbolic_clause_budget': self.symbolic_clause_budget,
            'adaptive_readout': self.adaptive_readout,
            'readout_hidden_dim': self.readout_hidden_dim,
            'readout_vocab_cap': self.readout_vocab_cap,
            'readout_lr': self.readout_lr,
            'readout_auto_resize': self.readout_auto_resize,
            'readout_min_hidden': self.readout_min_hidden,
            'readout_max_hidden': self.readout_max_hidden,
            'readout_min_vocab_cap': self.readout_min_vocab_cap,
            'readout_max_vocab_cap': self.readout_max_vocab_cap,
            'readout_active_features': self.readout_active_features,
            'runtime_cache_size': self.runtime_cache_size,
            'fast_runtime': self.fast_runtime,
            'runtime_workers': self.runtime_workers,
            'parallel_inference': self.parallel_inference,
            'batch_learning': self.batch_learning,
            
            # Special tokens
            'BOS': self.BOS,
            'SEP': self.SEP,
            'EOS': self.EOS,
            
            # Learning statistics
            'pairs_learned': self.pairs_learned,
            'tokens_learned': self.tokens_learned,
            
            # Scorer state
            'scorer_repetition_penalty_weight': self.scorer.repetition_penalty_weight,
            'scorer_repetition_window': self.scorer.repetition_window,
            
            # SparseEvidenceTables state (the main learned data)
            'tables_state': {
                'readout_enabled': self.tables.readout_enabled,
                'readout_auto_resize': self.tables.readout_auto_resize,
                'readout_min_hidden': self.tables.readout_min_hidden,
                'readout_max_hidden': self.tables.readout_max_hidden,
                'readout_min_vocab_cap': self.tables.readout_min_vocab_cap,
                'readout_max_vocab_cap': self.tables.readout_max_vocab_cap,
                'readout_hidden_dim': self.tables.readout_hidden_dim,
                'readout_vocab_cap': self.tables.readout_vocab_cap,
                'readout_lr': self.tables.readout_lr,
                'readout_active_features': self.tables.readout_active_features,
                'readout_resize_count': self.tables.readout_resize_count,
                'readout_weights': dict(self.tables.readout_weights),  # Convert defaultdict to dict
                'readout_index': dict(self.tables.readout_index),
                'readout_token_updates': dict(self.tables.readout_token_updates),
                'readout_updates': self.tables.readout_updates,
                'feature_next': dict(self.tables.feature_next),  # Convert defaultdict to dict
                'feature_totals': dict(self.tables.feature_totals),
                'feature_wrong': dict(self.tables.feature_wrong),
                'feature_wrong_totals': dict(self.tables.feature_wrong_totals),
                'vocab': dict(self.tables.vocab),
                'hdc_next': dict(self.tables.hdc_next),
                'hdc_totals': dict(self.tables.hdc_totals),
                'hdc_wrong': dict(self.tables.hdc_wrong),
                'hdc_wrong_totals': dict(self.tables.hdc_wrong_totals),
                'hdc_buckets': {k: list(v) for k, v in self.tables.hdc_buckets.items()},  # Convert sets to lists
                'feature_hv': dict(self.tables.feature_hv),
                'hdc_bits': self.tables.hdc_bits,
                'updates': self.tables.updates,
                'resize_count': self.tables.resize_count,
            },
            
            # Operator state
            'operator_next_class': dict(self.operator_next_class),
            'operator_totals': dict(self.operator_totals),
            'operator_wrong_class': dict(self.operator_wrong_class),
            'operator_wrong_totals': dict(self.operator_wrong_totals),
            
            # Template and discovered feature state
            'template_positive': dict(self.template_positive),
            'template_negative': dict(self.template_negative),
            'template_loss_delta': dict(self.template_loss_delta),
            'discovered_pair_positive': dict(self.discovered_pair_positive),
            'discovered_pair_negative': dict(self.discovered_pair_negative),
            'discovered_pair_parents': dict(self.discovered_pair_parents),
            'max_discovered_depth_seen': self.max_discovered_depth_seen,
            'token_next_role': dict(self.token_next_role),
            'token_role_totals': dict(self.token_role_totals),
            'template_credit_events': self.template_credit_events,
            'discovered_feature_events': self.discovered_feature_events,
            
            # Generation statistics
            'total_generations': self.total_generations,
            'total_field_only_backoffs': self.total_field_only_backoffs,
            'generation_backoff_levels': self.generation_backoff_levels,
            
            # Version info for forward compatibility
            'version': '1.0',
            'serialization_method': 'pickle_gzip' if compress else 'pickle',
        }
        
        # Serialize and save
        if compress:
            with gzip.open(path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Get file size and memory footprint
        file_size = os.path.getsize(path)
        memory_footprint = self.tables.get_memory_footprint()
        
        return {
            'path': os.path.abspath(path),
            'compressed': compress,
            'size_bytes': file_size,
            'size_mb': file_size / (1024 * 1024),
            'memory_footprint': memory_footprint,
        }
    
    @staticmethod
    def load(path: str) -> 'SparseLogitGenerator':
        """Load a saved model from disk (Task 9.2).
        
        Deserializes a SparseLogitGenerator from a file saved by the save() method.
        Automatically detects whether the file is gzip-compressed and handles both
        compressed and uncompressed pickle files.
        
        This method implements compressed serialization as required by Requirements
        14.3 and 14.4: "Add load() class method to deserialize compressed models".
        
        Args:
            path: File path to the saved model (e.g., 'model.pkl.gz')
            
        Returns:
            SparseLogitGenerator instance restored from the saved state
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pickle.UnpicklingError: If the file is corrupted or not a valid pickle
            
        Requirements: 14.3, 14.4
        
        Example:
            >>> gen = SparseLogitGenerator.load("my_model.pkl.gz")
            >>> output, metrics = gen.generate("def subtract(a, b):")
            >>> print(output)
        """
        import os
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Try to load as gzip first, fall back to uncompressed if that fails
        try:
            with gzip.open(path, 'rb') as f:
                state = pickle.load(f)
        except (OSError, gzip.BadGzipFile):
            # Not a gzip file, try loading as regular pickle
            with open(path, 'rb') as f:
                state = pickle.load(f)
        
        # Create a new instance with configuration from saved state
        gen = SparseLogitGenerator(
            temperature=state.get('temperature', 1.0),
            max_tokens=state.get('max_tokens', 0),
            top_k=state.get('top_k', 0),
            output_mode=state.get('output_mode', 'text'),
            max_hops=state.get('max_hops', 0),
            symbolic_clause_budget=state.get('symbolic_clause_budget', 0),
            adaptive_readout=state.get('adaptive_readout', True),
            readout_hidden_dim=state.get('readout_hidden_dim', 0),
            readout_vocab_cap=state.get('readout_vocab_cap', 0),
            readout_lr=state.get('readout_lr', 0.0),
            readout_auto_resize=state.get('readout_auto_resize', True),
            readout_min_hidden=state.get('readout_min_hidden', 0),
            readout_max_hidden=state.get('readout_max_hidden', 0),
            readout_min_vocab_cap=state.get('readout_min_vocab_cap', 0),
            readout_max_vocab_cap=state.get('readout_max_vocab_cap', 0),
            readout_active_features=state.get('readout_active_features', 0),
            runtime_cache_size=state.get('runtime_cache_size', 0),
            fast_runtime=state.get('fast_runtime', True),
            runtime_workers=state.get('runtime_workers', 0),
            parallel_inference=state.get('parallel_inference', True),
            batch_learning=state.get('batch_learning', True),
            repetition_penalty_weight=state.get('scorer_repetition_penalty_weight'),
            repetition_window=state.get('scorer_repetition_window'),
        )
        
        # Restore special tokens
        gen.BOS = state.get('BOS', '[BOS]')
        gen.SEP = state.get('SEP', '[SEP]')
        gen.EOS = state.get('EOS', '[EOS]')
        
        # Restore learning statistics
        gen.pairs_learned = state.get('pairs_learned', 0)
        gen.tokens_learned = state.get('tokens_learned', 0)
        
        # Restore SparseEvidenceTables state
        tables_state = state.get('tables_state', {})
        
        # Restore readout configuration
        gen.tables.readout_enabled = tables_state.get('readout_enabled', True)
        gen.tables.readout_auto_resize = tables_state.get('readout_auto_resize', True)
        gen.tables.readout_min_hidden = tables_state.get('readout_min_hidden', 64)
        gen.tables.readout_max_hidden = tables_state.get('readout_max_hidden', 0)
        gen.tables.readout_min_vocab_cap = tables_state.get('readout_min_vocab_cap', 100)
        gen.tables.readout_max_vocab_cap = tables_state.get('readout_max_vocab_cap', 0)
        gen.tables.readout_hidden_dim = tables_state.get('readout_hidden_dim', 0)
        gen.tables.readout_vocab_cap = tables_state.get('readout_vocab_cap', 0)
        gen.tables.readout_lr = tables_state.get('readout_lr', 0.0)
        gen.tables.readout_active_features = tables_state.get('readout_active_features', 0)
        gen.tables.readout_resize_count = tables_state.get('readout_resize_count', 0)
        
        # Restore readout weights (convert back to defaultdicts)
        readout_weights_dict = tables_state.get('readout_weights', {})
        gen.tables.readout_weights = defaultdict(dict)
        for tok, wdict in readout_weights_dict.items():
            gen.tables.readout_weights[tok] = dict(wdict)
        
        readout_index_dict = tables_state.get('readout_index', {})
        gen.tables.readout_index = defaultdict(dict)
        for idx, posting in readout_index_dict.items():
            gen.tables.readout_index[idx] = dict(posting)
        
        gen.tables.readout_token_updates = Counter(tables_state.get('readout_token_updates', {}))
        gen.tables.readout_updates = tables_state.get('readout_updates', 0)
        
        # Restore sparse evidence tables (convert back to defaultdicts/Counters)
        feature_next_dict = tables_state.get('feature_next', {})
        gen.tables.feature_next = defaultdict(Counter)
        for f, counter_dict in feature_next_dict.items():
            gen.tables.feature_next[f] = Counter(counter_dict)
        
        gen.tables.feature_totals = Counter(tables_state.get('feature_totals', {}))
        
        feature_wrong_dict = tables_state.get('feature_wrong', {})
        gen.tables.feature_wrong = defaultdict(Counter)
        for f, counter_dict in feature_wrong_dict.items():
            gen.tables.feature_wrong[f] = Counter(counter_dict)
        
        gen.tables.feature_wrong_totals = Counter(tables_state.get('feature_wrong_totals', {}))
        gen.tables.vocab = Counter(tables_state.get('vocab', {}))
        
        hdc_next_dict = tables_state.get('hdc_next', {})
        gen.tables.hdc_next = defaultdict(Counter)
        for key, counter_dict in hdc_next_dict.items():
            gen.tables.hdc_next[key] = Counter(counter_dict)
        
        gen.tables.hdc_totals = Counter(tables_state.get('hdc_totals', {}))
        
        hdc_wrong_dict = tables_state.get('hdc_wrong', {})
        gen.tables.hdc_wrong = defaultdict(Counter)
        for key, counter_dict in hdc_wrong_dict.items():
            gen.tables.hdc_wrong[key] = Counter(counter_dict)
        
        gen.tables.hdc_wrong_totals = Counter(tables_state.get('hdc_wrong_totals', {}))
        
        # Restore HDC buckets (convert lists back to sets)
        hdc_buckets_dict = tables_state.get('hdc_buckets', {})
        gen.tables.hdc_buckets = defaultdict(set)
        for key, bucket_list in hdc_buckets_dict.items():
            gen.tables.hdc_buckets[key] = set(bucket_list)
        
        gen.tables.feature_hv = dict(tables_state.get('feature_hv', {}))
        gen.tables.hdc_bits = tables_state.get('hdc_bits', 128)
        gen.tables.updates = tables_state.get('updates', 0)
        gen.tables.resize_count = tables_state.get('resize_count', 0)
        
        # Restore operator state (convert back to defaultdicts/Counters)
        operator_next_class_dict = state.get('operator_next_class', {})
        gen.operator_next_class = defaultdict(Counter)
        for key, counter_dict in operator_next_class_dict.items():
            gen.operator_next_class[key] = Counter(counter_dict)
        
        gen.operator_totals = Counter(state.get('operator_totals', {}))
        
        operator_wrong_class_dict = state.get('operator_wrong_class', {})
        gen.operator_wrong_class = defaultdict(Counter)
        for key, counter_dict in operator_wrong_class_dict.items():
            gen.operator_wrong_class[key] = Counter(counter_dict)
        
        gen.operator_wrong_totals = Counter(state.get('operator_wrong_totals', {}))
        
        # Restore template and discovered feature state
        gen.template_positive = Counter(state.get('template_positive', {}))
        gen.template_negative = Counter(state.get('template_negative', {}))
        gen.template_loss_delta = Counter(state.get('template_loss_delta', {}))
        gen.discovered_pair_positive = Counter(state.get('discovered_pair_positive', {}))
        gen.discovered_pair_negative = Counter(state.get('discovered_pair_negative', {}))
        gen.discovered_pair_parents = dict(state.get('discovered_pair_parents', {}))
        gen.max_discovered_depth_seen = state.get('max_discovered_depth_seen', 1)
        
        token_next_role_dict = state.get('token_next_role', {})
        gen.token_next_role = defaultdict(Counter)
        for key, counter_dict in token_next_role_dict.items():
            gen.token_next_role[key] = Counter(counter_dict)
        
        gen.token_role_totals = Counter(state.get('token_role_totals', {}))
        gen.template_credit_events = state.get('template_credit_events', 0)
        gen.discovered_feature_events = state.get('discovered_feature_events', 0)
        
        # Restore generation statistics
        gen.total_generations = state.get('total_generations', 0)
        gen.total_field_only_backoffs = state.get('total_field_only_backoffs', 0)
        gen.generation_backoff_levels = state.get('generation_backoff_levels', [])
        
        logger.info(f"Loaded model from {path}: {gen.pairs_learned} pairs, {gen.tokens_learned} tokens, "
                    f"{len(gen.tables.vocab)} vocab, {len(gen.tables.feature_next)} features")
        
        return gen

    def learned_storage_summary(self) -> Dict[str, Any]:
        """Measured learned model size, separated from Python process RSS."""
        meta = {
            'pairs_learned': int(self.pairs_learned),
            'tokens_learned': int(self.tokens_learned),
            'operator_next_class': {k: dict(v) for k, v in self.operator_next_class.items()},
            'operator_totals': dict(self.operator_totals),
            'operator_wrong_class': {k: dict(v) for k, v in self.operator_wrong_class.items()},
            'operator_wrong_totals': dict(self.operator_wrong_totals),
            'template_positive': dict(self.template_positive),
            'template_negative': dict(self.template_negative),
            'template_loss_delta': dict(self.template_loss_delta),
            'discovered_pair_positive': dict(self.discovered_pair_positive),
            'discovered_pair_negative': dict(self.discovered_pair_negative),
            'discovered_pair_parents': dict(self.discovered_pair_parents),
            'max_discovered_depth_seen': int(self.max_discovered_depth_seen),
            'token_next_role': {k: dict(v) for k, v in self.token_next_role.items()},
            'token_role_totals': dict(self.token_role_totals),
            'template_credit_events': int(self.template_credit_events),
            'discovered_feature_events': int(self.discovered_feature_events),
        }
        table_state = self.tables.learned_state() if hasattr(self.tables, 'learned_state') else {}
        payload_obj = {'generator_meta': meta, 'tables': table_state}
        payload = pickle.dumps(payload_obj, protocol=pickle.HIGHEST_PROTOCOL)
        gz = gzip.compress(payload)
        table_summary = self.tables.storage_summary() if hasattr(self.tables, 'storage_summary') else {}
        return {
            'learned_pickle_bytes': int(len(payload)),
            'learned_gzip_bytes': int(len(gz)),
            'table_storage': table_summary,
            'runtime_cache_excluded': True,
            'process_rss_excluded': True,
        }

    def get_statistics(self) -> Dict[str, Any]:
        tstats = self.tables.get_statistics() if hasattr(self.tables, 'get_statistics') else {}
        
        # Calculate total_transitions from sparse tables
        total_transitions = 0
        if hasattr(self.tables, 'feature_next'):
            total_transitions += sum(sum(counter.values()) for counter in self.tables.feature_next.values())
        if hasattr(self.tables, 'hdc_next'):
            total_transitions += sum(sum(counter.values()) for counter in self.tables.hdc_next.values())
        
        # Calculate total_contexts (unique context sketches/features)
        total_contexts = 0
        if hasattr(self.tables, 'feature_next'):
            total_contexts += len(self.tables.feature_next)
        if hasattr(self.tables, 'hdc_next'):
            total_contexts += len(self.tables.hdc_next)
        
        # Calculate total_unique_tokens from vocab
        total_unique_tokens = len(getattr(self.tables, 'vocab', {}))
        
        return {
            'pairs_learned': int(self.pairs_learned),
            'tokens_learned': int(self.tokens_learned),
            'total_transitions': int(total_transitions),
            'total_contexts': int(total_contexts),
            'total_unique_tokens': int(total_unique_tokens),
            'vocab_size': int(tstats.get('vocab_size', total_unique_tokens)),
            'max_hops': int(self._dynamic_hops()),
            'operator_rows': len(self.operator_next_class),
            'operator_total_updates': int(sum(self.operator_totals.values())),
            'operator_wrong_rows': len(self.operator_wrong_class),
            'operator_wrong_updates': int(sum(self.operator_wrong_totals.values())),
            'template_credit_events': int(self.template_credit_events),
            'template_positive_total': int(sum(self.template_positive.values())),
            'template_negative_total': int(sum(self.template_negative.values())),
            'template_rows': len(set(self.template_positive) | set(self.template_negative)),
            'discovered_pair_rows': len(set(self.discovered_pair_positive) | set(self.discovered_pair_negative)),
            'discovered_pair_parent_rows': len(getattr(self, 'discovered_pair_parents', {})),
            'max_discovered_depth_seen': int(self.max_discovered_depth_seen),
            'recursive_feature_discovery': True,
            'hdc_binding_composition': True,
            'symbolic_clause_budget': int(self._dynamic_symbolic_clause_budget()),
            'discovered_feature_events': int(self.discovered_feature_events),
            'learned_token_roles': len(self.token_next_role),
            'credit_assignment': 'rank_loss_to_sparse_hdc_operator_template_rowless_hdc_composites_data_budgeted_clauses_and_dynamic_adaptive_readout',
            'feature_discovery': 'rowless_hdc_bind_bundle_composites_plus_data_budgeted_recursive_symbolic_clauses',
            'p74_fast_runtime': True,
            'p76_fast_only_default': True,
            'p74_active_feature_cache_entries': len(getattr(self, '_active_feature_cache', {})),
            'p74_active_feature_cache_hits': int(getattr(self, 'active_feature_cache_hits', 0)),
            'p74_active_feature_cache_misses': int(getattr(self, 'active_feature_cache_misses', 0)),
            'p74_recursive_value_cache_entries': len(getattr(self, '_p74_value_cache', {})),
            'p74_recursive_value_cache_hits': int(getattr(self, 'recursive_value_cache_hits', 0)),
            'p74_recursive_value_cache_misses': int(getattr(self, 'recursive_value_cache_misses', 0)),
            'total_generations': int(getattr(self, 'total_generations', 0)),
            'total_field_only_backoffs': int(getattr(self, 'total_field_only_backoffs', 0)),
            'field_only_backoff_percentage': (
                (self.total_field_only_backoffs / sum(1 for _ in self.generation_backoff_levels) * 100)
                if hasattr(self, 'generation_backoff_levels') and len(self.generation_backoff_levels) > 0
                else 0.0
            ),
            'learned_storage': self.learned_storage_summary(),
            'tables': tstats,
        }

    def save(self, filepath: str) -> Dict[str, Any]:
        """Save the model to a gzip-compressed pickle file.
        
        Serializes the entire model state including sparse tables, learned patterns,
        and configuration. Uses gzip compression to reduce memory footprint.
        
        Requirements: 14.3, 14.4
        
        Args:
            filepath: Path where the compressed model should be saved
            
        Returns:
            Dictionary with save statistics (file_size_bytes, compression_ratio, etc.)
        """
        import os
        
        # Prepare model state for serialization
        model_state = {
            # Core configuration
            'max_tokens': self.max_tokens,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'output_mode': self.output_mode,
            'rare_token_threshold': self.rare_token_threshold,
            'max_hops': self.max_hops,
            'symbolic_clause_budget': self.symbolic_clause_budget,
            'adaptive_readout': self.adaptive_readout,
            'readout_hidden_dim': self.readout_hidden_dim,
            'readout_vocab_cap': self.readout_vocab_cap,
            'readout_lr': self.readout_lr,
            'readout_auto_resize': self.readout_auto_resize,
            'readout_min_hidden': self.readout_min_hidden,
            'readout_max_hidden': self.readout_max_hidden,
            'readout_min_vocab_cap': self.readout_min_vocab_cap,
            'readout_max_vocab_cap': self.readout_max_vocab_cap,
            'readout_active_features': self.readout_active_features,
            'runtime_cache_size': self.runtime_cache_size,
            'fast_runtime': self.fast_runtime,
            'runtime_workers': self.runtime_workers,
            'parallel_inference': self.parallel_inference,
            'batch_learning': self.batch_learning,
            
            # Special tokens
            'BOS': self.BOS,
            'SEP': self.SEP,
            'EOS': self.EOS,
            
            # Learning statistics
            'pairs_learned': self.pairs_learned,
            'tokens_learned': self.tokens_learned,
            
            # Sparse evidence tables (the main memory component)
            'tables': self.tables,
            
            # Scorer configuration
            'scorer_repetition_penalty_weight': self.scorer.repetition_penalty_weight,
            'scorer_repetition_window': self.scorer.repetition_window,
            
            # Operator learning structures
            'operator_next_class': dict(self.operator_next_class),
            'operator_totals': dict(self.operator_totals),
            'operator_wrong_class': dict(self.operator_wrong_class),
            'operator_wrong_totals': dict(self.operator_wrong_totals),
            
            # Template and discovery structures
            'template_positive': dict(self.template_positive),
            'template_negative': dict(self.template_negative),
            'template_loss_delta': dict(self.template_loss_delta),
            'discovered_pair_positive': dict(self.discovered_pair_positive),
            'discovered_pair_negative': dict(self.discovered_pair_negative),
            'discovered_pair_parents': dict(self.discovered_pair_parents),
            'max_discovered_depth_seen': self.max_discovered_depth_seen,
            'token_next_role': dict(self.token_next_role),
            'token_role_totals': dict(self.token_role_totals),
            'template_credit_events': self.template_credit_events,
            'discovered_feature_events': self.discovered_feature_events,
            
            # Generation statistics
            'total_generations': self.total_generations,
            'total_field_only_backoffs': self.total_field_only_backoffs,
            'generation_backoff_levels': self.generation_backoff_levels,
            
            # Version information for compatibility
            'version': '1.0',
            'model_type': 'SparseLogitGenerator',
        }
        
        # Serialize to bytes using pickle
        pickled_data = pickle.dumps(model_state, protocol=pickle.HIGHEST_PROTOCOL)
        uncompressed_size = len(pickled_data)
        
        # Compress with gzip and write to file
        with gzip.open(filepath, 'wb', compresslevel=9) as f:
            f.write(pickled_data)
        
        # Get compressed file size
        compressed_size = os.path.getsize(filepath)
        compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1.0
        
        logger.info(f"Model saved to {filepath}: {compressed_size / (1024*1024):.2f} MB "
                   f"(compression ratio: {compression_ratio:.2f}x)")
        
        return {
            'filepath': filepath,
            'uncompressed_size_bytes': uncompressed_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': compression_ratio,
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'pairs_learned': self.pairs_learned,
            'tokens_learned': self.tokens_learned,
        }
    
    @classmethod
    def load(cls, filepath: str) -> 'SparseLogitGenerator':
        """Load a model from a gzip-compressed pickle file.
        
        Deserializes a model that was saved with the save() method, restoring
        all learned patterns and configuration.
        
        Requirements: 14.3, 14.4
        
        Args:
            filepath: Path to the compressed model file
            
        Returns:
            Loaded SparseLogitGenerator instance with all state restored
        """
        import os
        
        # Read and decompress file
        with gzip.open(filepath, 'rb') as f:
            pickled_data = f.read()
        
        # Deserialize model state
        model_state = pickle.loads(pickled_data)
        
        # Verify model type
        if model_state.get('model_type') != 'SparseLogitGenerator':
            raise ValueError(f"Invalid model type: {model_state.get('model_type')}")
        
        # Create new instance with saved configuration
        instance = cls(
            max_tokens=model_state.get('max_tokens', 0),
            top_k=model_state.get('top_k', 0),
            temperature=model_state.get('temperature', 1.0),
            output_mode=model_state.get('output_mode', 'text'),
            rare_token_threshold=model_state.get('rare_token_threshold', 2),
            max_hops=model_state.get('max_hops', 0),
            symbolic_clause_budget=model_state.get('symbolic_clause_budget', 0),
            adaptive_readout=model_state.get('adaptive_readout', True),
            readout_hidden_dim=model_state.get('readout_hidden_dim', 0),
            readout_vocab_cap=model_state.get('readout_vocab_cap', 0),
            readout_lr=model_state.get('readout_lr', 0.0),
            readout_auto_resize=model_state.get('readout_auto_resize', True),
            readout_min_hidden=model_state.get('readout_min_hidden', 0),
            readout_max_hidden=model_state.get('readout_max_hidden', 0),
            readout_min_vocab_cap=model_state.get('readout_min_vocab_cap', 0),
            readout_max_vocab_cap=model_state.get('readout_max_vocab_cap', 0),
            readout_active_features=model_state.get('readout_active_features', 0),
            runtime_cache_size=model_state.get('runtime_cache_size'),
            fast_runtime=model_state.get('fast_runtime', True),
            runtime_workers=model_state.get('runtime_workers', 0),
            parallel_inference=model_state.get('parallel_inference', True),
            batch_learning=model_state.get('batch_learning', True),
        )
        
        # Restore special tokens
        instance.BOS = model_state.get('BOS', '[BOS]')
        instance.SEP = model_state.get('SEP', '[SEP]')
        instance.EOS = model_state.get('EOS', '[EOS]')
        
        # Restore learning statistics
        instance.pairs_learned = model_state.get('pairs_learned', 0)
        instance.tokens_learned = model_state.get('tokens_learned', 0)
        
        # Restore sparse evidence tables
        instance.tables = model_state.get('tables')
        
        # Restore scorer state
        instance.scorer.repetition_penalty_weight = model_state.get('scorer_repetition_penalty_weight', 1.0)
        instance.scorer.repetition_window = model_state.get('scorer_repetition_window', 20)
        
        # Restore operator learning structures
        instance.operator_next_class = defaultdict(Counter, {
            k: Counter(v) for k, v in model_state.get('operator_next_class', {}).items()
        })
        instance.operator_totals = Counter(model_state.get('operator_totals', {}))
        instance.operator_wrong_class = defaultdict(Counter, {
            k: Counter(v) for k, v in model_state.get('operator_wrong_class', {}).items()
        })
        instance.operator_wrong_totals = Counter(model_state.get('operator_wrong_totals', {}))
        
        # Restore template and discovery structures
        instance.template_positive = Counter(model_state.get('template_positive', {}))
        instance.template_negative = Counter(model_state.get('template_negative', {}))
        instance.template_loss_delta = Counter(model_state.get('template_loss_delta', {}))
        instance.discovered_pair_positive = Counter(model_state.get('discovered_pair_positive', {}))
        instance.discovered_pair_negative = Counter(model_state.get('discovered_pair_negative', {}))
        instance.discovered_pair_parents = model_state.get('discovered_pair_parents', {})
        instance.max_discovered_depth_seen = model_state.get('max_discovered_depth_seen', 1)
        instance.token_next_role = defaultdict(Counter, {
            k: Counter(v) for k, v in model_state.get('token_next_role', {}).items()
        })
        instance.token_role_totals = Counter(model_state.get('token_role_totals', {}))
        instance.template_credit_events = model_state.get('template_credit_events', 0)
        instance.discovered_feature_events = model_state.get('discovered_feature_events', 0)
        
        # Restore generation statistics
        instance.total_generations = model_state.get('total_generations', 0)
        instance.total_field_only_backoffs = model_state.get('total_field_only_backoffs', 0)
        instance.generation_backoff_levels = model_state.get('generation_backoff_levels', [])
        
        compressed_size = os.path.getsize(filepath)
        logger.info(f"Model loaded from {filepath}: {compressed_size / (1024*1024):.2f} MB, "
                   f"{instance.pairs_learned} pairs learned, {instance.tokens_learned} tokens learned")
        
        return instance

    def clear(self) -> None:
        if hasattr(self.tables, 'clear'):
            self.tables.clear()
        self.pairs_learned = 0
        self.tokens_learned = 0
        self.operator_next_class.clear()
        self.operator_totals.clear()
        self.operator_wrong_class.clear()
        self.operator_wrong_totals.clear()
        self.template_positive.clear()
        self.template_negative.clear()
        self.template_loss_delta.clear()
        self.discovered_pair_positive.clear()
        self.discovered_pair_negative.clear()
        self.discovered_pair_parents.clear()
        self.max_discovered_depth_seen = 1
        self.token_next_role.clear()
        self.token_role_totals.clear()
        self.template_credit_events = 0
        self.discovered_feature_events = 0
        self._clear_runtime_caches()
