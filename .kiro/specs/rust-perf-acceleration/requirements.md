# Requirements Document

## Introduction

PUHL-LUCK is a non-gradient cognitive memory system (BrainMemory) that includes a compiled Rust extension (`puhl_luck_core.pyd`) exposing seven modules: `hdc`, `energy`, `resonance`, `field`, `matching`, `completion`, and `memory_management`. Several of these Rust functions are already exposed but not yet wired into the Python hotpaths that dominate inference and learning latency.

This feature accelerates the three key performance dimensions of BrainMemory:

1. **Inference speed** — `generate()` / `answer()` latency, dominated by field formation (`_form_field_from_prompt`), candidate scoring, and per-token decoding (`next_token_energy_scores`).
2. **Learning speed** — `expose_text()` / `expose_pair()` throughput, dominated by O(n²) edge building in `expose_event()` and O(n²) pairwise HDC similarity in `induce_from_history()`.
3. **Memory management efficiency** — pruning throughput and triggered frequency, currently all pure Python.

The work has two tracks:
- **Track A — Wire existing Rust**: route Python hotpaths to already-compiled Rust functions, keeping data-conversion overhead minimal via shared representation conventions.
- **Track B — Add missing Rust functions**: implement `build_edges_rust`, `n_gram_next_token_scores_rust`, and `transition_similarity_batch_rust` in the Rust extension and wire them.

---

## Glossary

- **BrainMemory**: The main Python class composing all mixin layers into a single cognitive memory object.
- **Rust_Extension**: The compiled `puhl_luck_core.pyd` binary exposing all seven Rust modules to Python.
- **HDC_Vector**: A binary hyperdimensional computing vector of fixed bit-width (default 10 000 bits), represented as `numpy.ndarray` of dtype `int8`.
- **Edge_Graph**: The weighted co-occurrence graph stored in BrainMemory (`self.add_edge`), used for activation spreading.
- **Activation_Map**: A `dict[str, float]` mapping event or concept IDs to their current activation strength.
- **Feature_List**: A Python `list[str]` of string feature tokens prefixed by type (e.g., `tok:word`, `bi:a_b`, `id:hash`).
- **Transition**: A stored `StateTransition` pairing a partial `StateField` with a complete `StateField`.
- **Operator**: A stored `OperatorRecord` encoding a learned state-transformation pattern.
- **Candidate**: A `Candidate` object produced by `CandidateEmergence`, holding generated text and predicted energy reduction.
- **Field_Energy**: A `FieldEnergy` dataclass holding `total`, `evidence`, `conflicts`, and `tension_level` for a `StateField`.
- **N_Gram_Context**: A tuple of 1–`max_order` preceding tokens used as the key into `self.order_contexts`.
- **Conversion_Overhead**: The CPU time spent serialising Python objects into Rust-compatible types before a Rust call, and deserialising results back.
- **Prune_Cycle**: One invocation of `BrainMemory.prune()`, triggered after every 4 096 feature updates.

---

## Requirements

### Requirement 1: Wire `find_similar_transitions_rust` into `TransitionMemoryLayer`

**User Story:** As a developer, I want `TransitionMemoryLayer.find_similar_transitions()` to delegate its inner similarity loop to the Rust extension, so that transition search scales sub-linearly with the number of stored transitions instead of iterating all of them in Python.

#### Acceptance Criteria

1. WHEN `find_similar_transitions()` is called and `puhl_luck_core` is importable, THE `TransitionMemoryLayer` SHALL delegate candidate scoring to `find_similar_transitions_rust`, passing each candidate transition's `query_features` list as its `from_state` argument.
2. WHEN `find_similar_transitions_rust` returns results, THE `TransitionMemoryLayer` SHALL produce a ranking identical in sort order to the pure-Python fallback (within floating-point tolerance of ±1 × 10⁻⁶ per score).
3. IF `puhl_luck_core` is not importable, THEN THE `TransitionMemoryLayer` SHALL fall back to the existing pure-Python similarity loop without raising an exception.
4. THE `TransitionMemoryLayer` SHALL NOT convert individual `StateTransition` objects on every call; WHILE the layer is alive, THE `TransitionMemoryLayer` SHALL maintain a pre-built `list[(transition_id, from_features, to_features, confidence)]` cache invalidated only when transitions are added or removed.
5. WHEN the candidate set after domain/modality filtering is empty, THE `TransitionMemoryLayer` SHALL return an empty list without invoking the Rust extension.

---

### Requirement 2: Wire `find_applicable_operators_rust` into `OperatorMemoryLayer`

**User Story:** As a developer, I want `OperatorMemoryLayer.find_applicable_operators()` to use the Rust extension for precondition matching, so that operator lookup remains fast as the operator store grows.

#### Acceptance Criteria

1. WHEN `find_applicable_operators()` is called and `puhl_luck_core` is importable, THE `OperatorMemoryLayer` SHALL delegate precondition-matching to `find_applicable_operators_rust`, supplying `field_state.query_features` as `current_state` and the serialised operator list as `operators`.
2. WHEN `find_applicable_operators_rust` returns a match score, THE `OperatorMemoryLayer` SHALL still apply the `_compute_match_score` confidence weighting before including the operator in final results, so confidence-filtered operators are excluded.
3. IF the operator store is empty, THEN THE `OperatorMemoryLayer` SHALL return an empty list without calling the Rust extension.
4. THE `OperatorMemoryLayer` SHALL maintain a serialised operator cache `list[(op_id, preconditions, effects, confidence)]` that is invalidated whenever an operator is added, removed, or has its confidence updated.
5. WHEN `find_applicable_operators_rust` is called, THE `OperatorMemoryLayer` SHALL pass the `min_confidence` argument as the `min_match` parameter, so low-confidence operators are filtered inside Rust before Python post-processing.

---

### Requirement 3: Wire `batch_similarity_search_rust` into `OperatorInduction.induce_from_history`

**User Story:** As a developer, I want `OperatorInduction.identify_repeated_patterns()` to use `batch_similarity_search_rust` for the O(n²) pairwise clustering loop, so operator induction scales to hundreds of stored transitions without blocking the learning loop.

#### Acceptance Criteria

1. WHEN `induce_from_history()` is called with more than 2 transitions and `puhl_luck_core` is importable, THE `OperatorInduction` SHALL use `batch_similarity_search_rust` to compute pairwise similarities across all transition completion vectors represented as `Feature_List`.
2. WHEN `batch_similarity_search_rust` returns the similarity matrix, THE `OperatorInduction` SHALL assign each transition to the cluster whose centroid has the highest returned similarity, producing the same cluster membership as the greedy sequential algorithm (ties broken by insertion order).
3. IF `puhl_luck_core` is not importable, THEN THE `OperatorInduction` SHALL fall back to the existing sequential `hv_similarity` loop without raising an exception.
4. THE `OperatorInduction` SHALL NOT re-encode transitions that have already been converted; WHILE a single `induce_from_history()` call is executing, THE `OperatorInduction` SHALL build the feature-list representation once and reuse it for all Rust calls within that invocation.

---

### Requirement 4: Wire `activate_field_items_rust` into `_form_field_from_prompt`

**User Story:** As a developer, I want `_form_field_from_prompt` in `MemoryTransitionMixin` to use `activate_field_items_rust` for computing initial event activations from query features, so that field formation during `generate()` does not iterate events in a Python loop.

#### Acceptance Criteria

1. WHEN `_form_field_from_prompt()` is called and `puhl_luck_core` is importable, THE `MemoryTransitionMixin` SHALL call `activate_field_items_rust` with the list of `(event_id, features)` pairs for all activated events and the expanded query features, replacing the manual Jaccard-similarity loop.
2. WHEN `activate_field_items_rust` returns an activation dict, THE `MemoryTransitionMixin` SHALL normalise it to [0, 1] using the maximum activation value before assigning it to `activated_events`, matching the existing normalisation behaviour.
3. IF the event store is empty or the query feature list is empty, THEN THE `MemoryTransitionMixin` SHALL skip the Rust call and return an empty `activated_events` dict.
4. THE `MemoryTransitionMixin` SHALL cap the event list passed to `activate_field_items_rust` at 2 000 events (taking the top 2 000 by existing activation or recency) to prevent Conversion_Overhead from dominating over small Python fallbacks.

---

### Requirement 5: Wire `batch_compute_energy_rust` into `CandidateEmergence`

**User Story:** As a developer, I want `CandidateEmergence.generate_candidates()` to score all candidates' energy deltas in a single Rust call instead of calling `predict_energy_after_update` individually per candidate, so candidate scoring is parallelised.

#### Acceptance Criteria

1. WHEN `generate_candidates()` is called with one or more candidates and `puhl_luck_core` is importable, THE `CandidateEmergence` SHALL collect all candidates' conflict and evidence lists and submit them in a single `batch_compute_energy_rust` call.
2. WHEN `batch_compute_energy_rust` returns energy deltas, THE `CandidateEmergence` SHALL assign `candidate.energy_reduction = base_energy - delta` for each candidate, producing values equal to the sequential `compute_energy_reduction` path (within floating-point tolerance ±1 × 10⁻⁶).
3. IF `puhl_luck_core` is not importable, THEN THE `CandidateEmergence` SHALL fall back to the sequential per-candidate `compute_energy_reduction` loop.
4. IF the candidate list is empty, THEN THE `CandidateEmergence` SHALL skip the Rust call and return an empty list.

---

### Requirement 6: Wire `batch_compute_resonance_rust` and `propagate_resonance_rust` into field formation

**User Story:** As a developer, I want the resonance computation step in `_form_field_from_prompt` to use `batch_compute_resonance_rust` and `propagate_resonance_rust` from the Rust extension, so that inter-event resonance and activation propagation are computed in parallel.

#### Acceptance Criteria

1. WHEN `_form_field_from_prompt()` computes resonance between activated events and `puhl_luck_core` is importable, THE `MemoryTransitionMixin` SHALL call `batch_compute_resonance_rust` with the activated event feature lists as targets.
2. WHEN resonance scores are computed, THE `MemoryTransitionMixin` SHALL pass them to `propagate_resonance_rust` with `iterations=2` and `damping=0.85` to spread activation through the field before constructing the final `StateField`.
3. IF the number of activated events is fewer than 2, THEN THE `MemoryTransitionMixin` SHALL skip both Rust calls and proceed with unmodified activations.
4. IF `puhl_luck_core` is not importable, THEN THE `MemoryTransitionMixin` SHALL proceed with unmodified activations without raising an exception.

---

### Requirement 7: Wire pruning functions into `BrainMemory.prune()`

**User Story:** As a developer, I want `BrainMemory.prune()` to call `prune_events_rust`, `prune_transitions_rust`, and `prune_operators_rust` from the Rust extension, so that the pruning decision loop is parallelised and Python GIL contention is eliminated during Prune_Cycles.

#### Acceptance Criteria

1. WHEN `prune()` is triggered and `puhl_luck_core` is importable, THE `BrainMemory` SHALL call `prune_events_rust` with current novelty, last-access timestamps, and activation scores to determine the set of event IDs to remove.
2. WHEN `prune_events_rust` returns a list of IDs to remove, THE `BrainMemory` SHALL delete those events from `self.events`, `self.event_hv`, `self.event_novelty`, and all feature-to-event index structures in a single Python pass.
3. WHEN `prune()` is triggered and `puhl_luck_core` is importable, THE `BrainMemory` SHALL call `prune_transitions_rust` and `prune_operators_rust` with their respective relevance, usage, and timestamp data.
4. IF `puhl_luck_core` is not importable, THEN THE `BrainMemory` SHALL fall back to the existing pure-Python pruning logic without raising an exception.
5. THE `BrainMemory` SHALL pass `protected_ids` containing any event IDs referenced by stored transitions or operators to `prune_events_rust`, so memory cross-references are not broken.

---

### Requirement 8: Add `build_edges_rust` to the Rust extension and wire it into `expose_event()`

**User Story:** As a developer, I want a new Rust function `build_edges_rust` that replaces the O(n²) nested Python loop in `expose_event()`, so that `expose_text()` / `expose_pair()` throughput improves proportionally to the number of features per event.

#### Acceptance Criteria

1. THE `Rust_Extension` SHALL expose a function `build_edges_rust(ids: list[int], decay: float, edge_gain: float, window_size: int) -> list[(int, int, float, float)]` that returns a list of `(left_id, right_id, forward_weight, reverse_weight)` tuples representing all edges that the O(n²) Python loop would produce.
2. WHEN `expose_event()` processes a new event and `puhl_luck_core` is importable, THE `MemoryExposureMixin` SHALL call `build_edges_rust` once per new event with `ids`, `self.decay`, `edge_gain`, and the adaptive window size, then apply the returned edge list by calling `self.add_edge(left, right, weight)` for each tuple.
3. WHEN `build_edges_rust` returns the same `(left_id, right_id, forward_weight, reverse_weight)` tuples as the Python loop would have generated for identical inputs, THE `MemoryExposureMixin` SHALL produce an Edge_Graph state equal to the pure-Python baseline (weights within ±1 × 10⁻⁹).
4. IF `puhl_luck_core` is not importable, THEN THE `MemoryExposureMixin` SHALL execute the existing Python loop unchanged.
5. THE `build_edges_rust` function SHALL use the same decay formula `weight = edge_gain × decay^(j − i − 1)` and reverse-weight formula `reverse_weight = forward_weight × 0.35` as the Python implementation, so numerical equivalence is guaranteed.
6. FOR ALL inputs where `len(ids) >= 2`, the set of `(left_id, right_id)` pairs returned by `build_edges_rust` SHALL be identical to the set produced by the Python loop (round-trip correctness property).

---

### Requirement 9: Add `n_gram_next_token_scores_rust` to the Rust extension and wire it into `next_token_energy_scores()`

**User Story:** As a developer, I want a new Rust function `n_gram_next_token_scores_rust` that parallelises the n-gram backoff lookup in `next_token_energy_scores()`, so per-token decoding latency during `memory_energy_decode_text()` decreases.

#### Acceptance Criteria

1. THE `Rust_Extension` SHALL expose a function `n_gram_next_token_scores_rust(context_tokens: list[str], order_contexts: dict[tuple[str, ...], dict[str, int]], max_order: int) -> list[(str, float, int)]` returning `(token, score, matched_order)` tuples equivalent to the `order_backoff_options` loop.
2. WHEN `next_token_energy_scores()` is called and `puhl_luck_core` is importable, THE `MemoryEnergyDecodeMixin` SHALL call `n_gram_next_token_scores_rust` instead of `order_backoff_options` to populate the n-gram portion of `scores`.
3. WHEN `n_gram_next_token_scores_rust` returns scores, THE scores SHALL be combined with `event_next_token_options` and `semantic_energy` contributions using the same weight factors (1.0 + matched_order for n-gram, 1.4 for event, 0.20 for semantic) as the existing Python logic.
4. IF `puhl_luck_core` is not importable, THEN THE `MemoryEnergyDecodeMixin` SHALL fall back to the existing `order_backoff_options` Python function.
5. FOR ALL context sequences of length 1 to 8 and all max_order values of 1 to 6, the token ranking returned by `n_gram_next_token_scores_rust` SHALL be identical to the ranking returned by the Python backoff loop (round-trip ranking property: applying the same operation twice yields the same order).

---

### Requirement 10: Add `transition_similarity_batch_rust` to the Rust extension and integrate it as an alternative search path in `TransitionMemoryLayer`

**User Story:** As a developer, I want a new Rust function `transition_similarity_batch_rust` that computes HDC-based cosine similarity between a query vector and all stored transition vectors in one batched call, so transition search benefits from SIMD-level parallelism even for small HDC dimensions.

#### Acceptance Criteria

1. THE `Rust_Extension` SHALL expose a function `transition_similarity_batch_rust(query_hv: list[int], transition_hvs: list[(str, list[int])]) -> list[(str, float)]` that returns `(transition_id, similarity)` pairs sorted by similarity descending.
2. WHEN `query_hv` has at least 64 elements and at least 4 transitions are stored, THE `TransitionMemoryLayer` SHALL prefer `transition_similarity_batch_rust` over the feature-based `find_similar_transitions_rust` path, because HDC similarity is more precise than Jaccard on feature lists.
3. WHEN `transition_similarity_batch_rust` is called, THE function SHALL compute similarity as `(count_matching_bits / total_bits) × 2 − 1` normalised to [−1, 1], producing an equivalent ranking to `hv_similarity` (within ±1 × 10⁻⁶).
4. IF `query_hv` is empty or fewer than 4 transitions are stored, THEN THE `TransitionMemoryLayer` SHALL use the feature-based Rust path (Requirement 1) as the fallback.
5. FOR ALL HDC_Vector inputs, applying `transition_similarity_batch_rust` twice on the same inputs SHALL return lists with identical ordering (idempotence property).

---

### Requirement 11: Eliminate repeated Python↔Rust conversion overhead via shared representation caches

**User Story:** As a developer, I want each layer that holds persistent data (transitions, operators) to maintain pre-serialised Python-list representations of its objects, so Rust calls never trigger per-call object conversion that exceeds the Rust computation time.

#### Acceptance Criteria

1. THE `TransitionMemoryLayer` SHALL maintain a `_rust_cache: dict[str, tuple[list[str], list[str], float]]` mapping `transition_id → (from_features, to_features, confidence)` that is updated incrementally when transitions are stored or removed.
2. THE `OperatorMemoryLayer` SHALL maintain a `_rust_cache: dict[str, tuple[list[str], list[str], float]]` mapping `operator_id → (preconditions, effects, confidence)` that is updated incrementally when operators are stored or updated.
3. WHEN a Rust call requires a full operator or transition list, THE respective layer SHALL construct it from its `_rust_cache` dict values in a single `list(cache.values())` call, with no per-item `dataclass` attribute access.
4. WHEN an operator's `confidence` is updated via `update_operator_stats()`, THE `OperatorMemoryLayer` SHALL update the cached tuple for that operator in `_rust_cache` before the method returns.
5. THE `_rust_cache` structures SHALL use only Python built-in types (`list`, `str`, `float`, `tuple`) so that no serialisation step is needed before passing them to Rust via PyO3.

---

### Requirement 12: Maintain correctness across Python fallback and Rust-accelerated paths

**User Story:** As a developer, I want every Rust-accelerated hotpath to have an identical Python fallback, so that the system remains fully functional when the Rust extension is unavailable (e.g., on a new platform before recompilation).

#### Acceptance Criteria

1. THE `BrainMemory` SHALL detect `puhl_luck_core` availability once at module import time and expose a module-level boolean `RUST_AVAILABLE` that all mixins consult without re-importing.
2. WHEN `RUST_AVAILABLE` is `False`, THE system SHALL produce outputs semantically equivalent to the Rust-accelerated path for all seven accelerated hotpaths (Requirements 1–7) and the three new Rust functions (Requirements 8–10).
3. THE test suite SHALL include at least one test per hotpath that asserts numerical agreement (within ±1 × 10⁻⁶ for similarity scores, ±1 × 10⁻⁹ for edge weights) between the Rust and Python outputs on the same input.
4. IF both `RUST_AVAILABLE` is `True` and the Rust call raises a `RuntimeError` or `ValueError` at runtime, THEN THE respective mixin SHALL catch the exception, log a warning to `logging.getLogger(__name__)`, and re-execute the Python fallback for that call.
