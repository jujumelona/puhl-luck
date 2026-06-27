from __future__ import annotations

from ._brain_common import *


class MemoryExposureMixin:
    def feature_id(self, feature: str) -> int:
        found = self.feature_to_id.get(feature)
        if found is not None:
            return found
        idx = len(self.id_to_feature)
        self.feature_to_id[feature] = idx
        self.id_to_feature.append(feature)
        return idx

    # ------------------------------------------------------------------
    # Transition-based exposure
    # ------------------------------------------------------------------

    def expose_text(self, text: str, source: str = "text", label: Optional[str] = None) -> str:
        """Store text as an exposure event AND record as a transition.

        Every call contributes to the transition memory as a completed state
        (the full text is the "complete" half of some earlier partial context).
        When text is long enough it also produces a partial→complete pair
        automatically by splitting at a natural midpoint.
        """
        features, sequence, preview = self.extract_text(text)
        event_id = self.expose_event("text", features, sequence, source=source, label=label, preview=preview)

        # ---- Build & store the partial → complete transition ----
        self._maybe_store_self_transition(text, features, sequence, source)

        return event_id

    def expose_pair(
        self,
        partial: str,
        complete: str,
        continuation: Optional[str] = None,
        source: str = "text",
        label: Optional[str] = None,
        domain: str = "conversation",
        modality: str = "text",
    ) -> Tuple[str, str]:
        """Primary method for transition learning.

        Stores a (partial context → continuation → completed state) triple.
        The *partial* is what we observed first (incomplete state).
        The *continuation* is an optional intermediate step.
        The *complete* is the final resolved state.

        Examples
        --------
        Question fragment → answer fragment → full Q&A:
            brain.expose_pair("what is HDC?", "HDC 는 초고차원 벡터를 이용한 ...메모리다.", domain="conversation")

        Incomplete code → next line → runnable code:
            brain.expose_pair("def foo(x):", "    return x * 2", "def foo(x):\n    return x * 2", domain="code")

        Partial explanation → next step → full explanation:
            brain.expose_pair("먼저 활성화 에너지를", "측정하고 그 다음 결과를 비교한다.", domain="reasoning")
        """
        # 1. Store both sides as exposure events
        p_features, p_seq, p_preview = self.extract_text(partial)
        c_features, c_seq, c_preview = self.extract_text(complete)

        partial_id = self.expose_event("text", p_features, p_seq, source=source, label=label, preview=p_preview)
        complete_id = self.expose_event("text", c_features, c_seq, source=source, label=label, preview=c_preview)

        if continuation:
            cont_features, cont_seq, cont_preview = self.extract_text(continuation)
            self.expose_event("text", cont_features, cont_seq, source=source, preview=cont_preview)

        # 2. Build StateField representations for transition storage
        partial_field = self._features_to_state_field(p_features, p_seq)
        complete_field = self._features_to_state_field(c_features, c_seq)

        # 3. Store the transition in the TransitionMemoryLayer
        transition_layer = getattr(self, "_transition_layer", None)
        if transition_layer is not None:
            transition_layer.store_transition(
                partial=partial_field,
                complete=complete_field,
                modality=modality,
                domain=domain,
            )
            self._pending_induction_count = getattr(self, "_pending_induction_count", 0) + 1

        # 4. Learn the actual surface answer for this partial state.
        # This is the missing bridge from internal transition memory to real output.
        surface_layer = getattr(self, "_surface_layer", None)
        if surface_layer is not None:
            state_pattern = "|".join(sorted(p_features[:10]))
            surface_text = (continuation or complete).strip()
            surface_features = cont_features if continuation else c_features
            surface_layer.learn_surface_form(
                state_pattern=state_pattern,
                surface_text=surface_text,
                modality=modality,
                features=surface_features,
            )

        # 5. TOKEN-LEVEL LEARNING for code domain
        # Split complete answer into tokens and store token transitions
        if domain == "code" and transition_layer is not None:
            self._store_token_level_transitions(partial, complete, continuation, transition_layer)

        # 6. Periodically induce operators from accumulated transitions
        self._maybe_induce_operators()

        return partial_id, complete_id

    # ------------------------------------------------------------------
    # Helpers for transition support
    # ------------------------------------------------------------------

    def _features_to_state_field(self, features: List[str], sequence: List[str]):
        """Build a minimal StateField from raw feature/sequence lists."""
        from ._memory_field_core import StateField

        try:
            from ._brain_hdc import bundle_hv
            hv = bundle_hv([self.feature_id(f) for f in features[:64]], self.hdc_bits)
        except Exception:
            import numpy as np
            hv = np.zeros(max(1, getattr(self, "hdc_bits", 256)), dtype=np.int8)

        return StateField(
            query_features=features[:64],
            query_hv=hv,
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[],
            resonance={},
            field_energy=None,
            previous_outputs=[],
            iteration=0,
        )

    def _maybe_store_self_transition(
        self,
        text: str,
        features: List[str],
        sequence: List[str],
        source: str,
    ) -> None:
        """Auto-split long text into partial→complete and store the transition."""
        transition_layer = getattr(self, "_transition_layer", None)
        if transition_layer is None:
            return

        tokens = [s.split(":", 1)[1] for s in sequence if s.startswith("text:")]
        if len(tokens) < 6:
            return  # Too short to split meaningfully

        split = len(tokens) // 2
        partial_tokens = tokens[:split]
        complete_tokens = tokens

        partial_text = " ".join(partial_tokens)
        complete_text = " ".join(complete_tokens)

        p_features, p_seq, _ = self.extract_text(partial_text)
        c_features, c_seq, _ = self.extract_text(complete_text)

        partial_field = self._features_to_state_field(p_features, p_seq)
        complete_field = self._features_to_state_field(c_features, c_seq)

        # Infer domain from source
        domain = "code" if any(s in source for s in (".py", ".js", ".ts", ".rs", ".cpp", ".c", ".java")) else "conversation"

        transition_layer.store_transition(
            partial=partial_field,
            complete=complete_field,
            modality="text",
            domain=domain,
        )
        
        # CRITICAL: Learn surface form mapping (state → actual text)
        surface_layer = getattr(self, "_surface_layer", None)
        if surface_layer is not None:
            # State pattern: combination of partial features
            state_pattern = "|".join(sorted(p_features[:10]))
            
            # Surface text: the ANSWER/CONTINUATION only (not full complete)
            surface_text = complete_text.strip()
            
            # Learn this mapping
            surface_layer.learn_surface_form(
                state_pattern=state_pattern,
                surface_text=surface_text,
                modality="text",
                features=c_features,
            )
        
        self._pending_induction_count = getattr(self, "_pending_induction_count", 0) + 1

    def _maybe_induce_operators(self, force: bool = False) -> None:
        """Trigger operator induction from accumulated transitions (every N transitions)."""
        induction_interval = 6  # induce operators every 6 new transitions
        pending = getattr(self, "_pending_induction_count", 0)
        if not force and pending < induction_interval:
            return

        transition_layer = getattr(self, "_transition_layer", None)
        operator_layer = getattr(self, "_operator_layer", None)
        induction = getattr(self, "_operator_induction", None)

        if transition_layer is None or operator_layer is None or induction is None:
            return

        transitions = list(transition_layer.transitions.values())
        if len(transitions) < 3:
            return

        new_operators = induction.induce_from_history(
            transitions,
            min_pattern_count=2,
            similarity_threshold=0.4,
        )
        for op in new_operators:
            operator_layer.store_operator(op)

        self._pending_induction_count = 0

    def _store_token_level_transitions(
        self,
        partial: str,
        complete: str,
        continuation: Optional[str],
        transition_layer,
    ) -> None:
        """
        Store token-level transitions for code generation.
        
        KEY: Store only the NEW tokens (answer part), not the entire complete text.
        """
        from ._memory_tokenization import tokenize_code
        
        # Determine what's NEW (the answer/continuation)
        if continuation:
            # Explicit continuation provided
            answer_text = continuation
        else:
            # Extract answer from complete by removing partial prefix
            # Simple heuristic: if complete starts with partial, take remainder
            if complete.startswith(partial):
                answer_text = complete[len(partial):].lstrip()
            else:
                # Fallback: use complete as-is
                answer_text = complete
        
        # Tokenize
        try:
            partial_tokens = tokenize_code(partial)
            answer_tokens = tokenize_code(answer_text)
        except Exception:
            partial_tokens = partial.split()
            answer_tokens = answer_text.split()
        
        # Store: partial + answer[:i] → answer[i]
        for i in range(len(answer_tokens)):
            context = partial_tokens + answer_tokens[:i]
            next_token = answer_tokens[i]
            
            transition_layer.store_token_transition(
                context_tokens=context,
                next_token=next_token,
                modality="code",
                domain="code",
            )

    def expose_file(self, path: str | Path, label: Optional[str] = None) -> str:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            return self.expose_text(read_text_file_as_one_event(p), source=str(p), label=label)
        data = p.read_bytes()
        if suffix in IMAGE_SUFFIXES:
            features, sequence, preview = self.extract_image_bytes(data, suffix=suffix, source=str(p))
            modality = "image"
        elif suffix in AUDIO_SUFFIXES:
            features, sequence, preview = self.extract_audio_bytes(data, suffix=suffix, source=str(p))
            modality = "audio"
        else:
            features, sequence, preview = self.extract_binary(data, suffix=suffix, source=str(p))
            modality = "bytes"
        return self.expose_event(modality, features, sequence, source=str(p), label=label, preview=preview)

    def expose_event(
        self,
        modality: str,
        features: Iterable[str],
        sequence: Iterable[str] = (),
        source: str = "",
        label: Optional[str] = None,
        preview: str = "",
    ) -> str:
        uniq = list(dict.fromkeys(str(f) for f in features if f))
        seq = [str(s) for s in sequence if s]
        if label:
            uniq.append(f"label:{label.lower()}")
        identity_features = list(uniq)
        event_id = stable_id(json.dumps([modality, label, identity_features[:128], seq[:128]], ensure_ascii=False))
        novelty = self.novelty_score(uniq)
        now = self.total_exposures + 1
        existing = self.events.get(event_id)

        if existing is not None:
            if source and source not in existing.source.split(" | "):
                existing.source = " | ".join([existing.source, source]) if existing.source else source
            existing.last_accessed_at = now
            existing.novelty = max(existing.novelty, novelty)
            if preview and preview not in existing.preview:
                existing.preview = (existing.preview + " | " + preview)[:240] if existing.preview else preview[:240]
            self.event_novelty[event_id] = existing.novelty
            for feature in identity_features:
                fid = self.feature_id(feature)
                self.feature_freq[feature] += 1
                self.total_feature_count += 1
                self.feature_to_events[fid][event_id] += 1
                top_events = self.feature_top_events[fid]
                top_events[event_id] += 1
                if len(top_events) > self.dynamic_rank_event_cap() * 2:
                    self.feature_top_events[fid] = Counter(dict(top_events.most_common(self.dynamic_rank_event_cap())))
            self.total_exposures += 1
            self.learn_order_trace(seq)
            self.clear_rank_caches()
            self.remember_short_term(event_id)
            return event_id

        ids = [self.feature_id(f) for f in uniq]
        concept_features, concept_ids = self.observe_concepts(ids)
        if concept_features:
            uniq.extend(concept_features)
            ids.extend(concept_ids)
        self.refresh_dynamic_hdc_if_needed(extra_events=1)
        event_vec = self.bundle_event(uniq, seq)
        rec = EventRecord(event_id, modality, source, label, uniq, seq, preview[:240], novelty, event_vec, now, now)
        self.events[event_id] = rec
        self.event_novelty[event_id] = novelty
        self.event_content_sets[event_id] = set(content_features(rec.features))
        self.event_hv[event_id] = rec.hv
        self.index_event_hv(event_id, rec.hv)
        self.modality_freq[modality] += 1
        if label:
            self.label_freq[label] += 1

        for f, fid in zip(uniq, ids):
            self.feature_freq[f] += 1
            self.total_feature_count += 1
            self.feature_to_events[fid][event_id] += 1
            top_events = self.feature_top_events[fid]
            top_events[event_id] += 1
            if len(top_events) > self.dynamic_rank_event_cap() * 2:
                self.feature_top_events[fid] = Counter(dict(top_events.most_common(self.dynamic_rank_event_cap())))

        adaptive_window = max(2, min(self.window_size * 2, int(round(self.window_size * (0.5 + novelty)))))
        edge_gain = self.surprisal_gain(uniq)
        for i, left in enumerate(ids):
            limit = min(len(ids), i + adaptive_window + 1)
            for j in range(i + 1, limit):
                right = ids[j]
                if left == right:
                    continue
                weight = edge_gain * (self.decay ** (j - i - 1))
                self.add_edge(left, right, weight)
                self.add_edge(right, left, weight * 0.35)

        self.total_exposures += 1
        self.learn_order_trace(seq)
        self.clear_rank_caches()
        self.remember_short_term(event_id)
        self._updates_since_prune += len(ids)
        if self._updates_since_prune >= 4096:
            self.prune()
            self._updates_since_prune = 0
        return event_id

    def learn_order_trace(self, seq: List[str]) -> None:
        tokens = [symbol.split(":", 1)[1] for symbol in seq if symbol.startswith("text:")]
        max_order = self.dynamic_generation_order()
        if tokens:
            self.sequence_starts[tokens[0]] += 1
        for pos, token in enumerate(tokens):
            if not token:
                continue
            self.token_unigrams[token] += 1
            for order in range(1, min(max_order, pos) + 1):
                context = tuple(tokens[pos - order:pos])
                if context:
                    self.order_contexts[context][token] += 1
            if pos > 0 and tokens[pos - 1] and tokens[pos - 1] != token:
                self.token_successors[tokens[pos - 1]][token] += 1
        if len(self.token_unigrams) > 50000:
            self.token_unigrams = Counter(dict(self.token_unigrams.most_common(50000)))
        if len(self.sequence_starts) > 20000:
            self.sequence_starts = Counter(dict(self.sequence_starts.most_common(20000)))
        if len(self.token_successors) > 20000:
            for token, rows in list(self.token_successors.items()):
                self.token_successors[token] = Counter(dict(rows.most_common(32)))
        if len(self.order_contexts) > 50000:
            for context, rows in list(self.order_contexts.items()):
                self.order_contexts[context] = Counter(dict(rows.most_common(32)))

