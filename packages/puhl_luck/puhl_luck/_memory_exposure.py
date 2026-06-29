
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


    def expose_event(self, modality: str, features: List[str], sequence: List[str], source: str = 'text', label: Optional[str] = None, preview: str = '') -> str:
        """Store one event in the legacy event graph for ranking/stats.

        This is not a generation path. Generation uses only _logit_generator.
        """
        self.refresh_dynamic_hdc_if_needed(extra_features=len(features), extra_events=1)
        ids = [self.feature_id(f) for f in features]
        seq_ids = [self.feature_id(s) for s in sequence[: max(0, getattr(self, 'window_size', 12))]]
        novelty = self.novelty_score(features) if hasattr(self, 'novelty_score') else 1.0
        hv = self.bundle_event(features, sequence[:32]) if hasattr(self, 'bundle_event') else np.zeros(max(1, getattr(self, 'hdc_bits', 256)), dtype=np.int8)
        raw = f"{modality}|{source}|{label}|{preview}|{self.total_exposures}|{len(self.events)}"
        event_id = stable_id(raw, 16)
        while event_id in self.events:
            raw += 'x'
            event_id = stable_id(raw, 16)
        rec = EventRecord(
            event_id=event_id,
            modality=modality,
            source=source,
            label=label,
            features=list(features),
            sequence=list(sequence),
            preview=preview,
            novelty=novelty,
            hv=hv,
            created_at=self.total_exposures,
            last_accessed_at=self.total_exposures,
        )
        self.events[event_id] = rec
        self.event_novelty[event_id] = novelty
        self.event_content_sets[event_id] = set(features)
        self.event_hv[event_id] = hv
        try:
            self.index_event_hv(event_id, hv)
        except Exception:
            pass
        self.total_exposures += 1
        self.modality_freq[modality] += 1
        if label is not None:
            self.label_freq[label] += 1
        for f in features:
            self.feature_freq[f] += 1
            self.total_feature_count += 1
        for fid in ids[:256]:
            self.feature_to_events[fid][event_id] += 1
            self.feature_top_events[fid][event_id] += 1
        # lightweight coactivation graph, bounded window
        window = list(dict.fromkeys(ids[: max(2, getattr(self, 'window_size', 12))] + seq_ids[:4]))
        gain = self.surprisal_gain(features) if hasattr(self, 'surprisal_gain') else 1.0
        for i, a in enumerate(window):
            for b in window[i + 1:]:
                try:
                    self.add_edge(a, b, gain)
                    self.add_edge(b, a, gain)
                except Exception:
                    pass
        # old token successor stats for compatibility only
        toks = [s.split(':', 1)[1] for s in sequence if isinstance(s, str) and s.startswith('text:')]
        if toks:
            self.sequence_starts[toks[0]] += 1
            for a, b in zip(toks, toks[1:]):
                self.token_unigrams[a] += 1
                self.token_successors[a][b] += 1
            self.token_unigrams[toks[-1]] += 1
        try:
            self.observe_concepts(ids)
            self.remember_short_term(event_id)
        except Exception:
            pass
        self.clear_rank_caches()
        return event_id

    def expose_text(self, text: str, source: str = 'text', label: Optional[str] = None) -> str:
        features, sequence, preview = self.extract_text(text)
        event_id = self.expose_event('text', features, sequence, source=source, label=label, preview=preview)
        lg = getattr(self, '_logit_generator', None)
        if lg is not None and text:
            toks = str(text)
            if len(toks.strip()) > 0:
                # P62: real corpus next-token exposure. No raw answer path, no grammar path.
                if hasattr(lg, 'learn_sequence'):
                    lg.learn_sequence(toks, field_features=features[:128] if features else None, structural_targets_only=False)
                else:
                    lg.learn(input_text='[TEXT_EXPOSURE]', target_text=toks, field_features=features[:128] if features else None)
        return event_id

    def expose_corpus(self, texts, source: str = 'corpus', label: Optional[str] = None, structural_targets_only: bool = False, max_items: Optional[int] = None):
        """Expose a list/iterable of texts before evaluation.

        This is a convenience wrapper for public text/code data exposure. It uses
        the same weightless next-token count engine as expose_text/expose_pair.
        """
        stats = {'items': 0, 'events': 0, 'transitions_added': 0}
        lg = getattr(self, '_logit_generator', None)
        buffered_texts = []
        for idx, text in enumerate(texts):
            if max_items is not None and idx >= int(max_items):
                break
            if text is None or not str(text).strip():
                continue
            st = str(text)
            features, sequence, preview = self.extract_text(st)
            self.expose_event('text', features, sequence, source=source, label=label, preview=preview)
            stats['events'] += 1
            stats['items'] += 1
            buffered_texts.append(st)
        if lg is not None and buffered_texts:
            if hasattr(lg, 'learn_sequences_many'):
                r = lg.learn_sequences_many(buffered_texts, structural_targets_only=structural_targets_only)
                stats['transitions_added'] += int(r.get('transitions_added', 0))
                stats['p75_batch_sequence_learning'] = r
            elif hasattr(lg, 'learn_sequence'):
                for st in buffered_texts:
                    r = lg.learn_sequence(st, structural_targets_only=structural_targets_only)
                    stats['transitions_added'] += int(r.get('transitions_added', 0))
        return stats

    def expose_pair(
        self,
        partial: str,
        complete: str,
        continuation: Optional[str] = None,
        source: str = 'text',
        label: Optional[str] = None,
        domain: str = 'conversation',
        modality: str = 'text',
    ) -> Tuple[str, str]:
        p_features, p_seq, p_preview = self.extract_text(partial)
        c_features, c_seq, c_preview = self.extract_text(complete)

        partial_id = self.expose_event('text', p_features, p_seq, source=source, label=label, preview=p_preview)
        complete_id = self.expose_event('text', c_features, c_seq, source=source, label=label, preview=c_preview)

        target_text = continuation if continuation is not None else complete
        lg = getattr(self, '_logit_generator', None)
        if lg is not None:
            # Supervised prompt -> target next-token evidence.
            lg.learn(input_text=partial, target_text=target_text, field_features=p_features[:128] if p_features else None)
            # P62: add data-exposure evidence from the target sequence itself.
            # For code-like targets, only structural/keyword targets are added so
            # arbitrary function/variable names from one task do not become global answers.
            if hasattr(lg, 'learn_sequence'):
                st = str(target_text)
                code_like = ('def ' in st) or ('return' in st) or ('\n' in st and any(x in st for x in ('(', ')', ':', '=')))
                lg.learn_sequence(st, field_features=c_features[:128] if c_features else None, structural_targets_only=bool(code_like))

        # Diagnostic transition only; never used by generate().
        transition_layer = getattr(self, '_transition_layer', None)
        if transition_layer is not None:
            try:
                transition_layer.store_transition(
                    partial=self._features_to_state_field(p_features, p_seq),
                    complete=self._features_to_state_field(c_features, c_seq),
                    modality=modality,
                    domain=domain,
                )
            except Exception:
                pass
        return partial_id, complete_id

    def _features_to_state_field(self, features: List[str], sequence: List[str]):
        from ._memory_field_core import StateField
        try:
            from ._brain_hdc import bundle_hv
            hv = bundle_hv([self.feature_id(f) for f in features[:64]], self.hdc_bits)
        except Exception:
            import numpy as np
            hv = np.zeros(max(1, getattr(self, 'hdc_bits', 256)), dtype=np.int8)
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
