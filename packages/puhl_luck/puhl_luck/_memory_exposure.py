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

    def expose_text(self, text: str, source: str = "text", label: Optional[str] = None) -> str:
        features, sequence, preview = self.extract_text(text)
        return self.expose_event("text", features, sequence, source=source, label=label, preview=preview)

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

