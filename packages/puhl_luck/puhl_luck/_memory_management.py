from __future__ import annotations

from ._brain_common import *


class MemoryManagementMixin:
    def inspect_events(
        self,
        source: Optional[str] = None,
        modality: Optional[str] = None,
        contains: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        contains_text = str(contains).lower() if contains else ""
        rows = []
        for eid, rec in self.events.items():
            if source and source not in rec.source:
                continue
            if modality and modality != rec.modality:
                continue
            haystack = f"{rec.source}\n{rec.preview}\n{' '.join(rec.features[:64])}".lower()
            if contains_text and contains_text not in haystack:
                continue
            rows.append({
                "event_id": eid,
                "source": rec.source,
                "modality": rec.modality,
                "label": rec.label,
                "preview": rec.preview,
                "feature_count": len(rec.features),
                "created_at": rec.created_at,
                "last_accessed_at": rec.last_accessed_at,
            })
            if len(rows) >= max(1, limit):
                break
        return rows

    def rebuild_runtime_indexes(self) -> None:
        self.feature_to_events = defaultdict(Counter)
        self.feature_top_events = defaultdict(Counter)
        self.event_content_sets = {}
        self.event_hv = {}
        self.token_successors = defaultdict(Counter)
        self.order_contexts = defaultdict(Counter)
        self.hdc_words = dynamic_hdc_words(len(self.feature_to_id), len(self.events))
        self.hdc_bits = self.hdc_words * HDC_WORD_BITS
        self.hdc_index = defaultdict(set)
        self.hdc_indexed_bands = hdc_band_count(self.hdc_words, len(self.events))
        for eid, rec in self.events.items():
            content = set(content_features(rec.features))
            self.event_content_sets[eid] = content
            for feature in rec.features:
                fid = self.feature_to_id.get(feature)
                if fid is not None:
                    self.feature_to_events[fid][eid] += 1
                    self.feature_top_events[fid][eid] += 1
            rec.hv = self.bundle_event(rec.features, rec.sequence)
            self.event_hv[eid] = rec.hv
            self.index_event_hv(eid, rec.hv)
            self.learn_order_trace(rec.sequence)
        cap = self.dynamic_rank_event_cap()
        for fid, rows in list(self.feature_top_events.items()):
            self.feature_top_events[fid] = Counter(dict(rows.most_common(cap)))
        self.short_term_events = [eid for eid in self.short_term_events if eid in self.events][-self.dynamic_short_term_limit():]
        self.clear_rank_caches()

    def forget_events(
        self,
        event_ids: Optional[Iterable[str]] = None,
        source: Optional[str] = None,
        modality: Optional[str] = None,
        contains: Optional[str] = None,
    ) -> Dict[str, Any]:
        explicit_ids = set(str(eid) for eid in (event_ids or ()) if str(eid))
        contains_text = str(contains).lower() if contains else ""
        remove_ids = set()
        for eid, rec in self.events.items():
            matched = eid in explicit_ids if explicit_ids else True
            if matched and source:
                matched = source in rec.source
            if matched and modality:
                matched = modality == rec.modality
            if matched and contains_text:
                haystack = f"{rec.source}\n{rec.preview}\n{' '.join(rec.features[:64])}".lower()
                matched = contains_text in haystack
            if matched:
                remove_ids.add(eid)
        if not remove_ids:
            return {"removed": 0, "event_ids": []}

        removed_features: Counter[str] = Counter()
        removed_modalities: Counter[str] = Counter()
        removed_labels: Counter[str] = Counter()
        for eid in remove_ids:
            rec = self.events.pop(eid, None)
            if rec is None:
                continue
            removed_modalities[rec.modality] += 1
            if rec.label:
                removed_labels[rec.label] += 1
            for feature in rec.features:
                removed_features[feature] += 1
            self.event_novelty.pop(eid, None)

        for feature, count in removed_features.items():
            current = self.feature_freq.get(feature, 0) - count
            if current > 0:
                self.feature_freq[feature] = current
            else:
                self.feature_freq.pop(feature, None)
        self.total_feature_count = max(0, self.total_feature_count - sum(removed_features.values()))
        for modality_name, count in removed_modalities.items():
            current = self.modality_freq.get(modality_name, 0) - count
            if current > 0:
                self.modality_freq[modality_name] = current
            else:
                self.modality_freq.pop(modality_name, None)
        for label_name, count in removed_labels.items():
            current = self.label_freq.get(label_name, 0) - count
            if current > 0:
                self.label_freq[label_name] = current
            else:
                self.label_freq.pop(label_name, None)

        active_feature_ids = {self.feature_to_id[feature] for feature in self.feature_freq if feature in self.feature_to_id}
        self.edges = {
            key: weight for key, weight in self.edges.items()
            if key[0] in active_feature_ids and key[1] in active_feature_ids
        }
        self.edge_last_seen = {key: seen for key, seen in self.edge_last_seen.items() if key in self.edges}
        self.cluster_freq.clear()
        self.concept_members = {
            concept: [fid for fid in members if fid in active_feature_ids]
            for concept, members in self.concept_members.items()
            if concept in self.feature_freq
        }
        self._neighbors_dirty = True
        self.rebuild_runtime_indexes()
        return {"removed": len(remove_ids), "event_ids": sorted(remove_ids)}

