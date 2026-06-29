from __future__ import annotations

from ._brain_common import *


class MemoryMicroExportMixin:
    def _compact_micro_rank_bytes(
        self,
        max_events: int,
        max_features: int,
        event_feature_cap: Optional[int],
        postings_per_feature: Optional[int],
        hash_bits: int = 16,
    ) -> bytes:
        max_events = max(1, int(max_events))
        max_features = max(1, int(max_features))
        feature_cap = None if event_feature_cap is None else max(1, int(event_feature_cap))
        posting_cap = None if postings_per_feature is None else max(1, int(postings_per_feature))
        selected_hash_bits = 32 if int(hash_bits) == 32 else 16
        event_ids = {
            eid: idx
            for idx, (eid, _) in enumerate(
                sorted(self.event_novelty.items(), key=lambda item: item[1], reverse=True)[:max_events]
            )
        }
        merged_rows: Dict[int, Counter[int]] = defaultdict(Counter)
        for fid, rows in self.feature_top_events.items():
            feature = self.id_to_feature[fid] if fid < len(self.id_to_feature) else ""
            h = micro_feature_hash(feature, bits=selected_hash_bits)
            if h is None:
                continue
            for eid, count in rows.items():
                compact_eid = event_ids.get(eid)
                if compact_eid is not None:
                    merged_rows[h][compact_eid] += int(count)
        ranked_features = sorted(
            merged_rows.items(),
            key=lambda item: (sum(item[1].values()), len(item[1])),
            reverse=True,
        )[:max_features]
        selected_feature_hashes = {h for h, _ in ranked_features}
        top_rows: Dict[int, List[Tuple[int, int]]] = {}
        for h, counts in ranked_features:
            rows = counts.most_common(posting_cap)
            packed = [(eid, count) for eid, count in rows]
            if packed:
                top_rows[h] = packed
        event_rows: Dict[int, List[int]] = {}
        for eid, compact_eid in event_ids.items():
            feats = self.event_content_sets.get(eid)
            if feats is None and eid in self.events:
                feats = set(content_features(self.events[eid].features))
            all_hashes = [h for h in (micro_feature_hash(feature, bits=selected_hash_bits) for feature in (feats or ())) if h is not None]
            preferred = sorted({h for h in all_hashes if h in selected_feature_hashes})
            if feature_cap is None:
                extra = sorted({h for h in all_hashes if h not in selected_feature_hashes})
                preferred.extend(extra)
                hashes = preferred
            elif len(preferred) < feature_cap:
                extra = sorted({h for h in all_hashes if h not in selected_feature_hashes})
                preferred.extend(extra[: max(0, feature_cap - len(preferred))])
                hashes = preferred[:feature_cap]
            else:
                hashes = preferred[:feature_cap]
            if hashes:
                event_rows[compact_eid] = hashes
        out = bytearray()
        out.extend(b"PMR2")
        write_varuint(out, selected_hash_bits)
        write_varuint(out, len(top_rows))
        for h, rows in sorted(top_rows.items()):
            write_varuint(out, h)
            write_varuint(out, len(rows))
            for eid, count in rows:
                write_varuint(out, eid)
                write_varuint(out, count)
        write_varuint(out, len(event_rows))
        for eid, hashes in sorted(event_rows.items()):
            write_varuint(out, eid)
            write_varuint(out, len(hashes))
            for h in hashes:
                write_varuint(out, h)
        return zlib.compress(bytes(out), level=9)

    def compact_micro_rank_bytes(self, max_bytes: Optional[int] = None, hash_bits: int = 16) -> bytes:
        event_count = max(1, len(self.events))
        feature_count = max(1, len(self.feature_freq))
        if max_bytes is None:
            return self._compact_micro_rank_bytes(
                max_events=event_count,
                max_features=feature_count,
                event_feature_cap=None,
                postings_per_feature=None,
                hash_bits=hash_bits,
            )
        attempts = []
        max_events = min(4096, event_count)
        max_features = min(8192, feature_count)
        event_feature_cap = 32
        postings_per_feature = 4
        for _ in range(12):
            attempts.append((max_events, max_features, event_feature_cap, postings_per_feature))
            if max_events <= 32 and max_features <= 128 and event_feature_cap <= 8 and postings_per_feature <= 1:
                break
            max_events = max(32, max_events // 2)
            max_features = max(128, max_features // 2)
            event_feature_cap = max(8, event_feature_cap // 2)
            postings_per_feature = max(1, postings_per_feature // 2)
        best = b""
        for config in attempts:
            payload = self._compact_micro_rank_bytes(*config, hash_bits=hash_bits)
            best = payload
            if max_bytes is None or len(payload) <= max_bytes:
                return payload
        return best

    def save_rank_micro_only(self, path: str | Path, max_bytes: Optional[int] = None, hash_bits: int = 16) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.compact_micro_rank_bytes(max_bytes=max_bytes, hash_bits=hash_bits))

