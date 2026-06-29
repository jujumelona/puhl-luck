"""
Layer 1: Exposure Events Layer

Preserves the original PUHL event storage structure, maintaining backward compatibility
while serving as the foundation for the field-based memory architecture.

This layer stores:
- Raw observational data (events)
- HDC feature vectors
- Co-activation edge graph
- Feature frequencies and statistics
"""

from __future__ import annotations

from ._brain_common import *


class ExposureEventsLayer:
    """
    Layer 1 of the Predictive Field Memory system.
    
    Preserves the Original PUHL event storage structure for backward compatibility.
    Stores raw observational data with HDC-based feature extraction and co-activation
    edge tracking.
    
    This layer is designed to be a drop-in replacement for the event storage components
    of the original BrainMemory class while providing a cleaner interface for the
    field-based architecture.
    """

    def __init__(self, window_size: int = 12, decay: float = 0.72):
        """
        Initialize the Exposure Events Layer.
        
        Args:
            window_size: Co-activation window size for edge creation
            decay: Decay factor for edge weight aging
        """
        # Core parameters
        self.window_size = int(window_size)
        self.decay = float(decay)
        
        # Event storage
        self.events: Dict[str, EventRecord] = {}
        self.event_hv: Dict[str, np.ndarray] = {}
        self.event_novelty: Dict[str, float] = {}
        self.event_content_sets: Dict[str, set[str]] = {}
        
        # Feature tracking
        self.feature_freq: Counter[str] = Counter()
        self.total_feature_count = 0
        self.feature_to_id: Dict[str, int] = {}
        self.id_to_feature: List[str] = []
        self.feature_to_events: Dict[int, Counter[str]] = defaultdict(Counter)
        self.feature_top_events: Dict[int, Counter[str]] = defaultdict(Counter)
        
        # Modality and label tracking
        self.modality_freq: Counter[str] = Counter()
        self.label_freq: Counter[str] = Counter()
        
        # Co-activation graph
        self.edges: Dict[Tuple[int, int], float] = {}
        self.edge_last_seen: Dict[Tuple[int, int], int] = {}
        self._neighbors: Dict[int, List[Tuple[int, float]]] = {}
        self._neighbors_dirty = True
        
        # HDC indexing
        self.hdc_words = dynamic_hdc_words(0, 0)
        self.hdc_bits = dynamic_hdc_bits(0, 0)
        self.hdc_indexed_bands = hdc_band_count(self.hdc_words, 0)
        self.hdc_index: Dict[Tuple[int, int], set[str]] = defaultdict(set)
        
        # Concept tracking
        self.cluster_freq: Counter[Tuple[int, ...]] = Counter()
        self.concept_members: Dict[str, List[int]] = {}
        
        # Short-term memory
        self.short_term_events: List[str] = []
        
        # Counters
        self.total_exposures = 0
        self._updates_since_prune = 0

    def store_event(
        self,
        modality: str,
        features: List[str],
        sequence: Optional[List[str]] = None,
        source: str = "",
        label: Optional[str] = None,
        preview: str = "",
    ) -> str:
        """
        Store an event with HDC feature vectors and co-activation edges.
        
        This is the primary method for adding events to Layer 1, preserving the
        original PUHL exposure logic.
        
        Args:
            modality: Event modality (text, image, audio, bytes)
            features: List of feature strings
            sequence: Optional ordered sequence of features
            source: Source identifier (file path, URL, etc.)
            label: Optional label for the event
            preview: Human-readable preview text
            
        Returns:
            event_id: Stable identifier for the stored event
        """
        if sequence is None:
            sequence = []
        
        # Normalize features and sequence
        uniq = list(dict.fromkeys(str(f) for f in features if f))
        seq = [str(s) for s in sequence if s]
        
        # Add label as feature if provided
        if label:
            uniq.append(f"label:{label.lower()}")
        
        # Generate stable event ID from identity features
        identity_features = list(uniq)
        event_id = stable_id(
            json.dumps([modality, label, identity_features[:128], seq[:128]], ensure_ascii=False)
        )
        
        # Compute novelty score
        novelty = self._novelty_score(uniq)
        now = self.total_exposures + 1
        
        # Check if event already exists
        existing = self.events.get(event_id)
        if existing is not None:
            # Update existing event
            if source and source not in existing.source.split(" | "):
                existing.source = " | ".join([existing.source, source]) if existing.source else source
            existing.last_accessed_at = now
            existing.novelty = max(existing.novelty, novelty)
            if preview and preview not in existing.preview:
                existing.preview = (
                    (existing.preview + " | " + preview)[:240] if existing.preview else preview[:240]
                )
            
            self.event_novelty[event_id] = existing.novelty
            
            # Update feature tracking
            for feature in identity_features:
                fid = self._feature_id(feature)
                self.feature_freq[feature] += 1
                self.total_feature_count += 1
                self.feature_to_events[fid][event_id] += 1
                
                # Update top events for this feature
                top_events = self.feature_top_events[fid]
                top_events[event_id] += 1
                cap = self._dynamic_rank_event_cap()
                if len(top_events) > cap * 2:
                    self.feature_top_events[fid] = Counter(dict(top_events.most_common(cap)))
            
            self.total_exposures += 1
            self._remember_short_term(event_id)
            return event_id
        
        # Create new event
        ids = [self._feature_id(f) for f in uniq]
        
        # Observe concepts (clusters of co-occurring features)
        concept_features, concept_ids = self._observe_concepts(ids)
        if concept_features:
            uniq.extend(concept_features)
            ids.extend(concept_ids)
        
        # Refresh HDC dimensions if needed
        self._refresh_dynamic_hdc_if_needed(extra_events=1)
        
        # Create HDC hypervector for event
        event_vec = self._bundle_event(uniq, seq)
        
        # Create event record
        rec = EventRecord(
            event_id, modality, source, label, uniq, seq, preview[:240], novelty, event_vec, now, now
        )
        
        # Store event
        self.events[event_id] = rec
        self.event_novelty[event_id] = novelty
        self.event_content_sets[event_id] = set(self._content_features(rec.features))
        self.event_hv[event_id] = rec.hv
        self._index_event_hv(event_id, rec.hv)
        
        # Update modality and label frequencies
        self.modality_freq[modality] += 1
        if label:
            self.label_freq[label] += 1
        
        # Update feature tracking
        for f, fid in zip(uniq, ids):
            self.feature_freq[f] += 1
            self.total_feature_count += 1
            self.feature_to_events[fid][event_id] += 1
            
            # Update top events for this feature
            top_events = self.feature_top_events[fid]
            top_events[event_id] += 1
            cap = self._dynamic_rank_event_cap()
            if len(top_events) > cap * 2:
                self.feature_top_events[fid] = Counter(dict(top_events.most_common(cap)))
        
        # Create co-activation edges
        adaptive_window = max(
            2, min(self.window_size * 2, int(round(self.window_size * (0.5 + novelty))))
        )
        edge_gain = self._surprisal_gain(uniq)
        
        for i, left in enumerate(ids):
            limit = min(len(ids), i + adaptive_window + 1)
            for j in range(i + 1, limit):
                right = ids[j]
                if left == right:
                    continue
                weight = edge_gain * (self.decay ** (j - i - 1))
                self._add_edge(left, right, weight)
                self._add_edge(right, left, weight * 0.35)
        
        # Update counters
        self.total_exposures += 1
        self._remember_short_term(event_id)
        self._updates_since_prune += len(ids)
        
        # Periodic pruning
        if self._updates_since_prune >= 4096:
            self._prune()
            self._updates_since_prune = 0
        
        return event_id

    def get_event(self, event_id: str) -> Optional[EventRecord]:
        """
        Retrieve an event by its ID.
        
        Args:
            event_id: Event identifier
            
        Returns:
            EventRecord or None if not found
        """
        return self.events.get(event_id)

    def find_similar_events(self, query_hv: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find events with similar HDC hypervectors using band-based indexing.
        
        Args:
            query_hv: Query hypervector
            top_k: Number of top results to return
            
        Returns:
            List of (event_id, similarity_score) tuples, sorted by similarity (descending)
        """
        if query_hv.size == 0 or not self.events:
            return []
        
        # Use band-based indexing to find candidate events
        bands = hdc_bands(query_hv, len(self.events))
        candidates: Counter[str] = Counter()
        
        for band in bands:
            for event_id in self.hdc_index.get(band, []):
                candidates[event_id] += 1
        
        # If no candidates from indexing, check all events (fallback)
        if not candidates:
            candidates = Counter({eid: 1 for eid in self.events.keys()})
        
        # Compute actual similarities for top candidates
        # Take more candidates than needed to account for filtering
        top_candidates = [eid for eid, _ in candidates.most_common(min(top_k * 3, len(candidates)))]
        
        results = []
        for event_id in top_candidates:
            event_hv = self.event_hv.get(event_id)
            if event_hv is not None:
                sim = hv_similarity(query_hv, event_hv, self.hdc_bits)
                results.append((event_id, float(sim)))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_coactivated_events(self, event_ids: List[str], max_results: int = 20) -> List[Tuple[str, float]]:
        """
        Find events that are strongly co-activated with the given events.
        
        Uses the co-activation edge graph to find related events based on
        feature co-occurrence patterns.
        
        Args:
            event_ids: List of seed event IDs
            max_results: Maximum number of results to return
            
        Returns:
            List of (event_id, activation_strength) tuples
        """
        if not event_ids:
            return []
        
        # Collect all features from seed events
        seed_features: set[int] = set()
        for event_id in event_ids:
            event = self.events.get(event_id)
            if event:
                for feature in event.features:
                    fid = self.feature_to_id.get(feature)
                    if fid is not None:
                        seed_features.add(fid)
        
        if not seed_features:
            return []
        
        # Propagate activation through edges
        activation: Counter[int] = Counter()
        for fid in seed_features:
            activation[fid] = 1.0
            
            # Add activation from neighbors
            for neighbor_id, weight in self._neighbors_of(fid):
                activation[neighbor_id] += weight
        
        # Map activated features to events
        event_activation: Counter[str] = Counter()
        for fid, act in activation.items():
            if fid >= len(self.id_to_feature):
                continue
            
            # Find events containing this feature
            for eid in self.feature_to_events.get(fid, {}):
                if eid not in event_ids:  # Exclude seed events
                    event_activation[eid] += act
        
        # Return top activated events
        return event_activation.most_common(max_results)

    def compute_event_features(self, content: str, modality: str) -> List[str]:
        """
        Extract features from content using HDC-compatible feature extraction.
        
        This method provides the feature extraction logic that was previously
        embedded in expose_text and other methods.
        
        Args:
            content: Content to extract features from
            modality: Content modality (text, image, audio, bytes)
            
        Returns:
            List of feature strings
        """
        if modality == "text":
            return text_feature_list(content)
        else:
            # For other modalities, return basic features
            # More sophisticated extraction would go here
            return [f"mod:{modality}"]

    # =========================================================================
    # Internal helper methods (prefixed with _)
    # =========================================================================

    def _feature_id(self, feature: str) -> int:
        """Get or create feature ID."""
        found = self.feature_to_id.get(feature)
        if found is not None:
            return found
        idx = len(self.id_to_feature)
        self.feature_to_id[feature] = idx
        self.id_to_feature.append(feature)
        return idx

    def _bundle_event(self, features: List[str], sequence: List[str]) -> np.ndarray:
        """Create HDC hypervector by bundling features."""
        return bundle_hv(features, self.hdc_bits)

    def _index_event_hv(self, event_id: str, event_vec: np.ndarray, start_word: int = 0) -> None:
        """Add event to HDC band index."""
        for band in hdc_bands(event_vec, max(1, len(self.events)), start_word=start_word):
            self.hdc_index[band].add(event_id)

    def _refresh_dynamic_hdc_if_needed(
        self, extra_features: int = 0, extra_events: int = 0
    ) -> None:
        """Refresh HDC dimensions if vocabulary or event count has grown."""
        target_words = dynamic_hdc_words(
            len(self.feature_to_id) + extra_features, len(self.events) + extra_events
        )
        target_bands = hdc_band_count(target_words, len(self.events) + extra_events)
        
        if target_words <= self.hdc_words and target_bands <= self.hdc_indexed_bands:
            return
        
        old_bands = self.hdc_indexed_bands
        old_words = self.hdc_words
        self.hdc_words = target_words
        self.hdc_bits = target_words * HDC_WORD_BITS
        self.hdc_indexed_bands = target_bands
        
        # Recompute hypervectors for all events with new dimensions
        for eid, rec in self.events.items():
            full_vec = self._bundle_event(rec.features, rec.sequence)
            old_vec = self.event_hv.get(eid)
            if old_vec is not None and old_vec.size == old_words:
                full_vec[:old_words] = old_vec
            rec.hv = full_vec
            self.event_hv[eid] = full_vec
            self._index_event_hv(eid, full_vec, start_word=old_bands)

    def _observe_concepts(self, ids: List[int]) -> Tuple[List[str], List[int]]:
        """Detect and track concept clusters from co-occurring features."""
        content_ids = [
            fid
            for fid in dict.fromkeys(ids)
            if fid < len(self.id_to_feature)
            and not self.id_to_feature[fid].startswith(("mod:", "label:", "concept:"))
        ]
        
        if len(content_ids) < 3:
            return [], []
        
        width = max(3, int(math.sqrt(len(content_ids))) + 1)
        cluster = tuple(sorted(content_ids[:width]))
        self.cluster_freq[cluster] += 1
        
        threshold = self._dynamic_concept_threshold()
        if self.cluster_freq[cluster] < threshold:
            return [], []
        
        concept = f"concept:{stable_id(','.join(map(str, cluster)), 8)}"
        concept_id = self._feature_id(concept)
        self.concept_members[concept] = list(cluster)
        
        # Create edges between concept and members
        gain = math.log1p(self.cluster_freq[cluster])
        for member in cluster:
            self._add_edge(member, concept_id, gain)
            self._add_edge(concept_id, member, gain)
        
        return [concept], [concept_id]

    def _dynamic_concept_threshold(self) -> int:
        """Compute dynamic threshold for concept formation."""
        return max(2, int(math.sqrt(max(1, self.total_exposures + 1))))

    def _add_edge(self, left: int, right: int, weight: float) -> None:
        """Add or update a co-activation edge with decay."""
        if left == right:
            return
        
        key = (left, right)
        last_seen = self.edge_last_seen.get(key, self.total_exposures)
        age = max(0, self.total_exposures - last_seen)
        aged = self.edges.get(key, 0.0) * (self.decay ** min(age, 64))
        self.edges[key] = aged + weight
        self.edge_last_seen[key] = self.total_exposures
        self._neighbors_dirty = True

    def _neighbors_of(self, fid: int) -> List[Tuple[int, float]]:
        """Get neighbors of a feature from the edge graph."""
        if self._neighbors_dirty:
            self._rebuild_neighbors()
        return self._neighbors.get(fid, [])

    def _rebuild_neighbors(self) -> None:
        """Rebuild neighbor adjacency list from edges."""
        graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (left, right), weight in self.edges.items():
            graph[left].append((right, weight))
        self._neighbors = dict(graph)
        self._neighbors_dirty = False

    def _remember_short_term(self, event_id: str) -> None:
        """Add event to short-term memory buffer."""
        self.short_term_events.append(event_id)
        limit = self._dynamic_short_term_limit()
        if len(self.short_term_events) > limit:
            del self.short_term_events[: len(self.short_term_events) - limit]

    def _dynamic_short_term_limit(self) -> int:
        """Compute dynamic size of short-term memory."""
        return max(4, int(math.sqrt(max(1, len(self.events)))))

    def _novelty_score(self, features: List[str]) -> float:
        """Compute novelty score based on feature rarity."""
        if not features or not self.feature_freq:
            return 1.0
        
        total = max(1, sum(self.feature_freq.values()))
        vocab = max(1, len(self.feature_freq))
        surprisal = 0.0
        
        for feature in features:
            prob = (self.feature_freq.get(feature, 0) + 1.0) / (total + vocab)
            surprisal += -math.log2(prob)
        
        return max(0.1, min(2.0, (surprisal / max(1, len(features))) / 8.0))

    def _surprisal_gain(self, features: List[str]) -> float:
        """Compute edge weight gain from novelty."""
        return 0.5 + self._novelty_score(features)

    def _dynamic_rank_event_cap(self) -> int:
        """Compute dynamic cap for top events per feature."""
        return max(8, min(64, int(math.sqrt(max(1, len(self.events))))))

    def _content_features(self, features: List[str]) -> List[str]:
        """Filter to content features (exclude metadata prefixes)."""
        return [f for f in features if not f.startswith(("mod:", "label:", "concept:"))]

    def _prune(self) -> None:
        """Prune low-weight edges (basic implementation)."""
        # Implement targeted forgetting if needed
        # For now, just clear the neighbors cache
        self._neighbors_dirty = True


__all__ = ["ExposureEventsLayer"]
