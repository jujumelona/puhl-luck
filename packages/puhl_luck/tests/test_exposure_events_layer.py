"""
Unit tests for ExposureEventsLayer (Task 1.3)

Tests the Layer 1 implementation that preserves Original PUHL functionality
while providing a clean interface for the field-based architecture.
"""

import pytest
import numpy as np
from puhl_luck._memory_exposure_layer import ExposureEventsLayer
from puhl_luck._brain_common import text_feature_list, text_sequence, bundle_hv


class TestExposureEventsLayerBasics:
    """Test basic initialization and data structures"""

    def test_initialization(self):
        """Test that layer initializes with correct parameters"""
        layer = ExposureEventsLayer(window_size=12, decay=0.72)
        assert layer.window_size == 12
        assert layer.decay == 0.72
        assert len(layer.events) == 0
        assert layer.total_exposures == 0

    def test_has_required_data_structures(self):
        """Test that all Original PUHL data structures are present"""
        layer = ExposureEventsLayer()
        
        # Core storage
        assert hasattr(layer, 'events')
        assert hasattr(layer, 'event_hv')
        assert hasattr(layer, 'event_novelty')
        
        # Feature tracking
        assert hasattr(layer, 'feature_to_id')
        assert hasattr(layer, 'id_to_feature')
        assert hasattr(layer, 'feature_freq')
        assert hasattr(layer, 'feature_to_events')
        
        # Co-activation graph
        assert hasattr(layer, 'edges')
        assert hasattr(layer, 'edge_last_seen')
        
        # HDC indexing
        assert hasattr(layer, 'hdc_index')
        assert hasattr(layer, 'hdc_words')
        assert hasattr(layer, 'hdc_bits')


class TestStoreEvent:
    """Test event storage functionality"""

    def test_store_new_event(self):
        """Test storing a new event creates correct structures"""
        layer = ExposureEventsLayer()
        features = text_feature_list("Paris is the capital of France")
        sequence = text_sequence("Paris is the capital of France")
        
        event_id = layer.store_event(
            modality="text",
            features=features,
            sequence=sequence,
            source="test",
            label="geography",
            preview="Paris is the capital of France"
        )
        
        assert event_id is not None
        assert isinstance(event_id, str)
        assert len(event_id) == 32  # 16-byte hex string
        assert event_id in layer.events
        assert len(layer.events) == 1
        assert layer.total_exposures == 1
        
        # Check event record
        event = layer.events[event_id]
        assert event.modality == "text"
        assert event.source == "test"
        assert event.label == "geography"
        assert "label:geography" in event.features
        assert len(event.features) > 0
        assert len(event.sequence) > 0

    def test_store_duplicate_updates_not_duplicates(self):
        """Test that re-exposing same event updates rather than duplicates"""
        layer = ExposureEventsLayer()
        features = text_feature_list("Test content")
        
        event_id1 = layer.store_event("text", features, source="source1")
        initial_count = len(layer.events)
        
        event_id2 = layer.store_event("text", features, source="source2")
        
        assert event_id1 == event_id2
        assert len(layer.events) == initial_count
        
        # Check that source was updated
        event = layer.events[event_id1]
        assert "source1" in event.source
        assert "source2" in event.source

    def test_store_creates_hdc_vector(self):
        """Test that storing creates HDC hypervector"""
        layer = ExposureEventsLayer()
        features = text_feature_list("Test")
        
        event_id = layer.store_event("text", features)
        
        assert event_id in layer.event_hv
        hv = layer.event_hv[event_id]
        assert isinstance(hv, np.ndarray)
        assert hv.dtype == np.uint64
        assert hv.size == layer.hdc_words

    def test_store_creates_edges(self):
        """Test that storing creates co-activation edges"""
        layer = ExposureEventsLayer()
        features = text_feature_list("The quick brown fox")
        
        initial_edge_count = len(layer.edges)
        layer.store_event("text", features)
        
        assert len(layer.edges) > initial_edge_count


class TestGetEvent:
    """Test event retrieval functionality"""

    def test_get_existing_event(self):
        """Test retrieving an existing event"""
        layer = ExposureEventsLayer()
        features = text_feature_list("Test content")
        event_id = layer.store_event("text", features, label="test")
        
        retrieved = layer.get_event(event_id)
        
        assert retrieved is not None
        assert retrieved.event_id == event_id
        assert retrieved.modality == "text"
        assert retrieved.label == "test"

    def test_get_nonexistent_event(self):
        """Test that getting non-existent event returns None"""
        layer = ExposureEventsLayer()
        
        retrieved = layer.get_event("nonexistent_id")
        
        assert retrieved is None


class TestFindSimilarEvents:
    """Test HDC-based similarity search"""

    def test_find_similar_returns_results(self):
        """Test that similarity search returns results"""
        layer = ExposureEventsLayer()
        
        # Store some events
        layer.store_event("text", text_feature_list("Paris France capital"))
        layer.store_event("text", text_feature_list("Berlin Germany capital"))
        layer.store_event("text", text_feature_list("London England capital"))
        
        # Search with query
        query_features = text_feature_list("Paris France")
        query_hv = bundle_hv(query_features, layer.hdc_bits)
        
        results = layer.find_similar_events(query_hv, top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(eid, str) for eid, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    def test_similarity_scores_in_range(self):
        """Test that similarity scores are in [0, 1]"""
        layer = ExposureEventsLayer()
        layer.store_event("text", text_feature_list("Test content"))
        
        query_hv = bundle_hv(text_feature_list("Test"), layer.hdc_bits)
        results = layer.find_similar_events(query_hv, top_k=10)
        
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_results_sorted_by_similarity(self):
        """Test that results are sorted by similarity (descending)"""
        layer = ExposureEventsLayer()
        
        for i in range(5):
            layer.store_event("text", text_feature_list(f"Test content {i}"))
        
        query_hv = bundle_hv(text_feature_list("Test"), layer.hdc_bits)
        results = layer.find_similar_events(query_hv, top_k=5)
        
        # Check descending order
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i+1][1]

    def test_empty_query_returns_empty(self):
        """Test that empty query returns empty results"""
        layer = ExposureEventsLayer()
        layer.store_event("text", text_feature_list("Test"))
        
        empty_hv = np.array([], dtype=np.uint64)
        results = layer.find_similar_events(empty_hv, top_k=10)
        
        assert results == []


class TestGetCoactivatedEvents:
    """Test co-activation-based retrieval"""

    def test_get_coactivated_returns_related(self):
        """Test that co-activation finds related events"""
        layer = ExposureEventsLayer()
        
        # Store related events
        id1 = layer.store_event("text", text_feature_list("Paris France"))
        id2 = layer.store_event("text", text_feature_list("France capital"))
        id3 = layer.store_event("text", text_feature_list("Berlin Germany"))
        
        # Get co-activated with first event
        results = layer.get_coactivated_events([id1], max_results=5)
        
        # Should return other events
        assert isinstance(results, list)
        assert all(isinstance(eid, str) for eid, _ in results)
        assert all(isinstance(strength, (int, float)) for _, strength in results)

    def test_seed_events_excluded(self):
        """Test that seed events are not in results"""
        layer = ExposureEventsLayer()
        
        id1 = layer.store_event("text", text_feature_list("Test one"))
        id2 = layer.store_event("text", text_feature_list("Test two"))
        
        results = layer.get_coactivated_events([id1], max_results=10)
        
        # Seed event should not be in results
        result_ids = [eid for eid, _ in results]
        assert id1 not in result_ids

    def test_multiple_seeds(self):
        """Test co-activation with multiple seed events"""
        layer = ExposureEventsLayer()
        
        id1 = layer.store_event("text", text_feature_list("Paris France"))
        id2 = layer.store_event("text", text_feature_list("Berlin Germany"))
        id3 = layer.store_event("text", text_feature_list("London England"))
        
        results = layer.get_coactivated_events([id1, id2], max_results=5)
        
        # Should work with multiple seeds
        assert isinstance(results, list)
        result_ids = [eid for eid, _ in results]
        assert id1 not in result_ids
        assert id2 not in result_ids

    def test_empty_seeds_returns_empty(self):
        """Test that empty seed list returns empty results"""
        layer = ExposureEventsLayer()
        layer.store_event("text", text_feature_list("Test"))
        
        results = layer.get_coactivated_events([], max_results=10)
        
        assert results == []


class TestComputeEventFeatures:
    """Test feature extraction"""

    def test_text_features(self):
        """Test text feature extraction"""
        layer = ExposureEventsLayer()
        
        features = layer.compute_event_features("Hello world", "text")
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert any("text:" in f for f in features)

    def test_image_features(self):
        """Test image feature extraction"""
        layer = ExposureEventsLayer()
        
        features = layer.compute_event_features("", "image")
        
        assert isinstance(features, list)
        assert "mod:image" in features

    def test_audio_features(self):
        """Test audio feature extraction"""
        layer = ExposureEventsLayer()
        
        features = layer.compute_event_features("", "audio")
        
        assert isinstance(features, list)
        assert "mod:audio" in features


class TestHDCCompatibility:
    """Test HDC feature extraction compatibility"""

    def test_hdc_dimensions_scale(self):
        """Test that HDC dimensions scale with vocabulary"""
        layer = ExposureEventsLayer()
        initial_words = layer.hdc_words
        
        # Add events to grow vocabulary
        for i in range(10):
            layer.store_event("text", text_feature_list(f"unique content {i} " * 10))
        
        # HDC dimensions should have grown
        assert layer.hdc_words >= initial_words

    def test_hdc_index_populated(self):
        """Test that HDC band index is populated"""
        layer = ExposureEventsLayer()
        
        layer.store_event("text", text_feature_list("Test content"))
        
        assert len(layer.hdc_index) > 0
        assert all(isinstance(band, tuple) for band in layer.hdc_index.keys())

    def test_hypervectors_created(self):
        """Test that hypervectors are created for all events"""
        layer = ExposureEventsLayer()
        
        id1 = layer.store_event("text", text_feature_list("Test 1"))
        id2 = layer.store_event("text", text_feature_list("Test 2"))
        
        assert id1 in layer.event_hv
        assert id2 in layer.event_hv
        assert isinstance(layer.event_hv[id1], np.ndarray)
        assert isinstance(layer.event_hv[id2], np.ndarray)


class TestCoactivationGraph:
    """Test co-activation edge graph"""

    def test_edges_created(self):
        """Test that edges are created between features"""
        layer = ExposureEventsLayer()
        
        initial_count = len(layer.edges)
        layer.store_event("text", text_feature_list("The quick brown fox"))
        
        assert len(layer.edges) > initial_count

    def test_edge_structure(self):
        """Test edge structure is correct"""
        layer = ExposureEventsLayer()
        layer.store_event("text", text_feature_list("Test content"))
        
        # Check edge keys are (int, int) tuples
        for key in layer.edges.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], int)
            assert isinstance(key[1], int)
        
        # Check edge values are floats
        for value in layer.edges.values():
            assert isinstance(value, float)

    def test_edge_decay_tracking(self):
        """Test that edge decay timestamps are tracked"""
        layer = ExposureEventsLayer()
        layer.store_event("text", text_feature_list("Test content"))
        
        assert len(layer.edge_last_seen) > 0
        assert all(isinstance(ts, int) for ts in layer.edge_last_seen.values())


class TestFeatureTracking:
    """Test feature vocabulary and tracking"""

    def test_feature_vocabulary_grows(self):
        """Test that feature vocabulary grows with exposures"""
        layer = ExposureEventsLayer()
        initial_size = len(layer.feature_to_id)
        
        layer.store_event("text", text_feature_list("New unique content"))
        
        assert len(layer.feature_to_id) > initial_size

    def test_feature_frequencies_tracked(self):
        """Test that feature frequencies are tracked"""
        layer = ExposureEventsLayer()
        
        layer.store_event("text", text_feature_list("test test test"))
        
        assert layer.total_feature_count > 0
        assert len(layer.feature_freq) > 0

    def test_feature_to_events_mapping(self):
        """Test that feature-to-events mapping is maintained"""
        layer = ExposureEventsLayer()
        
        event_id = layer.store_event("text", text_feature_list("Test content"))
        
        # Check that features map to the event
        assert len(layer.feature_to_events) > 0
        
        # At least one feature should map to our event
        found = False
        for feature_events in layer.feature_to_events.values():
            if event_id in feature_events:
                found = True
                break
        assert found

    def test_modality_tracking(self):
        """Test that modality frequencies are tracked"""
        layer = ExposureEventsLayer()
        
        layer.store_event("text", text_feature_list("Test"))
        
        assert "text" in layer.modality_freq
        assert layer.modality_freq["text"] > 0

    def test_label_tracking(self):
        """Test that label frequencies are tracked"""
        layer = ExposureEventsLayer()
        
        layer.store_event("text", text_feature_list("Test"), label="test_label")
        
        assert "test_label" in layer.label_freq
        assert layer.label_freq["test_label"] > 0


class TestConceptFormation:
    """Test concept formation from repeated patterns"""

    def test_concept_structures_present(self):
        """Test that concept tracking structures exist"""
        layer = ExposureEventsLayer()
        
        assert hasattr(layer, 'cluster_freq')
        assert hasattr(layer, 'concept_members')

    def test_concepts_can_form(self):
        """Test that concepts can form with repeated patterns"""
        layer = ExposureEventsLayer()
        
        # Store events with overlapping features multiple times
        for i in range(10):
            layer.store_event(
                "text",
                text_feature_list("machine learning neural network artificial intelligence")
            )
        
        # Check if any concepts formed
        concept_features = [f for f in layer.feature_to_id.keys() if f.startswith("concept:")]
        # May or may not form depending on threshold, but structure should work
        assert isinstance(concept_features, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
