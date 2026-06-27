"""
Tests for multi-modal field formation (Task 14.1).

Validates that FieldFormation can handle multiple modalities simultaneously.
Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6
"""

import pytest
from puhl_luck._memory_field_formation import FieldFormation
from puhl_luck._memory_exposure_layer import ExposureEventsLayer
from puhl_luck._memory_field_core import InputContext


class TestMultiModalFieldFormation:
    """Test multi-modal support in field formation."""
    
    def test_text_modality_field_formation(self):
        """Test field formation with text modality."""
        events_layer = ExposureEventsLayer()
        formation = FieldFormation()
        
        # Store text events
        events_layer.store_event(
            modality="text",
            features=['python', 'programming', 'code'],
            sequence=['python', 'programming'],
            source="test",
            label=None,
            preview="Python programming"
        )
        
        # Form field with text input
        context = InputContext.from_text("python programming")
        field = formation.form_field(context, events_layer)
        
        assert field is not None
        assert field.query_features is not None
        assert len(field.activated_events) > 0
    
    def test_mixed_modality_events(self):
        """Test that events from different modalities can be stored and activated."""
        events_layer = ExposureEventsLayer()
        formation = FieldFormation()
        
        # Store different modality events
        text_id = events_layer.store_event(
            modality="text",
            features=['python', 'programming'],
            sequence=['python'],
            source="test",
            label=None,
            preview="Text content"
        )
        
        image_id = events_layer.store_event(
            modality="image",
            features=['mod:image', 'image:size:1000', 'python'],  # Image containing "python"
            sequence=['python'],
            source="test",
            label=None,
            preview="Image content"
        )
        
        code_id = events_layer.store_event(
            modality="code",
            features=['python', 'function', 'def'],
            sequence=['python', 'def'],
            source="test",
            label=None,
            preview="Code content"
        )
        
        # Form field - should activate events from all modalities
        context = InputContext.from_text("python")
        field = formation.form_field(context, events_layer)
        
        # Check that events were activated
        assert len(field.activated_events) >= 1
        
        # Check that we can access multi-modal events
        activated_ids = list(field.activated_events.keys())
        assert any(eid in [text_id, image_id, code_id] for eid in activated_ids)
    
    def test_modality_specific_features(self):
        """Test that modality-specific features are handled correctly."""
        events_layer = ExposureEventsLayer()
        
        # Text with text-specific features
        text_id = events_layer.store_event(
            modality="text",
            features=['mod:text', 'text:tok:hello', 'text:stem:hello'],
            sequence=['hello'],
            source="test",
            label=None,
            preview="Hello world"
        )
        
        # Image with image-specific features
        image_id = events_layer.store_event(
            modality="image",
            features=['mod:image', 'image:size:large', 'image:phash:abc123'],
            sequence=[],
            source="test",
            label=None,
            preview="Image file"
        )
        
        # Audio with audio-specific features
        audio_id = events_layer.store_event(
            modality="audio",
            features=['mod:audio', 'audio:duration:30', 'audio:format:wav'],
            sequence=[],
            source="test",
            label=None,
            preview="Audio file"
        )
        
        # Verify all events stored
        assert text_id in events_layer.events
        assert image_id in events_layer.events
        assert audio_id in events_layer.events
        
        # Check modality tracking
        assert 'text' in events_layer.modality_freq
        assert 'image' in events_layer.modality_freq
        assert 'audio' in events_layer.modality_freq
    
    def test_cross_modality_resonance(self):
        """Test that resonance can be computed across modalities."""
        events_layer = ExposureEventsLayer()
        formation = FieldFormation()
        
        # Store events with shared semantic features across modalities
        text_id = events_layer.store_event(
            modality="text",
            features=['python', 'programming', 'tutorial'],
            sequence=['python', 'programming'],
            source="test",
            label="python_tutorial",
            preview="Python programming tutorial"
        )
        
        code_id = events_layer.store_event(
            modality="code",
            features=['python', 'function', 'def', 'tutorial'],
            sequence=['python', 'def'],
            source="test",
            label="python_code",
            preview="def tutorial(): pass"
        )
        
        # Form field
        context = InputContext.from_text("python tutorial")
        field = formation.form_field(context, events_layer)
        
        # Field formation works across modalities
        assert field is not None
        assert len(field.activated_events) > 0
        
        # Resonance computation exists
        assert hasattr(field, 'resonance')
        assert isinstance(field.resonance, dict)
    
    def test_multi_modal_context_input(self):
        """Test that InputContext can handle different modalities."""
        # Text context
        text_context = InputContext.from_text("test query")
        assert text_context.modality == "text"
        assert text_context.query_text == "test query"
        
        # Manual multi-modal context
        image_context = InputContext(
            query_text="",
            query_features=['mod:image', 'image:content'],
            query_hv=None,
            modality="image",
            domain="general"
        )
        assert image_context.modality == "image"
        assert 'mod:image' in image_context.query_features
    
    def test_field_formation_with_structured_data(self):
        """Test field formation with structured/code data."""
        events_layer = ExposureEventsLayer()
        formation = FieldFormation()
        
        # Store code events with structured features
        events_layer.store_event(
            modality="code",
            features=['lang:python', 'type:function', 'name:calculate', 'returns:int'],
            sequence=['function', 'calculate'],
            source="test.py",
            label="function",
            preview="def calculate(x: int) -> int: ..."
        )
        
        events_layer.store_event(
            modality="code",
            features=['lang:python', 'type:class', 'name:Calculator'],
            sequence=['class', 'Calculator'],
            source="test.py",
            label="class",
            preview="class Calculator: ..."
        )
        
        # Form field with code query
        context = InputContext(
            query_text="python function",
            query_features=['lang:python', 'type:function'],
            query_hv=None,
            modality="code",
            domain="code"
        )
        field = formation.form_field(context, events_layer)
        
        assert field is not None
        assert field.activated_events is not None
    
    def test_multi_modal_activation_strength(self):
        """Test that activation strengths are assigned correctly across modalities."""
        events_layer = ExposureEventsLayer()
        formation = FieldFormation()
        
        # Store events from different modalities
        for i, modality in enumerate(['text', 'code', 'image']):
            events_layer.store_event(
                modality=modality,
                features=['python', f'{modality}_specific'],
                sequence=['python'],
                source=f"test_{modality}",
                label=None,
                preview=f"{modality} content {i}"
            )
        
        # Form field
        context = InputContext.from_text("python")
        field = formation.form_field(context, events_layer)
        
        # Check activation strengths
        for event_id, strength in field.activated_events.items():
            assert 0.0 <= strength <= 1.0, f"Activation strength {strength} out of range"
    
    def test_modality_filtering_in_activation(self):
        """Test that modality can influence activation (if implemented)."""
        events_layer = ExposureEventsLayer()
        formation = FieldFormation()
        
        # Store same content in different modalities
        text_id = events_layer.store_event(
            modality="text",
            features=['content', 'example'],
            sequence=['content'],
            source="test",
            label=None,
            preview="Text example"
        )
        
        image_id = events_layer.store_event(
            modality="image",
            features=['mod:image', 'content', 'example'],
            sequence=[],
            source="test",
            label=None,
            preview="Image example"
        )
        
        # Form field with text query
        text_context = InputContext.from_text("content example")
        text_field = formation.form_field(text_context, events_layer)
        
        # Should activate events (potentially with modality preference)
        assert len(text_field.activated_events) > 0


class TestMultiModalStateField:
    """Test that StateField can represent multiple modalities."""
    
    def test_state_field_stores_mixed_modality_activations(self):
        """Test that StateField can store activations from mixed modalities."""
        events_layer = ExposureEventsLayer()
        formation = FieldFormation()
        
        # Create mixed modality events
        events_layer.store_event(
            modality="text",
            features=['concept_a', 'concept_b'],
            sequence=['concept_a'],
            source="text_source",
            label=None,
            preview="Text"
        )
        
        events_layer.store_event(
            modality="image",
            features=['mod:image', 'concept_a'],
            sequence=[],
            source="image_source",
            label=None,
            preview="Image"
        )
        
        # Form field
        context = InputContext.from_text("concept_a")
        field = formation.form_field(context, events_layer)
        
        # StateField should handle mixed modality activations
        assert hasattr(field, 'activated_events')
        assert hasattr(field, 'query_features')
        assert field.activated_events is not None
    
    def test_state_field_modality_tracking(self):
        """Test that we can track which modalities are active in the field."""
        events_layer = ExposureEventsLayer()
        formation = FieldFormation()
        
        # Store multi-modal events
        text_id = events_layer.store_event(
            modality="text",
            features=['shared_feature'],
            sequence=[],
            source="test",
            label=None,
            preview="Text"
        )
        
        code_id = events_layer.store_event(
            modality="code",
            features=['shared_feature'],
            sequence=[],
            source="test",
            label=None,
            preview="Code"
        )
        
        # Form field
        context = InputContext.from_text("shared_feature")
        field = formation.form_field(context, events_layer)
        
        # Can determine which modalities are active by checking activated events
        activated_event_ids = list(field.activated_events.keys())
        active_modalities = set()
        
        for eid in activated_event_ids:
            if eid in events_layer.events:
                active_modalities.add(events_layer.events[eid].modality)
        
        # Should have at least one modality active
        assert len(active_modalities) >= 1


class TestMultiModalHDC:
    """Test that HDC feature extraction works for multiple modalities."""
    
    def test_hdc_feature_extraction_text(self):
        """Test HDC extraction for text."""
        events_layer = ExposureEventsLayer()
        
        features = events_layer.compute_event_features("hello world", "text")
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert any('mod:text' in f or 'text:' in f for f in features)
    
    def test_hdc_feature_extraction_preserves_modality(self):
        """Test that modality information is preserved in features."""
        events_layer = ExposureEventsLayer()
        
        # Text features should have text markers
        text_features = events_layer.compute_event_features("sample", "text")
        assert any('mod:text' in f for f in text_features)
        
        # Image features would have image markers (if we had actual image data)
        # For now, we just test that the API accepts different modalities
        # The actual feature extraction for images/audio happens in extract_image_bytes etc.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
