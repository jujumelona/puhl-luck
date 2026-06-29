"""
Manual test to verify resonance computation is working properly.
"""

from puhl_luck._memory_state_field import StateFieldLayer
from puhl_luck._memory_exposure_layer import ExposureEventsLayer
from puhl_luck._memory_field_core import InputContext

def test_resonance_with_real_data():
    """Test resonance computation with actual event data."""
    # Setup layers
    exposure_layer = ExposureEventsLayer(window_size=1000, decay=0.9)
    state_layer = StateFieldLayer()
    
    # Add events with overlapping content
    event1_features = exposure_layer.compute_event_features("Paris is the capital of France", "text")
    event1_id = exposure_layer.store_event(
        modality="text",
        features=event1_features,
        preview="Paris is the capital of France",
    )
    
    event2_features = exposure_layer.compute_event_features("France is in Europe", "text")
    event2_id = exposure_layer.store_event(
        modality="text",
        features=event2_features,
        preview="France is in Europe",
    )
    
    event3_features = exposure_layer.compute_event_features("Tokyo is the capital of Japan", "text")
    event3_id = exposure_layer.store_event(
        modality="text",
        features=event3_features,
        preview="Tokyo is the capital of Japan",
    )
    
    # Create input context that activates these events
    context = InputContext(
        text="What is the capital of France?",
        modality="text",
        metadata={},
        goals=["answer_question"],
        constraints=[],
    )
    
    # Activate field
    field = state_layer.activate_from_input(context, exposure_layer)
    
    print(f"Activated events: {list(field.activated_events.keys())}")
    print(f"Activated concepts: {list(field.activated_concepts.keys())}")
    
    # Debug: print event features
    event1 = exposure_layer.get_event(event1_id)
    event2 = exposure_layer.get_event(event2_id)
    event3 = exposure_layer.get_event(event3_id)
    print(f"\nEvent 1 features: {event1.features if event1 else 'Not found'}")
    print(f"Event 2 features: {event2.features if event2 else 'Not found'}")
    print(f"Event 3 features: {event3.features if event3 else 'Not found'}")
    
    # Test resonance between events with similar content (Paris/France)
    if event1_id in field.activated_events and event2_id in field.activated_events:
        print(f"\n=== Resonance between event1 and event2 ===")
        resonance_12 = state_layer.compute_resonance(event1_id, event2_id, exposure_layer, debug=True)
        print(f"Resonance between event1 (Paris/France) and event2 (France/Europe): {resonance_12:.4f}")
        print("  Expected: Positive (shared 'France' feature)")
    
    # Test resonance between events with dissimilar content (Paris vs Tokyo)
    if event1_id in field.activated_events and event3_id in field.activated_events:
        resonance_13 = state_layer.compute_resonance(event1_id, event3_id, exposure_layer)
        print(f"\nResonance between event1 (Paris/France) and event3 (Tokyo/Japan): {resonance_13:.4f}")
        print("  Expected: Low (different topics)")
    
    # Test self-resonance
    if event1_id in field.activated_events:
        resonance_11 = state_layer.compute_resonance(event1_id, event1_id, exposure_layer)
        print(f"\nSelf-resonance for event1: {resonance_11:.4f}")
        print("  Expected: 1.0 (always supports itself)")
    
    # Test concept resonance
    if len(field.activated_concepts) >= 2:
        concepts = list(field.activated_concepts.keys())
        resonance_concepts = state_layer.compute_resonance(concepts[0], concepts[1], exposure_layer)
        print(f"\nResonance between concepts {concepts[0]} and {concepts[1]}: {resonance_concepts:.4f}")

if __name__ == "__main__":
    test_resonance_with_real_data()
