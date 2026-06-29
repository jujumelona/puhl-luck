"""
Test surface mapping counts after training.
"""

from puhl_luck.brain_memory import BrainMemory

brain = BrainMemory()

# Train 5 pairs
pairs = [
    ("def add(a, b):", "return a + b"),
    ("def sub(a, b):", "return a - b"),
    ("classify: IT news", "IT과학"),
    ("classify: politics", "정치"),
    ("질문: 수도는?", "서울"),
]

print("Training 5 pairs...")
for inp, tgt in pairs:
    brain.expose_pair(inp, tgt)

# Check storage
print("\n=== Storage Stats ===")
print(f"surface_storage: {brain._surface_storage.get_stats()}")
print(f"surface_layer.state_to_surface: {len(brain._surface_layer.state_to_surface)}")
print(f"surface_layer.surface_atoms: {sum(len(v) for v in brain._surface_layer.surface_atoms.values())}")

# Test retrieval
print("\n=== Test Retrieval ===")
for inp, expected in pairs[:3]:
    result = brain.generate(inp)
    match = "✓" if result == expected else "✗"
    print(f"{match} '{inp[:20]}...' → '{result}' (expected: '{expected}')")
