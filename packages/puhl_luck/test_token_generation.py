"""
Quick test for token-level generation.
"""

from puhl_luck.brain_memory import BrainMemory

# Initialize
brain = BrainMemory()

# Train on simple code examples
print("Training on code examples...")

examples = [
    ("def add(a, b):", "def add(a, b):\n    return a + b"),
    ("def multiply(x, y):", "def multiply(x, y):\n    return x * y"),
    ("def square(n):", "def square(n):\n    return n ** 2"),
    ("def is_even(n):", "def is_even(n):\n    return n % 2 == 0"),
    ("def double(x):", "def double(x):\n    return x * 2"),
]

for partial, complete in examples:
    brain.expose_pair(partial, complete, domain="code", modality="code")
    print(f"  Trained: {partial[:30]}...")

# Check token transition stats
if hasattr(brain._transition_layer, "get_token_transition_stats"):
    stats = brain._transition_layer.get_token_transition_stats()
    print(f"\nToken transition stats:")
    print(f"  Contexts: {stats['total_contexts']}")
    print(f"  Transitions: {stats['total_transitions']}")
    print(f"  Avg fanout: {stats['avg_fanout']}")

# Test generation
print("\n" + "=" * 60)
print("Testing token generation...")
print("=" * 60)

test_prompts = [
    "def subtract(a, b):",
    "def divide(x, y):",
    "def is_odd(n):",
]

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    try:
        result = brain.generate_code(prompt, max_tokens=32, validate_syntax=False)
        print(f"Generated: {result[:100]}")
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("Test complete!")
