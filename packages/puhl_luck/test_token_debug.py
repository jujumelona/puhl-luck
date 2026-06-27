"""
Debug token generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "puhl_luck"))

from brain_memory import BrainMemory
from _memory_tokenization import tokenize_code

# Initialize
brain = BrainMemory()

# Train on ONE simple example
partial = "def add(a, b):"
complete = "def add(a, b):\n    return a + b"

print("Training...")
brain.expose_pair(partial, complete, domain="code", modality="code")

# Check what tokens were stored
print("\nTokens from partial:")
partial_tokens = tokenize_code(partial)
print(f"  {partial_tokens}")

print("\nTokens from complete:")
complete_tokens = tokenize_code(complete)
print(f"  {complete_tokens}")

# Check token transition storage
stats = brain._transition_layer.get_token_transition_stats()
print(f"\nToken stats: {stats}")

# Try to find next token manually
print("\nLooking for next token after 'def add(a, b):'...")
candidates = brain._transition_layer.find_next_token(partial_tokens, top_k=5)
print(f"Candidates: {candidates}")

# Try generation
print("\nAttempting generation...")
try:
    result = brain.generate_tokens(partial, max_tokens=10, domain="code")
    print(f"Result: '{result}'")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
