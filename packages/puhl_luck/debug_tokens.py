"""Direct token transition debugging."""

import sys
sys.path.insert(0, "puhl_luck")

from brain_memory import BrainMemory
from _memory_tokenization import tokenize_code

brain = BrainMemory()

# Train ONE example
partial = "def add(a, b):"
complete = "def add(a, b):\n    return a + b"

print(f"Partial: {partial}")
print(f"Complete: {complete}\n")

# Tokenize
partial_toks = tokenize_code(partial)
complete_toks = tokenize_code(complete)
answer_toks = complete_toks[len(partial_toks):]

print(f"Partial tokens: {partial_toks}")
print(f"Answer tokens: {answer_toks}\n")

# Store manually
print("Storing token transitions manually...")
for i in range(len(answer_toks)):
    context = partial_toks + answer_toks[:i]
    next_tok = answer_toks[i]
    print(f"  Context {context[-3:]} → {next_tok}")
    brain._transition_layer.store_token_transition(context, next_tok, "code", "code")

# Check stats
stats = brain._transition_layer.get_token_transition_stats()
print(f"\nStats: {stats}")

# Query
print(f"\nQuerying with partial tokens: {partial_toks}")
candidates = brain._transition_layer.find_next_token(partial_toks, top_k=5)
print(f"Candidates: {candidates}")

# Try generation
print(f"\nGenerating from: {partial}")
result = brain.generate_tokens(partial, max_tokens=10, domain="code")
print(f"Result: '{result}'")
