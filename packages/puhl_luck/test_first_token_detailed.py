"""
Detailed First Token Table Diagnosis
"""

from puhl_luck.brain_memory import BrainMemory

print("="*70)
print("DETAILED FIRST TOKEN TABLE DIAGNOSIS")
print("="*70)

brain = BrainMemory()

# Training examples
training = [
    ("def add(a, b):", "return a + b"),
    ("def subtract(a, b):", "return a - b"),
    ("def multiply(a, b):", "return a * b"),
]

print("\n[TRAINING PHASE]")
print("-"*70)

for i, (prompt, target) in enumerate(training, 1):
    print(f"\n{i}. Learning: '{prompt}' -> '{target}'")
    brain.expose_pair(prompt, target, domain="code")

# Inspect first_token table
print("\n" + "="*70)
print("[FIRST_TOKEN TABLE CONTENTS]")
print("="*70)

tables = brain._logit_generator.tables
if tables.first_token:
    for sketch, token_counts in sorted(tables.first_token.items()):
        print(f"\nSketch: {sketch}")
        print(f"  Marginal count: {tables.first_token_marginal[sketch]}")
        for token, count in sorted(token_counts.items(), key=lambda x: -x[1]):
            print(f"    '{token}': {count}")
else:
    print("  (empty)")

# Test queries
print("\n" + "="*70)
print("[TEST QUERIES]")
print("="*70)

test_cases = [
    "def add(a, b):",      # Seen - should match exactly
    "def multiply(a, b):", # Seen - should match exactly  
    "def cube(x):",        # Unseen - should match via generalization
    "def half(y):",        # Unseen - should match via generalization
]

for query in test_cases:
    print(f"\nQuery: '{query}'")
    
    # Extract features for this query
    p_features, _, _ = brain.extract_text(query)
    print(f"  Raw features (first 10): {p_features[:10]}")
    
    # Get candidates
    candidates = tables.get_first_token_candidates(
        field_features=p_features,
        top_k=5
    )
    
    print(f"  Candidates:")
    for token, evidence in candidates[:5]:
        source = "FIRST_TOKEN" if evidence >= 1.0 else "UNIGRAM"
        print(f"    '{token}': {evidence:.3f} [{source}]")

# Run full generation test
print("\n" + "="*70)
print("[GENERATION TEST]")
print("="*70)

query = "def cube(x):"
print(f"\nQuery: {query}")
output = brain.generate(query, max_new_tokens=10)
print(f"Output: {output}")
tokens = output.split()
print(f"Tokens: {tokens}")
print(f"First token: '{tokens[0]}' if tokens else '(empty)'")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
