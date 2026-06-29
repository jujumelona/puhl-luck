"""
P56: Joint Context-Prompt Filtering Test

THE KEY FIX:
Candidate must satisfy BOTH:
1. context_evidence > 0 (from ngrams)
2. prompt_field_evidence > 0 (from field_token)

This BLOCKS "from" collapse:
- "from" may have global transition evidence
- But "from" has NO connection to "def add(a, b):" features
- So "from" is FILTERED OUT!
"""

from puhl_luck._logit_generator_p56 import SparseLogitGeneratorP56

print("="*70)
print("P56: JOINT CONTEXT-PROMPT FILTERING")
print("="*70)

gen = SparseLogitGeneratorP56(output_mode="code")

# Training data
training = [
    # Arithmetic
    ("def add(a, b):", "return a + b"),
    ("def subtract(a, b):", "return a - b"),
    ("def multiply(a, b):", "return a * b"),
    ("def divide(a, b):", "return a / b"),
    ("def modulo(a, b):", "return a % b"),
    
    # Unary
    ("def square(x):", "return x * x"),
    ("def cube(x):", "return x * x * x"),
    ("def double(x):", "return x + x"),
    ("def triple(x):", "return x + x + x"),
    ("def negate(x):", "return -x"),
    ("def increment(x):", "return x + 1"),
    ("def decrement(x):", "return x - 1"),
    
    # Boolean
    ("def is_even(n):", "return n % 2 == 0"),
    ("def is_odd(n):", "return n % 2 == 1"),
    ("def is_positive(n):", "return n > 0"),
    ("def is_negative(n):", "return n < 0"),
    ("def is_zero(n):", "return n == 0"),
    ("def is_nonzero(n):", "return n != 0"),
]

print(f"\n[Training] {len(training)} pairs...")
for i, (prompt, target) in enumerate(training, 1):
    # Extract field features from prompt
    # Simple feature extraction: just use tokens as features
    tokens = prompt.lower().split()
    field_features = [f"text:tok:{tok}" for tok in tokens if tok.isalnum()]
    
    gen.learn(prompt, target, field_features=field_features)
    if i % 5 == 0:
        print(f"  {i}/{len(training)} learned")

stats = gen.get_statistics()
print(f"\n[Statistics]")
print(f"  Pairs: {stats['pairs_learned']}")
print(f"  Vocab: {stats['vocab_size']}")
print(f"  Transitions: {stats['transition_entries']}")
print(f"  Field-token: {stats['field_token_entries']}")

# Generation tests
print("\n" + "="*70)
print("[Generation Tests - Joint Filtering!]")
print("="*70)

test_queries = [
    "def quad(x):",
    "def half(x):",
    "def fifth(x):",
    "def is_greater(n):",
    "def sum_two(a, b):",
    "def difference(a, b):",
]

all_valid = True

for query in test_queries:
    # Extract field features from query
    tokens = query.lower().split()
    field_features = [f"text:tok:{tok}" for tok in tokens if tok.isalnum()]
    
    output, metrics = gen.generate(query, field_features=field_features, max_tokens=20)
    
    print(f"\nQuery:  {query}")
    print(f"Output: {output}")
    
    tokens = output.split()
    
    # Critical checks
    has_return = "return" in tokens
    starts_with_from = tokens[0] == "from" if tokens else False
    has_def = "def" in tokens
    
    if starts_with_from:
        print(f"  ❌ STARTS WITH 'from' - JOINT FILTERING FAILED!")
        all_valid = False
    elif has_def and tokens.index("def") > 0:
        print(f"  ❌ HAS 'def' in output - WRONG!")
        all_valid = False
    elif not has_return:
        print(f"  ❌ NO 'return' - INVALID CODE!")
        all_valid = False
    else:
        print(f"  ✅ Valid code structure")
    
    print(f"  Steps: {metrics['steps']}")
    print(f"  Stop: {metrics['stop_reason']}")
    print(f"  Length: {len(tokens)} tokens")

print("\n" + "="*70)
if all_valid:
    print("✅ ALL TESTS PASSED - NO 'from' COLLAPSE!")
else:
    print("❌ SOME TESTS FAILED - JOINT FILTERING NEEDS WORK")

print("\nP56 KEY MECHANISM:")
print("  Candidate qualification:")
print("    ✅ context_evidence(token) > 0  AND")
print("    ✅ prompt_field_evidence(token) > 0")
print("  ")
print("  Blocks 'from':")
print("    'from' may have global transition evidence")
print("    BUT 'from' has NO field evidence for 'def add(a, b):'")
print("    SO 'from' is FILTERED OUT!")
print("="*70)
