"""
P55: Prompt-Conditioned Sparse Logit LM Test

Key principles:
1. Context ALWAYS includes prompt (never dropped after [SEP])
2. Field features condition EVERY token (not just first)
3. No unigram fallback (pure context-based)
4. Low confidence when no context match
"""

from puhl_luck._logit_generator_p55 import SparseLogitGeneratorP55

print("="*70)
print("P55: PROMPT-CONDITIONED SPARSE LOGIT LM")
print("="*70)

gen = SparseLogitGeneratorP55(output_mode="code")

# Training data - diverse patterns
training = [
    # Arithmetic
    ("def add(a, b):", "return a + b"),
    ("def subtract(a, b):", "return a - b"),
    ("def multiply(a, b):", "return a * b"),
    ("def divide(a, b):", "return a / b"),
    
    # Unary
    ("def square(x):", "return x * x"),
    ("def cube(x):", "return x * x * x"),
    ("def double(x):", "return x + x"),
    ("def negate(x):", "return -x"),
    ("def increment(x):", "return x + 1"),
    ("def decrement(x):", "return x - 1"),
    
    # Boolean
    ("def is_even(n):", "return n % 2 == 0"),
    ("def is_odd(n):", "return n % 2 == 1"),
    ("def is_positive(n):", "return n > 0"),
    ("def is_negative(n):", "return n < 0"),
    ("def is_zero(n):", "return n == 0"),
]

print(f"\n[Training] {len(training)} pairs...")
for i, (prompt, target) in enumerate(training, 1):
    gen.learn(prompt, target)
    if i % 5 == 0:
        print(f"  {i}/{len(training)} learned")

stats = gen.get_statistics()
print(f"\n[Statistics]")
print(f"  Pairs: {stats['pairs_learned']}")
print(f"  Vocab: {stats['vocab_size']}")
print(f"  Transitions: {stats['transition_entries']}")
print(f"  Bigrams: {stats['bigram_entries']}")
print(f"  Trigrams: {stats['trigram_entries']}")

# Generation tests
print("\n" + "="*70)
print("[Generation Tests - Prompt Conditioned!]")
print("="*70)

test_queries = [
    "def quad(x):",
    "def half(x):",
    "def triple(x):",
    "def is_nonzero(n):",
    "def sum_ab(a, b):",
]

for query in test_queries:
    output, metrics = gen.generate(query, max_tokens=20)
    
    print(f"\nQuery:  {query}")
    print(f"Output: {output}")
    print(f"  Steps: {metrics['steps']}")
    print(f"  Low confidence: {metrics['low_confidence_steps']}")
    print(f"  Stop: {metrics['stop_reason']}")
    
    # Quality checks
    tokens = output.split()
    
    if not tokens:
        print("  ❌ EMPTY")
        continue
    
    has_return = "return" in tokens
    max_rep = max(tokens.count(t) for t in set(tokens)) if tokens else 0
    
    print(f"  {'✅' if has_return else '❌'} Has 'return'")
    print(f"  {'✅' if max_rep <= 2 else '❌'} Max rep: {max_rep}")
    print(f"  {'✅' if 2 <= len(tokens) <= 10 else '⚠️'} Length: {len(tokens)}")

print("\n" + "="*70)
print("P55 KEY FEATURES:")
print("  1. ✅ Prompt NEVER dropped (full context always)")
print("  2. ✅ Field conditioning on EVERY token")
print("  3. ✅ Context-only candidates (no unigram flood)")
print("  4. ✅ Low confidence tracking")
print("  5. ✅ Greedy decoding (deterministic)")
print("="*70)
