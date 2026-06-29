"""
P53 Evidence-Based Sparse LM Test
"""

from puhl_luck.brain_memory import BrainMemory

print("="*60)
print("P53 Evidence-Based Sparse LM Test")
print("="*60)

brain = BrainMemory()

# Training data
training = [
    ("def add(a, b):", "return a + b"),
    ("def subtract(a, b):", "return a - b"),
    ("def multiply(a, b):", "return a * b"),
    ("def divide(a, b):", "return a / b"),
    ("def square(x):", "return x * x"),
    ("def double(x):", "return x + x"),
    ("def negate(x):", "return -x"),
    ("def is_even(n):", "return n % 2 == 0"),
    ("def is_odd(n):", "return n % 2 == 1"),
    ("def absolute(x):", "return abs(x)"),
]

print(f"\n[Training] {len(training)} pairs...")
for prompt, target in training:
    brain.expose_pair(prompt, target, domain="code")

stats = brain._logit_generator.get_statistics()
print(f"\nStats:")
print(f"  Pairs learned: {stats['pairs_learned']}")
print(f"  Vocab size: {stats['vocab_size']}")
print(f"  Transitions: {stats['transition_entries']}")
print(f"  Bigrams: {stats['bigram_entries']}")
print(f"  Trigrams: {stats['trigram_entries']}")
print(f"  First token entries: {stats['first_token_entries']}")

# Test generation
print("\n" + "="*60)
print("[Generation Tests]")
print("="*60)

test_prompts = [
    "def cube(x):",
    "def half(x):",
    "def is_positive(n):",
]

for prompt in test_prompts:
    output = brain.generate(prompt, max_new_tokens=15)
    print(f"\nPrompt:  {prompt}")
    print(f"Output:  {output}")
    tokens = output.split()
    print(f"Tokens:  {len(tokens)}")
    if tokens:
        has_return = "return" in tokens
        max_rep = max(tokens.count(t) for t in set(tokens))
        print(f"  - Has 'return': {has_return}")
        print(f"  - Max repetition: {max_rep}")

print("\n" + "="*60)
print("P53 Test Complete!")
print("="*60)
