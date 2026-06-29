"""
Full integration test: train + generate across all domains.
"""

from puhl_luck.brain_memory import BrainMemory

brain = BrainMemory()

print("=" * 60)
print("FULL INTEGRATION TEST")
print("=" * 60)

# Train diverse examples
train_data = [
    # Code
    ("def add(a, b):", "return a + b", "code"),
    ("def multiply(x, y):", "return x * y", "code"),
    ("def square(n):", "return n ** 2", "code"),
    
    # Classification
    ("classify news: 정치 관련", "정치", "classification"),
    ("classify news: 경제 뉴스", "경제", "classification"),
    ("classify news: IT 기사", "IT과학", "classification"),
    
    # QA
    ("질문: 한국 수도는?", "서울", "qa"),
    ("질문: 1+1은?", "2", "qa"),
    
    # Labels
    ("sentiment: 너무 좋아요", "positive", "classification"),
    ("sentiment: 최악이네", "negative", "classification"),
]

print(f"\nTraining {len(train_data)} examples...")
for inp, tgt, domain in train_data:
    brain.expose_pair(inp, tgt, domain=domain)
print("✓ Training complete")

# Check storage
stats = brain._surface_storage.get_stats()
print(f"\n Storage: {stats['total_sequences']} sequences, {stats['unique_inputs']} inputs")

# Test exact matches
print("\n" + "=" * 60)
print("EXACT MATCH TESTS")
print("=" * 60)

test_cases = [
    ("def add(a, b):", "return a + b"),
    ("classify news: IT 기사", "IT과학"),
    ("질문: 한국 수도는?", "서울"),
    ("sentiment: 너무 좋아요", "positive"),
]

passed = 0
for inp, expected in test_cases:
    result = brain.generate(inp)
    match = result == expected
    status = "✓" if match else "✗"
    passed += match
    print(f"{status} '{inp[:30]}...'")
    if not match:
        print(f"  Expected: '{expected}'")
        print(f"  Got:      '{result}'")

print(f"\nPassed: {passed}/{len(test_cases)} ({passed/len(test_cases)*100:.0f}%)")

# Test token fallback
print("\n" + "=" * 60)
print("TOKEN FALLBACK TESTS")
print("=" * 60)

fallback_cases = [
    "def subtract(a, b):",  # Similar to trained
    "classify news: 스포츠",  # Different label
]

for inp in fallback_cases:
    result = brain.generate(inp)
    has_output = len(result) > 0
    status = "✓" if has_output else "✗"
    print(f"{status} '{inp}' → '{result[:40]}...'")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
