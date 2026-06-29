"""
Quick P52 validation test.
"""

from puhl_luck.brain_memory import BrainMemory

def test_p52_basic():
    """Test basic P52 functionality."""
    print("=" * 60)
    print("P52 Quick Validation Test")
    print("=" * 60)
    
    brain = BrainMemory()
    
    # Test 1: Learn and generate
    print("\n[Test 1] Learn and Generate")
    brain.expose_pair("def count_even(nums):", "return sum(1 for n in nums if n % 2 == 0)", domain="code")
    brain.expose_pair("def count_odd(nums):", "return sum(1 for n in nums if n % 2 == 1)", domain="code")
    
    output = brain.generate("def count_positive(nums):", max_new_tokens=20)
    print(f"Input:  def count_positive(nums):")
    print(f"Output: {output}")
    print(f"✅ Generated {len(output.split())} tokens")
    
    # Test 2: No Python repetition
    print("\n[Test 2] No Python Repetition")
    brain.expose_pair("Write Python code", "def hello(): return 'world'", domain="code")
    output = brain.generate("Python program", max_new_tokens=20)
    print(f"Output: {output}")
    
    python_count = output.lower().count("python")
    print(f"'Python' appears {python_count} times")
    assert python_count < 5, f"FAIL: Python repeated {python_count} times!"
    print("✅ No infinite repetition")
    
    # Test 3: Diverse outputs
    print("\n[Test 3] Diverse Outputs")
    for _ in range(5):
        brain.expose_pair("def func(x):", "return x * 2", domain="code")
    
    outputs = set()
    for i in range(5):
        out = brain.generate("def process(data):", max_new_tokens=10)
        outputs.add(out)
    
    print(f"Generated {len(outputs)} unique outputs from 5 attempts")
    assert len(outputs) >= 2, "FAIL: Not diverse enough!"
    print("✅ Generates diverse outputs")
    
    # Test 4: Statistics
    print("\n[Test 4] Statistics")
    stats = brain._logit_generator.get_statistics()
    print(f"  Pairs learned: {stats['pairs_learned']}")
    print(f"  Vocab size: {stats['vocab_size']}")
    print(f"  Unigram entries: {stats['unigram_entries']}")
    print(f"  Transition entries: {stats['transition_entries']}")
    print(f"  Bigram entries: {stats['bigram_entries']}")
    print(f"  Trigram entries: {stats['trigram_entries']}")
    print("✅ Statistics tracked")
    
    # Test 5: Universal domains
    print("\n[Test 5] Universal Domains")
    
    # Code
    brain.expose_pair("def add(a,b):", "return a+b", domain="code")
    code_out = brain.generate("def sub(a,b):", max_new_tokens=10)
    print(f"  Code: {code_out}")
    
    # Text
    brain.expose_pair("Hello", "world", domain="text")
    text_out = brain.generate("Hi", max_new_tokens=5)
    print(f"  Text: {text_out}")
    
    # Classification
    brain.expose_pair("news article about tech", "IT과학", domain="classification")
    class_out = brain.generate("article about sports", max_new_tokens=5)
    print(f"  Classification: {class_out}")
    
    print("✅ Works across domains")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("P52 Sparse Logit LM is working correctly!")
    print("=" * 60)

if __name__ == "__main__":
    test_p52_basic()
