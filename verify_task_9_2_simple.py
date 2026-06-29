"""Simple verification for Task 9.2: Compressed serialization

Demonstrates save/load functionality and compression effectiveness.

Requirements: 14.3, 14.4
"""

import os
import tempfile
from packages.puhl_luck.puhl_luck._logit_generator import SparseLogitGenerator

def main():
    print("Task 9.2: Compressed Serialization - Simple Verification")
    print("Requirements: 14.3, 14.4")
    print("=" * 60)
    
    # Create and train a model
    print("\n1. Creating and training model...")
    gen = SparseLogitGenerator(
        max_tokens=50,
        top_k=3,
        temperature=0.8,
        rare_token_threshold=2,
    )
    
    # Train on a few examples
    training_pairs = [
        ("def add(a, b):", "return a + b"),
        ("def multiply(x, y):", "return x * y"),
        ("def subtract(a, b):", "return a - b"),
    ]
    
    for inp, tgt in training_pairs:
        gen.learn(inp, tgt)
    
    stats_before = gen.get_statistics()
    print(f"   Pairs learned: {stats_before['pairs_learned']}")
    print(f"   Tokens learned: {stats_before['tokens_learned']}")
    print(f"   Vocab size: {stats_before['vocab_size']}")
    
    # Generate before save
    test_input = "def add(x, y):"
    output_before, _ = gen.generate(test_input, max_tokens=5)
    print(f"   Generation: {test_input} -> {output_before}")
    
    # Save model
    print("\n2. Saving model with gzip compression...")
    with tempfile.NamedTemporaryFile(suffix='.pkl.gz', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        save_info = gen.save(tmp_path)
        print(f"   ✓ Saved to {os.path.basename(tmp_path)}")
        print(f"   Uncompressed: {save_info['uncompressed_size_bytes'] / 1024:.2f} KB")
        print(f"   Compressed: {save_info['compressed_size_bytes'] / 1024:.2f} KB")
        print(f"   Compression ratio: {save_info['compression_ratio']:.2f}x")
        
        # Load model
        print("\n3. Loading model from compressed file...")
        gen_loaded = SparseLogitGenerator.load(tmp_path)
        print(f"   ✓ Loaded from {os.path.basename(tmp_path)}")
        
        # Verify loaded model
        stats_after = gen_loaded.get_statistics()
        print(f"   Pairs learned: {stats_after['pairs_learned']}")
        print(f"   Tokens learned: {stats_after['tokens_learned']}")
        print(f"   Vocab size: {stats_after['vocab_size']}")
        
        output_after, _ = gen_loaded.generate(test_input, max_tokens=5)
        print(f"   Generation: {test_input} -> {output_after}")
        
        # Verification
        print("\n4. Verification Results:")
        print("   " + "-" * 56)
        
        stats_match = (
            stats_before['pairs_learned'] == stats_after['pairs_learned'] and
            stats_before['tokens_learned'] == stats_after['tokens_learned'] and
            stats_before['vocab_size'] == stats_after['vocab_size']
        )
        
        generation_match = (output_before == output_after)
        compression_effective = (save_info['compression_ratio'] >= 2.0)
        
        print(f"   Statistics preserved: {'✓ PASS' if stats_match else '✗ FAIL'}")
        print(f"   Generation preserved: {'✓ PASS' if generation_match else '✗ FAIL'}")
        print(f"   Compression effective (>2x): {'✓ PASS' if compression_effective else '✗ FAIL'}")
        
        # Memory footprint extrapolation
        print("\n5. Memory Footprint Extrapolation:")
        print("   " + "-" * 56)
        size_per_pair_kb = save_info['compressed_size_bytes'] / (1024 * stats_before['pairs_learned'])
        extrapolated_10k_mb = (size_per_pair_kb * 10000) / 1024
        
        print(f"   Size per pair: {size_per_pair_kb:.2f} KB")
        print(f"   Extrapolated 10K pairs: {extrapolated_10k_mb:.2f} MB")
        print(f"   Requirement (<500MB): {'✓ PASS' if extrapolated_10k_mb < 500 else '✗ FAIL'}")
        
        # Overall result
        all_passed = stats_match and generation_match and compression_effective and (extrapolated_10k_mb < 500)
        
        print("\n" + "=" * 60)
        if all_passed:
            print("✓ TASK 9.2 VERIFICATION PASSED")
            print("\nImplemented features:")
            print("  • save() method with gzip compression")
            print("  • load() class method for deserialization")
            print("  • pickle serialization with compression")
            print("  • Memory footprint <500MB verified (extrapolated)")
        else:
            print("✗ TASK 9.2 VERIFICATION FAILED")
        print("=" * 60)
        
        return all_passed
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
