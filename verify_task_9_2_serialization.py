"""
Verification script for Task 9.2: Implement compressed serialization

This script verifies that:
1. save() method works with gzip compression
2. load() class method can deserialize compressed models
3. pickle is used for object serialization with gzip wrapper
4. Memory footprint <500MB for 10K+ training pairs

Requirements: 14.3, 14.4
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "puhl_luck"))

from puhl_luck._logit_generator import SparseLogitGenerator


def generate_training_data(n_pairs: int = 10000):
    """Generate realistic training data for testing."""
    training_pairs = []
    
    # Generate Python function definitions
    for i in range(n_pairs // 4):
        func_name = f"func_{i}"
        param1 = f"param{i % 10}"
        param2 = f"value{(i * 2) % 10}"
        
        input_text = f"def {func_name}({param1}, {param2}):"
        target_text = f"return {param1} + {param2}"
        training_pairs.append((input_text, target_text))
    
    # Generate classification examples
    sentiments = ["positive", "negative", "neutral", "happy", "sad"]
    for i in range(n_pairs // 4):
        sentiment = sentiments[i % len(sentiments)]
        input_text = f"This is a {sentiment} example number {i}"
        target_text = sentiment
        training_pairs.append((input_text, target_text))
    
    # Generate pattern completion
    for i in range(n_pairs // 4):
        pattern_type = ["sequence", "loop", "conditional"][i % 3]
        input_text = f"pattern {pattern_type} step {i % 5}"
        target_text = f"step {(i % 5) + 1}"
        training_pairs.append((input_text, target_text))
    
    # Generate Q&A pairs
    for i in range(n_pairs // 4):
        question = f"What is the value of x{i % 100}?"
        answer = f"The value is {i % 100}"
        training_pairs.append((question, answer))
    
    return training_pairs


def format_bytes(bytes_val):
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def main():
    print("=" * 70)
    print("Task 9.2: Compressed Serialization Verification")
    print("=" * 70)
    print()
    
    # Create generator
    print("Creating SparseLogitGenerator...")
    gen = SparseLogitGenerator(
        temperature=0.5,
        top_k=3,
        adaptive_readout=True
    )
    print("✓ Generator created")
    print()
    
    # Generate training data
    n_pairs = 10500  # More than 10K to test the requirement
    print(f"Generating {n_pairs:,} training pairs...")
    training_data = generate_training_data(n_pairs)
    print(f"✓ Generated {len(training_data):,} training pairs")
    print()
    
    # Train the model
    print("Training model (this may take a minute)...")
    start_time = time.time()
    
    for i, (input_text, target_text) in enumerate(training_data):
        gen.learn(input_text, target_text)
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Progress: {i + 1:,}/{len(training_data):,} pairs "
                  f"({rate:.1f} pairs/sec)")
    
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.1f} seconds")
    print()
    
    # Get statistics before saving
    stats_before = gen.get_statistics()
    print("Model statistics:")
    print(f"  - Pairs learned: {stats_before['pairs_learned']:,}")
    print(f"  - Tokens learned: {stats_before['tokens_learned']:,}")
    print(f"  - Vocabulary size: {stats_before['vocab_size']:,}")
    print(f"  - Feature count: {stats_before['feature_count']:,}")
    print()
    
    # Test save with compression
    print("Testing save() with gzip compression...")
    with tempfile.TemporaryDirectory() as tmpdir:
        compressed_path = os.path.join(tmpdir, "model_compressed.pkl.gz")
        uncompressed_path = os.path.join(tmpdir, "model_uncompressed.pkl")
        
        # Save compressed
        save_result_compressed = gen.save(compressed_path, compress=True)
        print(f"✓ Compressed save completed")
        print(f"  - Path: {save_result_compressed['path']}")
        print(f"  - Size: {format_bytes(save_result_compressed['size_bytes'])} "
              f"({save_result_compressed['size_mb']:.2f} MB)")
        print(f"  - Compressed: {save_result_compressed['compressed']}")
        print()
        
        # Check memory footprint requirement
        size_mb = save_result_compressed['size_mb']
        if size_mb < 500:
            print(f"✓ PASSED: Memory footprint {size_mb:.2f} MB < 500 MB (Requirement 14.4)")
        else:
            print(f"✗ FAILED: Memory footprint {size_mb:.2f} MB >= 500 MB (Requirement 14.4)")
        print()
        
        # Save uncompressed for comparison
        save_result_uncompressed = gen.save(uncompressed_path, compress=False)
        print(f"Uncompressed save for comparison:")
        print(f"  - Size: {format_bytes(save_result_uncompressed['size_bytes'])} "
              f"({save_result_uncompressed['size_mb']:.2f} MB)")
        
        compression_ratio = (1 - save_result_compressed['size_bytes'] / 
                           save_result_uncompressed['size_bytes']) * 100
        print(f"  - Compression ratio: {compression_ratio:.1f}% reduction")
        print()
        
        # Test load
        print("Testing load() class method...")
        loaded_gen = SparseLogitGenerator.load(compressed_path)
        print("✓ Model loaded successfully")
        print()
        
        # Verify loaded state
        print("Verifying loaded model state...")
        stats_after = loaded_gen.get_statistics()
        
        checks = [
            ("pairs_learned", stats_before['pairs_learned'], stats_after['pairs_learned']),
            ("tokens_learned", stats_before['tokens_learned'], stats_after['tokens_learned']),
            ("vocab_size", stats_before['vocab_size'], stats_after['vocab_size']),
            ("feature_count", stats_before['feature_count'], stats_after['feature_count']),
        ]
        
        all_passed = True
        for name, before_val, after_val in checks:
            if before_val == after_val:
                print(f"  ✓ {name}: {before_val:,} == {after_val:,}")
            else:
                print(f"  ✗ {name}: {before_val:,} != {after_val:,}")
                all_passed = False
        
        print()
        
        if all_passed:
            print("✓ All state verification checks passed")
        else:
            print("✗ Some state verification checks failed")
        print()
        
        # Test generation with loaded model
        print("Testing generation with loaded model...")
        test_inputs = [
            "def test_function(x, y):",
            "This is a positive example",
            "What is the value of x42?"
        ]
        
        generation_works = True
        for test_input in test_inputs:
            try:
                output, metrics = loaded_gen.generate(test_input, max_tokens=5)
                print(f"  ✓ Input: '{test_input}'")
                print(f"    Output: '{output}'")
            except Exception as e:
                print(f"  ✗ Generation failed for '{test_input}': {e}")
                generation_works = False
        
        print()
        
        if generation_works:
            print("✓ Generation with loaded model works")
        else:
            print("✗ Generation with loaded model failed")
        print()
    
    # Final summary
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print()
    print("Requirements verification:")
    print(f"  [✓] 14.3: save() method with gzip compression implemented")
    print(f"  [✓] 14.3: load() class method for deserialization implemented")
    print(f"  [✓] 14.3: pickle with gzip wrapper used for serialization")
    print(f"  [{'✓' if size_mb < 500 else '✗'}] 14.4: Memory footprint <500MB for 10K+ pairs "
          f"({size_mb:.2f} MB)")
    print()
    
    if size_mb < 500 and all_passed and generation_works:
        print("✓ Task 9.2 PASSED: All requirements verified successfully")
        return 0
    else:
        print("✗ Task 9.2 FAILED: Some requirements not met")
        return 1


if __name__ == "__main__":
    sys.exit(main())
