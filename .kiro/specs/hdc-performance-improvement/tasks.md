# Implementation Plan: HDC Performance Improvement

## Overview

This implementation plan addresses the critical performance issues in the HDC (Hyperdimensional Computing) system. The system currently achieves 33.3% accuracy with 178ms inference time. The goal is to improve accuracy to >85% and reduce inference time to <50ms through systematic optimization of the sparse autoregressive generation pipeline.

The plan follows a phased approach:
1. **Establish baseline measurement infrastructure** - benchmarking and diagnostics
2. **Implement core accuracy improvements** - credit assignment, backoff strategy, adaptive readout
3. **Add speed optimizations** - Rust acceleration, sparse table caching, incremental features
4. **Optimize hyperparameters** - grid search and tuning infrastructure
5. **Validate and integrate** - end-to-end testing and performance validation

## Tasks

- [x] 1. Create benchmarking and diagnostics infrastructure
  - [x] 1.1 Implement BenchmarkSuite class for performance testing
    - Create `packages/puhl_luck/puhl_luck/benchmarks/__init__.py` with BenchmarkSuite class
    - Implement `run_all_benchmarks()` method covering code generation, classification, pattern matching, and Q&A tasks
    - Implement `save_results()` method to save benchmark results to JSON with timestamps
    - Add methods to track backoff statistics, copy gate activations, and empty output rates
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

  - [x]* 1.2 Write unit tests for BenchmarkSuite
    - Test benchmark execution for all task types
    - Test JSON result saving and loading
    - Test metric calculation correctness
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 1.3 Implement generation statistics and diagnostics
    - Add `GenerationMetrics` dataclass to `_logit_generator.py`
    - Modify `generate()` method to return detailed metrics when `return_metrics=True`
    - Add `get_statistics()` method to return pairs_learned, total_transitions, total_contexts, total_unique_tokens
    - Implement logging for excessive backoff (>20% field_only usage)
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [x]* 1.4 Write unit tests for generation statistics
    - Test GenerationMetrics dataclass creation
    - Test get_statistics() return values
    - Test metric tracking during generation
    - _Requirements: 13.1, 13.2, 13.3_

- [x] 2. Checkpoint - Verify baseline measurement capability
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Implement core accuracy improvements
  - [x] 3.1 Enhance credit assignment with rank-loss
    - Modify `SparseEvidenceTables.credit_assign()` in `_logit_tables.py`
    - Implement ranking of all tokens by score
    - Add positive evidence reinforcement for target token with logarithmic amount
    - Add negative evidence for wrong-above tokens (tokens ranked higher than target)
    - Return diagnostic dict with positive_amount, negative_tokens, wrong_above_count, token_loss
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 3.4_

  - [x]* 3.2 Write property test for credit assignment
    - **Property 1: Forgetting Prevention**
    - **Validates: Requirements 3.1, 3.3**
    - Test that accuracy on training set A does not decrease by >5% after learning set B
    - Use Hypothesis to generate random training sequences
    - _Requirements: 3.1, 3.3_

  - [x] 3.3 Implement improved backoff strategy with HDC similarity
    - Create `lookup_with_backoff()` method in `SparseEvidenceTables`
    - Implement progressive backoff sequence: K → K/2 → K/4 → HDC similarity → unigram → field_only
    - Add HDC similarity search using `hv_similarity()` for approximate matching
    - Return token distribution and backoff level used
    - _Requirements: 4.1, 4.2, 4.3, 10.1, 10.2, 10.3, 10.4_

  - [x]* 3.4 Write property test for backoff strategy
    - **Property 3: Backoff Sequence Correctness**
    - **Validates: Requirements 10.1, 10.2**
    - Test that backoff follows correct sequence and terminates at first match
    - Verify that >80% of tokens resolve within 3 backoff levels
    - _Requirements: 10.1, 10.2, 10.4_

  - [x] 3.5 Implement dynamic adaptive readout configuration
    - Create `dynamic_readout_config()` function in `_logit_tables.py`
    - Calculate hidden_dim as sqrt(vocab_size * feature_count), clamped to [64, 2048]
    - Calculate vocab_cap as sqrt(event_count) * log(vocab_size), clamped appropriately
    - Implement adaptive learning rate decay: 0.01 / log2(event_count + 2)
    - Integrate with SparseEvidenceTables initialization
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 4.1_

  - [x]* 3.6 Write unit tests for adaptive readout
    - Test dynamic_readout_config() calculation correctness
    - Test hidden dimension scaling with data size
    - Test vocab cap scaling with event count
    - Test learning rate decay
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 4. Checkpoint - Verify accuracy improvements
  - Ensure all tests pass, run benchmarks to measure accuracy improvement, ask the user if questions arise.

- [x] 5. Implement speed optimizations
  - [x] 5.1 Add Rust acceleration detection and fallback
    - Verify Rust module loading in `_brain_hdc.py` (already implemented)
    - Ensure graceful fallback to Python when Rust unavailable
    - Add runtime detection flag `RUST_AVAILABLE`
    - Log Rust availability status on module load
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x]* 5.2 Write property test for Rust-Python equivalence
    - **Property 4: Rust-Python Equivalence**
    - **Validates: Requirements 8.5**
    - Test that feature_hv() produces same results in Rust and Python
    - Test that hv_similarity() produces same results in Rust and Python
    - Test that bundle_hv() and rotate_hv() are equivalent
    - _Requirements: 8.5_

  - [x] 5.3 Implement sparse table lookup caching
    - Create `SparseTableWithCache` class wrapper in `_logit_tables.py`
    - Implement LRU cache for context sketch computation using OrderedDict
    - Add `_context_sketch()` method with cache lookup and eviction
    - Set default cache size to 1000 entries, make configurable
    - Track cache hits/misses for diagnostics
    - _Requirements: 2.1, 2.2, 9.1, 9.2, 9.3, 9.4_

  - [x]* 5.4 Write unit tests for sparse table caching
    - Test cache hit/miss behavior
    - Test LRU eviction when cache full
    - Test performance improvement with caching
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 5.5 Implement incremental feature extraction
    - Create `IncrementalFeatureExtractor` class in `_logit_generator.py`
    - Use deque with maxlen for sliding context window
    - Implement `append_token()` method to update only new features (unigram, bigram, trigram, position features)
    - Cache unchanged features between generation steps
    - Integrate with `_active_features()` method
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x]* 5.6 Write unit tests for incremental feature extraction
    - Test deque sliding window behavior
    - Test feature updates when appending tokens
    - Test feature cache correctness
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 5.7 Optimize tokenization with punctuation preservation validation
    - Review `_tokenize()` method in `_logit_generator.py`
    - Ensure punctuation-preserving split using `_TOKEN_RE` regex
    - Validate special token handling ([BOS], [SEP], [EOS])
    - Implement `_detokenize()` method for round-trip validation
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

  - [x]* 5.8 Write property test for tokenization round-trip
    - **Property 5: Tokenization Round-Trip**
    - **Validates: Requirements 15.3**
    - Test that detokenize(tokenize(text)) preserves content
    - Use Hypothesis to generate random code and text inputs
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

- [x] 6. Checkpoint - Verify speed optimizations
  - Ensure all tests pass, run benchmarks to measure speed improvement, ask the user if questions arise.

- [x] 7. Implement hyperparameter optimization
  - [x] 7.1 Create HyperparameterTuner class
    - Create `packages/puhl_luck/puhl_luck/benchmarks/hyperparameter_tuner.py`
    - Implement `__init__()` to accept train/test datasets
    - Add configuration storage for search spaces
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [x] 7.2 Implement grid search functionality
    - Implement `grid_search()` method in HyperparameterTuner
    - Evaluate all combinations of context_window (3-10), rare_threshold (1-5), top_k (1-10)
    - For each configuration: train model, measure accuracy and speed on test set
    - Store results with configuration and metrics
    - _Requirements: 5.1, 5.2, 6.1, 6.2, 7.1, 7.2, 12.1, 12.2, 12.4_

  - [x] 7.3 Implement Pareto-optimal configuration selection
    - Add `identify_pareto_front()` method to find accuracy-speed tradeoffs
    - Implement `recommend_config()` method with priority options (accuracy, speed, balanced)
    - Return best configuration based on user priority
    - Save all tested configurations to JSON for analysis
    - _Requirements: 12.3, 12.4, 12.5_

  - [x]* 7.4 Write unit tests for hyperparameter tuning
    - Test grid search completeness (all combinations evaluated)
    - Test Pareto front identification
    - Test recommendation logic for different priorities
    - _Requirements: 12.1, 12.2, 12.3, 12.5_

  - [x] 7.5 Implement copy gate threshold optimization
    - Add copy gate threshold evaluation to grid search
    - Ensure tokens with frequency < threshold are marked as copy candidates
    - Integrate with rare_threshold parameter in grid search
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x]* 7.6 Write property test for copy gate threshold
    - **Property 6: Copy Gate Threshold**
    - **Validates: Requirements 6.4**
    - Test that tokens with frequency < threshold are marked for copy extraction
    - Verify copy gate activations in metrics
    - _Requirements: 6.4_

- [x] 8. Checkpoint - Verify hyperparameter optimization
  - Ensure all tests pass, validate tuner finds optimal configurations, ask the user if questions arise.

- [x] 9. Implement memory-efficient storage
  - [x] 9.1 Optimize sparse table storage format
    - Review `SparseEvidenceTables` storage in `_logit_tables.py`
    - Ensure feature_next and hdc_next use Dict[str, Counter[str]] (already sparse)
    - Implement pruning of zero-count entries during updates
    - Add memory footprint tracking method
    - _Requirements: 14.1, 14.2, 14.4_

  - [x] 9.2 Implement compressed serialization
    - Add `save()` method to SparseLogitGenerator using gzip compression
    - Add `load()` class method to deserialize compressed models
    - Use pickle for object serialization with gzip wrapper
    - Verify memory footprint <500MB for 10K+ training pairs
    - _Requirements: 14.3, 14.4_

  - [x]* 9.3 Write unit tests for memory-efficient storage
    - Test sparse storage (no dense arrays)
    - Test serialization/deserialization round-trip
    - Test compression effectiveness
    - Test memory limits for large datasets
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [x] 10. Integration and end-to-end validation
  - [x] 10.1 Create comprehensive benchmark dataset
    - Create `packages/puhl_luck/puhl_luck/benchmarks/benchmark_data.py`
    - Add code completion examples (10 training functions, 5 test cases)
    - Add sentiment classification examples (20 training, 10 test)
    - Add pattern matching examples (15 training sequences, 8 test)
    - Add Q&A examples (12 training pairs, 6 test questions)
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [x] 10.2 Run full benchmark suite and validate targets
    - Execute BenchmarkSuite on all task types
    - Validate accuracy >85% on code generation (Requirement 1.1)
    - Validate accuracy >85% on classification (Requirement 1.2)
    - Validate accuracy >85% on pattern matching (Requirement 1.3)
    - Validate accuracy >85% on Q&A (Requirement 1.4)
    - Validate inference speed <50ms per query (Requirement 2.1, 2.2)
    - Validate training speed <1000ms for 10 examples (Requirement 11.6)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x]* 10.3 Write integration tests for end-to-end workflows
    - Test code completion workflow (train on functions, generate similar)
    - Test classification workflow (train on labeled examples, classify new)
    - Test pattern matching workflow (train on sequences, complete patterns)
    - Test Q&A workflow (train on Q&A pairs, answer new questions)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.1, 4.2, 4.3_

  - [x]* 10.4 Write property test for generalization
    - **Property 2: Generalization Similarity**
    - **Validates: Requirements 4.1**
    - Test that perturbed inputs (variable renames, whitespace changes) produce >70% token overlap
    - Use Hypothesis to generate input perturbations
    - _Requirements: 4.1, 4.3_

  - [x] 10.5 Run hyperparameter tuning and apply best configuration
    - Execute full grid search on benchmark data
    - Identify optimal context_window, rare_threshold, and top_k
    - Update default configuration in SparseLogitGenerator
    - Document optimal hyperparameters in results
    - _Requirements: 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 7.1, 7.2, 7.3_

  - [x] 10.6 Validate overfitting prevention
    - Train on sequential datasets A, B, C
    - Measure accuracy on A after learning B and C
    - Verify accuracy degradation <5% (Requirement 3.1)
    - Verify consistent accuracy across training phases within 10% (Requirement 3.3)
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x]* 10.7 Write integration tests for overfitting prevention
    - Test sequential learning without catastrophic forgetting
    - Test mixed-phase evaluation consistency
    - Test recency bias detection
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 11. Final checkpoint - Complete performance validation
  - Ensure all tests pass
  - Verify accuracy >85% on all tasks
  - Verify inference speed <50ms per query
  - Verify training speed <1000ms for 10 examples
  - Verify memory usage <500MB for 10K pairs
  - Generate final benchmark report with before/after comparison
  - Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation after major phases
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows across all task types
- The implementation uses Python (current codebase language) with optional Rust acceleration
- Rust acceleration is already partially implemented in `rust_core/` and integrated in `_brain_hdc.py`
- Performance targets: >85% accuracy (improved from 33.3%), <50ms inference (improved from 178ms)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3"] },
    { "id": 1, "tasks": ["1.2", "1.4", "3.1", "5.1", "7.1", "9.1"] },
    { "id": 2, "tasks": ["3.2", "3.3", "5.2", "5.3", "7.2", "9.2"] },
    { "id": 3, "tasks": ["3.4", "3.5", "5.4", "5.5", "7.3", "9.3"] },
    { "id": 4, "tasks": ["3.6", "5.6", "5.7", "7.4", "7.5"] },
    { "id": 5, "tasks": ["5.8", "7.6", "10.1"] },
    { "id": 6, "tasks": ["10.2", "10.3", "10.4"] },
    { "id": 7, "tasks": ["10.5", "10.6", "10.7"] }
  ]
}
```
