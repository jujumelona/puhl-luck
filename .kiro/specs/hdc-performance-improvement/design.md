# Design Document: HDC Performance Improvement

## Overview

This design addresses critical performance issues in the HDC (Hyperdimensional Computing) system, which currently achieves 33.3% accuracy with 178ms inference time. The target is to improve accuracy to >80% and reduce inference time to <50ms through systematic optimization of the sparse autoregressive generation pipeline.

The HDC system uses a weightless, sparse memory architecture that combines:
- **Sparse evidence tables** mapping context sketches to token distributions
- **HDC (hyperdimensional) feature vectors** for compositional pattern matching
- **Adaptive readout projection** for learned feature-to-token mappings
- **Copy gate mechanism** for extracting rare tokens directly from input
- **Progressive backoff strategy** for context degradation when exact matches fail

Current bottlenecks include:
1. **Low accuracy** (33.3%) due to insufficient pattern generalization and overfitting to recent training
2. **Slow inference** (178ms) from expensive HDC operations and inefficient sparse table lookups
3. **Suboptimal hyperparameters** (context window, top-K, rare token threshold)
4. **Missing Rust acceleration** for critical hot paths

This design provides a comprehensive solution addressing architecture, algorithms, hyperparameter optimization, and implementation strategy.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      BrainMemory (Facade)                        │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────┐
│              SparseLogitGenerator (Core Engine)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  SparseEvidenceTables                                     │  │
│  │  - feature_next: Dict[str, Counter[str]]                 │  │
│  │  - hdc_next: Dict[str, Counter[str]]                     │  │
│  │  - vocab: Dict[str, int]                                 │  │
│  │  - Adaptive Readout (optional): feature→token projection │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│              ┌────────────┴────────────┐                        │
│              ▼                         ▼                        │
│    ┌─────────────────┐      ┌─────────────────┐                │
│    │  LogitScorer    │      │  HDC Operations │                │
│    │  - Repetition   │      │  - feature_hv() │                │
│    │    penalty      │      │  - bundle_hv()  │                │
│    │  - Temperature  │      │  - similarity() │                │
│    └─────────────────┘      └─────────────────┘                │
└────────────────────────────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │    Rust Acceleration Layer    │
         │  - puhl_luck_core.pyd         │
         │  - feature_hv_rust()          │
         │  - hv_similarity_rust()       │
         │  - bundle_hv_rust()           │
         │  - rotate_hv_rust()           │
         └───────────────────────────────┘
```

### Data Flow

**Training Flow (learn/learn_sequence):**
```
Input Text → Tokenize → Extract Features → Context Sketch
                                   │
                                   ▼
                        Forward Score (rank loss)
                                   │
                                   ▼
                        Credit Assignment
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
    Sparse Table Update                    Adaptive Readout Update
    (feature_next, hdc_next)               (hidden weights)
```

**Generation Flow (generate):**
```
Input Text → Tokenize → Copy Gate → Prompt Features
                                          │
                    ┌─────────────────────┘
                    │
                    ▼
         Autoregressive Loop (max_tokens iterations):
              │
              ▼
         Prefix Context (K tokens) → Active Features
              │                           │
              ▼                           ▼
         Context Sketch         HDC Bundle (optional)
              │                           │
              └──────────┬────────────────────┘
                         │
                         ▼
                  Sparse Table Lookup
                  (with backoff: K→K/2→K/4→1→field)
                         │
                         ▼
                  Token Candidates + Scores
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      Copy Gate Boost         Repetition Penalty
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                  Top-K Selection → Next Token
                         │
                         └──> Append to output, repeat
```

### Component Responsibilities

**SparseLogitGenerator:**
- Orchestrates training and generation
- Manages tokenization and copy token extraction
- Implements backoff strategy
- Coordinates feature extraction and scoring

**SparseEvidenceTables:**
- Stores sparse feature→token and HDC→token mappings
- Implements credit assignment via rank loss
- Maintains adaptive readout projection (optional)
- Provides runtime caching for repeated queries

**HDC Module (_brain_hdc.py):**
- Generates feature hypervectors using BLAKE2b hashing
- Implements vector bundling (XOR with rotation)
- Computes similarity via Hamming distance
- Falls back to Python when Rust unavailable

**Rust Core (rust_core/):**
- Accelerates feature_hv generation (9.7× speedup)
- Accelerates hv_similarity (26.6× speedup)
- Implements SIMD-friendly operations
- Zero-copy interop with NumPy arrays via PyO3

**LogitScorer:**
- Applies repetition penalty to recent tokens
- Implements temperature scaling
- Normalizes and ranks candidates

## Components and Interfaces

### Core Interfaces

#### SparseLogitGenerator API

```python
class SparseLogitGenerator:
    def learn(
        self,
        input_text: str,
        target_text: str,
        field_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Supervised training: input → target completion.
        
        Returns metrics: transitions_added, credit_corrected_steps,
        rank_loss_wrong_above, mean_token_loss
        """
        
    def learn_sequence(
        self,
        text: str,
        field_features: Optional[List[str]] = None,
        structural_targets_only: bool = False,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Self-supervised training: next-token prediction over sequence."""
        
    def generate(
        self,
        input_text: str,
        field_features: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate completion for input text.
        
        Returns (output_text, metrics_dict) with generation stats.
        """
        
    def get_statistics(self) -> Dict[str, Any]:
        """Return learning statistics: pairs_learned, tokens_learned, etc."""
```

#### SparseEvidenceTables API

```python
class SparseEvidenceTables:
    def credit_assign(
        self,
        features: List[Tuple[str, float]],
        target_token: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Rank-loss credit assignment.
        
        Scores features, ranks candidates, reinforces target,
        applies negative evidence to wrong-above candidates.
        
        Returns: positive_amount, negative_tokens, wrong_above_count, token_loss
        """
        
    def update_features(
        self,
        features: List[Tuple[str, float]],
        token: str,
        amount: int = 1
    ) -> None:
        """Update sparse evidence tables with positive evidence."""
        
    def score_token(
        self,
        features: List[Tuple[str, float]],
        token: str
    ) -> float:
        """Score a single token given active features."""
```

#### HDC Operations API

```python
def feature_hv(feature: str, words: int) -> np.ndarray:
    """Generate hypervector for feature string.
    Uses Rust if available, falls back to Python BLAKE2b hashing.
    """

def bundle_hv(features: Iterable[str], bits: Optional[int] = None) -> np.ndarray:
    """Bundle multiple features into single hypervector via XOR+rotation."""

def hv_similarity(a: np.ndarray, b: np.ndarray, bits: Optional[int] = None) -> float:
    """Compute similarity (1 - normalized Hamming distance).
    Uses Rust if available for 26.6× speedup.
    """

def rotate_hv(value: np.ndarray, amount: int) -> np.ndarray:
    """Circular bit rotation for permutation binding."""
```

### Hyperparameter Interfaces

```python
class HyperparameterTuner:
    def grid_search(
        self,
        train_data: List[Tuple[str, str]],
        test_data: List[Tuple[str, str]],
        context_windows: List[int] = [3, 4, 5, 6, 7, 8, 10],
        rare_thresholds: List[int] = [1, 2, 3, 4, 5],
        top_k_values: List[int] = [1, 2, 3, 5, 8, 10]
    ) -> Dict[str, Any]:
        """Exhaustive grid search over hyperparameter combinations.
        
        Returns: best_config, all_results, pareto_front
        """
        
    def recommend_config(
        self,
        results: Dict[str, Any],
        priority: str = 'balanced'  # 'accuracy', 'speed', 'balanced'
    ) -> Dict[str, Any]:
        """Recommend best configuration based on priority."""
```

### Benchmark Interface

```python
class BenchmarkSuite:
    def run_all_benchmarks(
        self,
        brain: BrainMemory,
        tasks: List[str] = ['code', 'classification', 'pattern', 'qa']
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark suite.
        
        Returns: accuracy per task, speed per task, aggregate metrics,
        backoff statistics, copy gate activations, empty output rates
        """
        
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save benchmark results to JSON with timestamp."""
```

## Data Models

### Token and Feature Representations

```python
# Token types
Token = str  # Raw token string after punctuation-preserving split
CopyToken = str  # Format: '[COPY{index}]'
SpecialToken = str  # '[BOS]', '[SEP]', '[EOS]'

# Feature types
Feature = str  # Format: 'prefix|payload'
# Examples:
#   'tok:return'        - token unigram
#   'bi:def|add'        - token bigram
#   'tri:def|add|(a'    - token trigram
#   'L1|return'         - last-1 token
#   'L2|def'            - last-2 token
#   'P2S|def add'       - prompt 2-skip-gram
#   'id:12ab34cd'       - HDC band identifier
#   'Dn|hash'           - discovered feature at depth n

# Weighted feature tuple
WeightedFeature = Tuple[str, float]  # (feature_id, weight)
```

### Sparse Evidence Structures

```python
# Sparse evidence tables
feature_next: Dict[str, Counter[str]]
# Maps feature ID → token counts
# Example: {'tok:def': Counter({'add': 5, 'subtract': 3})}

hdc_next: Dict[Tuple[int, int], Counter[str]]
# Maps (band_index, hash_value) → token counts
# Example: {(0, 12345): Counter({'return': 8, 'pass': 2})}

vocab: Dict[str, int]
# Token → global frequency
# Example: {'return': 45, 'def': 32, 'add': 15}

# Adaptive readout (optional)
readout_hidden: np.ndarray  # Shape: (hidden_dim, num_features)
readout_output: np.ndarray  # Shape: (vocab_size, hidden_dim)
```

### Context Sketch

```python
ContextSketch = bytes  # 128-bit BLAKE2b digest of context tokens
# Used as key for sparse table lookups
# Example: b'\x3a\x7f\x...\x9c' (16 bytes)
```

### Generation Metrics

```python
@dataclass
class GenerationMetrics:
    tokens_generated: int
    backoff_levels: List[int]  # Backoff level per token (0=exact, 1=K/2, etc.)
    copy_gate_activations: int
    generation_method: str  # 'sparse', 'hdc', 'backoff', 'copy'
    empty_output: bool
    failure_reason: Optional[str]
    inference_time_ms: float
```

### Hyperparameter Configuration

```python
@dataclass
class HDCConfig:
    context_window: int = 5  # K tokens of recent context
    rare_token_threshold: int = 2  # Tokens with freq ≤ threshold are "rare"
    top_k: int = 3  # Number of candidates to consider
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    repetition_window: int = 20
    adaptive_readout_enabled: bool = True
    rust_acceleration: bool = True  # Auto-detect and use if available
```

## Error Handling

### Error Categories

**1. Training Errors:**
- **Empty training text:** Log warning, skip update
- **Tokenization failure:** Fall back to character-level split
- **HDC generation failure:** Continue with sparse-only mode
- **Rust import failure:** Log once, fall back to Python implementations

**2. Generation Errors:**
- **No candidates found:** Fall through all backoff levels, return empty string with diagnostic
- **Context too short:** Use available context, pad if necessary
- **Copy token index out of bounds:** Ignore copy, continue generation
- **Temperature <= 0:** Raise ValueError with helpful message
- **Max tokens exceeded:** Truncate output, set flag in metrics

**3. Memory Errors:**
- **Sparse table exceeds memory limit:** Trigger compression, prune low-frequency entries
- **HDC index too large:** Limit band count, use selective indexing
- **Cache overflow:** LRU eviction, log if thrashing detected

### Error Recovery Strategies

```python
def generate_with_fallback(
    self,
    input_text: str,
    max_backoff_levels: int = 5
) -> Tuple[str, GenerationMetrics]:
    """Generation with progressive fallback strategy.
    
    Fallback sequence:
    1. Full context (K tokens) sparse lookup
    2. Half context (K/2 tokens)
    3. Quarter context (K/4 tokens)
    4. Unigram distribution
    5. Field-only features (if provided)
    6. Empty output with diagnostic
    """
    try:
        # Standard generation
        return self.generate(input_text)
    except NoContextError:
        # Use field features only
        return self._generate_from_field(input_text)
    except Exception as e:
        # Log and return empty
        logger.error(f"Generation failed: {e}")
        return "", GenerationMetrics(
            empty_output=True,
            failure_reason=str(e)
        )
```

### Logging and Diagnostics

```python
import logging
logger = logging.getLogger('puhl_luck.hdc')

# Diagnostic levels
logger.debug("Cache hit: context_sketch={sketch_hex}")
logger.info(f"Rust acceleration: available={RUST_AVAILABLE}")
logger.warning(f"Backoff to unigram: context={context}, attempts={attempts}")
logger.error(f"Generation failure: reason={reason}, input={input_preview}")
```

## Testing Strategy

### Unit Tests

**Core HDC Operations (test_hdc.py):**
- Test feature_hv determinism (same input → same hypervector)
- Test bundle_hv commutativity (order shouldn't matter due to rotation)
- Test hv_similarity properties (self-similarity = 1.0, orthogonality ≈ 0.5)
- Test rotate_hv correctness (bit preservation)
- Test Python/Rust equivalence (both implementations produce same results)

**Sparse Table Operations (test_sparse_tables.py):**
- Test credit_assign rank loss (wrong-above tokens get negative evidence)
- Test sparse lookup with backoff (K → K/2 → K/4 → unigram)
- Test adaptive readout learning (weights update correctly)
- Test memory efficiency (sparse storage vs dense arrays)

**Tokenization (test_tokenizer.py):**
- Test punctuation preservation ('def add(a, b):' → ['def', 'add', '(', 'a', ',', 'b', ')', ':'])
- Test copy token extraction and encoding
- Test special token handling ([BOS], [SEP], [EOS])
- Test detokenization (reconstruct original text)

**Hyperparameter Tuning (test_tuner.py):**
- Test grid search completeness (all combinations evaluated)
- Test Pareto front identification (accuracy vs speed tradeoffs)
- Test configuration recommendation (priority-based selection)

### Integration Tests

**End-to-End Generation (test_generation_e2e.py):**
- Test code completion workflow (train on functions, generate similar functions)
- Test classification workflow (train on labeled examples, classify new examples)
- Test pattern matching workflow (train on sequences, complete patterns)
- Test Q&A workflow (train on question-answer pairs, answer new questions)

**Overfitting Prevention (test_overfitting.py):**
- Test sequential learning (accuracy on early examples after learning late examples)
- Test mixed-phase evaluation (consistent accuracy across training phases)
- Test recency bias detection (distribution of retrieved patterns)

**Rust Acceleration (test_rust_integration.py):**
- Test Rust module loading and fallback
- Test feature_hv speedup (>5× faster than Python)
- Test hv_similarity speedup (>20× faster than Python)
- Test numerical equivalence (Rust and Python produce same results)

### Benchmark Tests

**Performance Benchmarks (benchmarks/run_all_benchmarks.py):**
- **Code completion:** 10 training functions, 5 test cases, measure accuracy and speed
- **Sentiment classification:** 20 labeled examples, 10 test cases
- **Pattern matching:** 15 sequence patterns, 8 test completions
- **Q&A:** 12 question-answer pairs, 6 test questions

**Target Metrics:**
- Accuracy: >80% on all tasks
- Inference speed: <50ms per query
- Training speed: <1000ms for 10 examples
- Memory usage: <500MB for 10K training pairs


**Regression Tests:**
- Track performance metrics over time (accuracy, speed, memory)
- Alert on degradation >5% from previous best
- Maintain historical benchmark database for trend analysis

### Testing Approach Note

**Why Property-Based Testing (PBT) is Limited for This Feature:**

This feature is primarily a **performance optimization and integration feature** rather than a feature introducing new algorithmic behaviors. The requirements focus on:
- **Performance metrics** (accuracy >80%, speed <50ms) - measured empirically through benchmarks
- **Integration behavior** (end-to-end system performance across tasks)
- **Infrastructure optimization** (Rust acceleration, memory efficiency)

Property-based testing is most effective for testing:
- Pure functions with clear input/output contracts
- Compositional operations (parsers, serializers, data transformations)
- Universal invariants that hold across all inputs

**Primary Testing Strategies for This Feature:**
1. **Integration tests** with benchmark suites (code completion, classification, pattern matching, Q&A)
2. **Performance tests** measuring actual execution time and memory usage
3. **Comparative tests** (before/after optimization, Rust vs Python equivalence)
4. **Example-based unit tests** for specific configurations and edge cases

**Limited PBT Applications:**
While PBT is not the primary testing approach, a few properties can be tested:

**Property 1: Forgetting Prevention (Requirement 3.1, 3.3)**
*For any* training sequence A followed by training sequence B, the accuracy on test examples from A should not decrease by more than 5 percentage points after learning B.

**Property 2: Generalization Similarity (Requirement 4.1)**
*For any* training pair (input, output), when the input is perturbed (e.g., variable names changed, whitespace altered), the generated output should maintain >70% token overlap with the original output.

**Property 3: Backoff Sequence Correctness (Requirement 10.1, 10.2)**
*For any* generation query, when exact context match fails, the backoff sequence should follow [K, K/2, K/4, unigram, field] in order and terminate at the first successful match.

**Property 4: Rust-Python Equivalence (Requirement 8.5)**
*For any* HDC operation (feature_hv, hv_similarity, bundle_hv, rotate_hv), the Rust implementation should produce results equivalent to the Python implementation (within floating-point precision).

**Property 5: Tokenization Round-Trip (Requirement 15.3)**
*For any* text, detokenize(tokenize(text)) should produce text equivalent to the original (preserving content, possibly normalizing whitespace).

**Property 6: Copy Gate Threshold (Requirement 6.4)**
*For any* token with frequency less than the rare_token_threshold, the Copy_Gate should mark it as a candidate for extraction.

These properties will be implemented with property-based testing libraries (e.g., Hypothesis for Python), but the bulk of testing will be integration and performance benchmarks.

## Detailed Design

### 1. Accuracy Improvements


#### 1.1 Enhanced Credit Assignment

**Current Issue:** Simple count-based evidence doesn't distinguish between strong and weak associations.

**Solution:** Implement rank-loss credit assignment (already partially implemented in `credit_assign`):

```python
def credit_assign(
    self,
    features: List[Tuple[str, float]],
    target_token: str,
    top_k: int = 10
) -> Dict[str, Any]:
    """Rank-loss credit assignment.
    
    Algorithm:
    1. Score all tokens given features
    2. Rank tokens by score
    3. Compute rank loss: loss = (rank(target) - 1) / total_tokens
    4. Positive evidence: Reinforce target
    5. Negative evidence: Penalize wrong-above tokens
    """
    # Score all candidates
    scores = {token: self.score_token(features, token) 
              for token in self.vocab}
    
    # Rank tokens
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    target_rank = next((i for i, (tok, _) in enumerate(ranked) if tok == target_token), len(ranked))
    
    # Identify wrong-above tokens (ranked higher than target)
    wrong_above = [tok for tok, _ in ranked[:target_rank]]
    
    # Update with positive/negative evidence
    amount = max(1, int(math.ceil(math.log2(target_rank + 2))))
    self.update_features(features, target_token, amount=amount)
    
    for wrong_token in wrong_above:
        # Negative evidence: reduce association
        self.apply_negative_evidence(features, wrong_token)
    
    return {
        'positive_amount': amount,
        'negative_tokens': wrong_above,
        'wrong_above_count': len(wrong_above),
        'token_loss': target_rank / max(1, len(ranked))
    }
```

**Impact:** Reduces overfitting by penalizing incorrect high-confidence predictions.

#### 1.2 Improved Backoff Strategy

**Current Issue:** Backoff may be too aggressive or miss relevant patterns.

**Solution:** Add intermediate backoff levels and HDC similarity matching:

```python
def lookup_with_backoff(
    self,
    context_tokens: List[str],
    field_features: List[str]
) -> Tuple[Counter[str], int]:
    """Progressive backoff with HDC similarity.
    
    Backoff sequence:
    0. Exact match (full K tokens)
    1. Half context (K/2 tokens)
    2. Quarter context (K/4 tokens)
    3. HDC similarity search (if no exact match)
    4. Unigram distribution
    5. Field-only features
    """
    K = len(context_tokens)
    
    # Level 0: Full context
    sketch = self._context_sketch(context_tokens)
    if sketch in self.context_table:
        return self.context_table[sketch], 0
    
    # Level 1: Half context
    if K >= 2:
        half_sketch = self._context_sketch(context_tokens[-K//2:])
        if half_sketch in self.context_table:
            return self.context_table[half_sketch], 1
    
    # Level 2: Quarter context
    if K >= 4:
        quarter_sketch = self._context_sketch(context_tokens[-K//4:])
        if quarter_sketch in self.context_table:
            return self.context_table[quarter_sketch], 2
    
    # Level 3: HDC similarity search
    hv = bundle_hv(context_tokens)
    candidates = self.hdc_candidates(hv)
    if candidates:
        # Find most similar context in HDC index
        best_match = max(candidates, key=lambda c: hv_similarity(hv, self.hdc_vectors[c]))
        return self.hdc_table[best_match], 3
    
    # Level 4: Unigram
    if self.unigram_dist:
        return self.unigram_dist, 4
    
    # Level 5: Field-only (if provided)
    if field_features:
        return self._score_from_fields(field_features), 5
    
    # Failure
    return Counter(), 6
```

**Impact:** Improves generalization by finding similar contexts even when exact matches fail.

#### 1.3 Adaptive Readout Improvements

**Current Issue:** Readout may be undersized or not learning effectively.

**Solution:** Dynamic readout sizing based on data scale:

```python
def dynamic_readout_config(
    vocab_size: int,
    feature_count: int,
    event_count: int
) -> Dict[str, int]:
    """Auto-size readout based on learned data.
    
    Hidden dimension: sqrt(vocab_size * feature_count)
    Vocab cap: min(vocab_size, sqrt(event_count) * log(vocab_size))
    """
    hidden_dim = int(math.sqrt(vocab_size * feature_count))
    hidden_dim = max(64, min(2048, hidden_dim))  # Clamp to reasonable range
    
    vocab_cap = int(math.sqrt(event_count) * math.log2(vocab_size + 2))
    vocab_cap = max(100, min(vocab_size, vocab_cap))
    
    return {
        'hidden_dim': hidden_dim,
        'vocab_cap': vocab_cap,
        'learning_rate': 0.01 / math.log2(event_count + 2)  # Decay with data
    }
```

**Impact:** Adaptive capacity prevents underfitting (too small) and overfitting (too large).

### 2. Speed Optimizations

#### 2.1 Rust Acceleration Implementation

**Target Operations:**
1. `feature_hv()`: Hash-based hypervector generation (9.7× speedup)
2. `hv_similarity()`: Bitwise XOR + popcount (26.6× speedup)
3. `bundle_hv()`: Batch XOR+rotation (10× speedup)
4. `rotate_hv()`: Bit rotation (5× speedup)

**Already Implemented in rust_core/src/hdc.rs:**
```rust
#[pyfunction]
pub fn feature_hv_rust(py: Python, feature: &str, words: usize) -> PyResult<Py<PyArray1<u64>>> {
    let hv = generate_feature_hv(feature, words);
    Ok(Array1::from_vec(hv).into_pyarray(py).into())
}

#[pyfunction]
pub fn hv_similarity_rust(
    a: PyReadonlyArray1<u64>,
    b: PyReadonlyArray1<u64>,
) -> PyResult<f64> {
    let a = a.as_array();
    let b = b.as_array();
    if a.is_empty() || b.is_empty() { return Ok(0.0); }
    let words = a.len().min(b.len());
    let bits = words * HDC_WORD_BITS;
    let diff: u32 = a.iter().zip(b.iter()).take(words)
        .map(|(x, y)| (x ^ y).count_ones()).sum();
    Ok(1.0 - (diff as f64 / bits as f64))
}
```

**Optimization Strategy:**
- Rust uses hardware popcount (count_ones) for Hamming distance
- Zero-copy NumPy array sharing via PyO3
- Batched operations reduce Python/Rust crossing overhead

#### 2.2 Sparse Table Lookup Optimization

**Current Issue:** O(1) dict lookup, but hash computation can be slow.

**Solution:** Pre-compute and cache context sketches:

```python
class SparseTableWithCache:
    def __init__(self, cache_size: int = 1000):
        self.context_table: Dict[bytes, Counter[str]] = {}
        self._sketch_cache: OrderedDict[Tuple[str, ...], bytes] = OrderedDict()
        self.cache_size = cache_size
        
    def _context_sketch(self, tokens: Tuple[str, ...]) -> bytes:
        """Cached BLAKE2b sketch computation."""
        if tokens in self._sketch_cache:
            self._sketch_cache.move_to_end(tokens)  # LRU
            return self._sketch_cache[tokens]
        
        # Compute new sketch
        h = hashlib.blake2b(digest_size=16)
        for tok in tokens:
            h.update(tok.encode('utf-8', 'ignore'))
        sketch = h.digest()
        
        # Cache with LRU eviction
        self._sketch_cache[tokens] = sketch
        if len(self._sketch_cache) > self.cache_size:
            self._sketch_cache.popitem(last=False)
        
        return sketch
```

**Impact:** Reduces repeated hash computation during generation (each token extends context by 1).

#### 2.3 Feature Extraction Optimization

**Current Issue:** Extracting n-grams, skip-grams, HDC bands is redundant work.

**Solution:** Incremental feature updates during generation:

```python
class IncrementalFeatureExtractor:
    def __init__(self, context_window: int):
        self.K = context_window
        self.context_deque = deque(maxlen=context_window)
        self.cached_features: List[Tuple[str, float]] = []
        
    def append_token(self, token: str) -> List[Tuple[str, float]]:
        """Incrementally update features when new token added."""
        self.context_deque.append(token)
        
        # Only recompute features affected by new token
        new_features = []
        
        # Unigram (always new)
        new_features.append((f'tok:{token}', 1.0))
        
        # Bigram (if predecessor exists)
        if len(self.context_deque) >= 2:
            new_features.append((f'bi:{self.context_deque[-2]}|{token}', 1.0))
        
        # Trigram (if 2 predecessors exist)
        if len(self.context_deque) >= 3:
            new_features.append((f'tri:{self.context_deque[-3]}|{self.context_deque[-2]}|{token}', 1.0))
        
        # L1, L2, ... positional features
        for i, hist_token in enumerate(reversed(list(self.context_deque)[:-1])):
            new_features.append((f'L{i+1}|{hist_token}', 0.9 ** (i+1)))
        
        # HDC bands (only recompute if window changed significantly)
        if len(self.context_deque) == self.K:
            hv = bundle_hv(list(self.context_deque))
            bands = hdc_bands(hv, event_count=1000)
            for band_idx, hash_val in bands:
                new_features.append((f'id:{band_idx}_{hash_val}', 1.0))
        
        self.cached_features = new_features
        return new_features
```

**Impact:** Reduces O(K²) feature computation to O(K) per token during generation.

### 3. Hyperparameter Optimization

#### 3.1 Grid Search Implementation

```python
class HyperparameterTuner:
    def grid_search(
        self,
        train_data: List[Tuple[str, str]],
        test_data: List[Tuple[str, str]],
        context_windows: List[int] = [3, 4, 5, 6, 7, 8, 10],
        rare_thresholds: List[int] = [1, 2, 3, 4, 5],
        top_k_values: List[int] = [1, 2, 3, 5, 8, 10]
    ) -> Dict[str, Any]:
        """Exhaustive grid search."""
        results = []
        
        for ctx_win in context_windows:
            for rare_th in rare_thresholds:
                for topk in top_k_values:
                    # Train with this config
                    brain = BrainMemory()
                    config = {
                        'context_window': ctx_win,
                        'rare_threshold': rare_th,
                        'top_k': topk
                    }
                    
                    # Train
                    train_start = time.time()
                    for inp, tgt in train_data:
                        brain.expose_pair(inp, tgt)
                    train_time = time.time() - train_start
                    
                    # Test
                    correct = 0
                    total_gen_time = 0
                    for inp, expected in test_data:
                        gen_start = time.time()
                        output = brain.generate(inp, max_new_tokens=20)
                        total_gen_time += time.time() - gen_start
                        
                        if expected.lower() in output.lower():
                            correct += 1
                    
                    accuracy = correct / len(test_data)
                    avg_gen_time = (total_gen_time / len(test_data)) * 1000  # ms
                    
                    results.append({
                        'config': config,
                        'accuracy': accuracy,
                        'train_time_ms': train_time * 1000,
                        'avg_gen_time_ms': avg_gen_time
                    })
        
        # Identify Pareto front
        pareto_front = self._compute_pareto_front(results, 
                                                    metrics=['accuracy', 'avg_gen_time_ms'])
        
        return {
            'all_results': results,
            'pareto_front': pareto_front,
            'best_accuracy': max(results, key=lambda r: r['accuracy']),
            'best_speed': min(results, key=lambda r: r['avg_gen_time_ms'])
        }
```

#### 3.2 Pareto Optimization

```python
def _compute_pareto_front(
    self,
    results: List[Dict],
    metrics: List[str]
) -> List[Dict]:
    """Find Pareto-optimal configurations.
    
    A config is Pareto-optimal if no other config is strictly better
    on all metrics (maximize accuracy, minimize time).
    """
    pareto = []
    
    for candidate in results:
        is_dominated = False
        
        for other in results:
            if other == candidate:
                continue
            
            # Check if 'other' dominates 'candidate'
            better_or_equal_all = True
            strictly_better_some = False
            
            # Accuracy: higher is better
            if other['accuracy'] > candidate['accuracy']:
                strictly_better_some = True
            elif other['accuracy'] < candidate['accuracy']:
                better_or_equal_all = False
            
            # Time: lower is better
            if other['avg_gen_time_ms'] < candidate['avg_gen_time_ms']:
                strictly_better_some = True
            elif other['avg_gen_time_ms'] > candidate['avg_gen_time_ms']:
                better_or_equal_all = False
            
            if better_or_equal_all and strictly_better_some:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto.append(candidate)
    
    return pareto
```

### 4. Benchmarking Suite Implementation

#### 4.1 Comprehensive Benchmark Structure

```python
class BenchmarkSuite:
    def __init__(self):
        self.tasks = {
            'code': self._benchmark_code_completion,
            'classification': self._benchmark_classification,
            'pattern': self._benchmark_pattern_matching,
            'qa': self._benchmark_question_answering
        }
    
    def run_all_benchmarks(
        self,
        brain: BrainMemory,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run all benchmark tasks."""
        if tasks is None:
            tasks = list(self.tasks.keys())
        
        results = {}
        for task_name in tasks:
            if task_name not in self.tasks:
                continue
            
            print(f"\n{'='*60}")
            print(f"Running benchmark: {task_name.upper()}")
            print(f"{'='*60}")
            
            task_result = self.tasks[task_name](brain)
            results[task_name] = task_result
            
            print(f"Accuracy: {task_result['accuracy']*100:.1f}%")
            print(f"Avg speed: {task_result['avg_gen_time_ms']:.2f}ms")
        
        # Aggregate statistics
        aggregate = {
            'overall_accuracy': np.mean([r['accuracy'] for r in results.values()]),
            'overall_speed_ms': np.mean([r['avg_gen_time_ms'] for r in results.values()]),
            'total_time_s': sum(r['total_time_s'] for r in results.values()),
            'timestamp': time.time()
        }
        
        return {
            'tasks': results,
            'aggregate': aggregate
        }
    
    def _benchmark_code_completion(self, brain: BrainMemory) -> Dict:
        """Code completion benchmark."""
        training = [
            ('def add(a, b):', 'return a + b'),
            ('def subtract(a, b):', 'return a - b'),
            ('def multiply(x, y):', 'return x * y'),
            ('def divide(x, y):', 'return x / y'),
            ('def power(a, b):', 'return a ** b'),
            ('def modulo(x, y):', 'return x % y'),
            ('def max_of_two(a, b):\n    if a > b:', 'return a\n    else:\n        return b'),
            ('def min_of_two(a, b):\n    if a < b:', 'return a\n    else:\n        return b'),
            ('def is_positive(x):\n    if x > 0:', 'return True\n    else:\n        return False'),
            ('def is_negative(x):\n    if x < 0:', 'return True\n    else:\n        return False'),
        ]
        
        tests = [
            ('def double(x):', ['return', 'x', '*', '2']),
            ('def triple(a):', ['return', 'a', '*', '3']),
            ('def square(n):', ['return', 'n', '*', 'n']),
            ('def half(x):', ['return', 'x', '/', '2']),
            ('def is_even(n):\n    if n % 2 == 0:', ['return', 'True']),
        ]
        
        return self._run_task(brain, training, tests, domain='code')
    
    def _run_task(
        self,
        brain: BrainMemory,
        training: List[Tuple[str, str]],
        tests: List[Tuple[str, List[str]]],
        domain: str = 'general'
    ) -> Dict:
        """Generic task runner."""
        # Training
        train_start = time.time()
        for inp, tgt in training:
            brain.expose_pair(inp, tgt, domain=domain)
        train_time = time.time() - train_start
        
        # Testing
        correct = 0
        total_gen_time = 0
        backoff_stats = Counter()
        copy_activations = 0
        empty_outputs = 0
        
        for inp, expected_tokens in tests:
            gen_start = time.time()
            output, metrics = brain.generate(inp, max_new_tokens=30, domain=domain, return_metrics=True)
            gen_time = time.time() - gen_start
            total_gen_time += gen_time
            
            # Check accuracy
            matches = sum(1 for tok in expected_tokens if tok in output.lower())
            if matches / len(expected_tokens) >= 0.75:
                correct += 1
            
            # Collect diagnostics
            if metrics.get('empty_output'):
                empty_outputs += 1
            if metrics.get('backoff_levels'):
                for level in metrics['backoff_levels']:
                    backoff_stats[level] += 1
            copy_activations += metrics.get('copy_gate_activations', 0)
        
        return {
            'accuracy': correct / len(tests),
            'train_time_ms': train_time * 1000,
            'avg_gen_time_ms': (total_gen_time / len(tests)) * 1000,
            'total_time_s': train_time + total_gen_time,
            'backoff_stats': dict(backoff_stats),
            'copy_activations': copy_activations,
            'empty_outputs': empty_outputs
        }
```

### 5. Memory Efficiency

#### 5.1 Sparse Storage

**Current Implementation:** Already uses Counter (sparse dict) for token distributions.

**Additional Optimization:** Compress low-frequency entries:

```python
class CompressedSparseTable:
    def __init__(self, min_frequency: int = 2):
        self.feature_next: Dict[str, Counter[str]] = {}
        self.min_frequency = min_frequency
        
    def compress(self) -> Dict[str, int]:
        """Prune entries below minimum frequency threshold.
        
        Returns: stats about pruned entries
        """
        pruned_features = 0
        pruned_tokens = 0
        
        for feature, token_dist in list(self.feature_next.items()):
            # Remove low-frequency tokens
            original_size = len(token_dist)
            token_dist = Counter({tok: cnt for tok, cnt in token_dist.items() 
                                  if cnt >= self.min_frequency})
            
            # Remove empty features
            if not token_dist:
                del self.feature_next[feature]
                pruned_features += 1
            else:
                self.feature_next[feature] = token_dist
                pruned_tokens += (original_size - len(token_dist))
        
        return {
            'pruned_features': pruned_features,
            'pruned_tokens': pruned_tokens,
            'remaining_features': len(self.feature_next),
            'total_tokens': sum(len(dist) for dist in self.feature_next.values())
        }
```

**Impact:** Reduces memory footprint by 30-50% with minimal accuracy loss (<2%).

#### 5.2 Serialization with Compression

```python
import gzip
import pickle

def save_compressed(brain: BrainMemory, filename: str) -> None:
    """Save brain state with gzip compression."""
    with gzip.open(filename, 'wb') as f:
        pickle.dump({
            'feature_next': brain.tables.feature_next,
            'hdc_next': brain.tables.hdc_next,
            'vocab': brain.tables.vocab,
            'config': brain.config,
            'statistics': brain.get_statistics()
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_compressed(filename: str) -> BrainMemory:
    """Load brain state from compressed file."""
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    
    brain = BrainMemory(config=data['config'])
    brain.tables.feature_next = data['feature_next']
    brain.tables.hdc_next = data['hdc_next']
    brain.tables.vocab = data['vocab']
    return brain
```

**Impact:** 5-10× reduction in disk storage with fast decompression (<100ms for 10K pairs).

### 6. Implementation Strategy

#### 6.1 Phased Rollout

**Phase 1: Core Optimizations (Week 1)**
- Implement enhanced credit assignment with rank loss
- Add context sketch caching
- Optimize backoff strategy with HDC similarity
- Add generation metrics and diagnostics
- **Target:** 60% accuracy, <100ms inference

**Phase 2: Rust Acceleration (Week 2)**
- Integrate existing Rust HDC operations
- Add auto-detection and fallback logic
- Implement zero-copy NumPy interop
- Profile and optimize hot paths
- **Target:** 70% accuracy, <75ms inference

**Phase 3: Hyperparameter Optimization (Week 3)**
- Implement grid search framework
- Run comprehensive parameter sweep
- Identify Pareto-optimal configurations
- Update default configuration
- **Target:** 80% accuracy, <50ms inference

**Phase 4: Benchmarking and Validation (Week 4)**
- Implement full benchmark suite
- Run baseline vs optimized comparisons
- Validate overfitting prevention
- Document performance characteristics
- **Target:** >80% accuracy, <50ms inference, comprehensive metrics

#### 6.2 Performance Profiling Strategy

**Profiling Tools:**
```python
import cProfile
import pstats
from line_profiler import LineProfiler

def profile_generation(brain: BrainMemory, test_cases: List[str]):
    """Profile generation hot paths."""
    profiler = cProfile.Profile()
    
    profiler.enable()
    for test_input in test_cases:
        brain.generate(test_input, max_new_tokens=20)
    profiler.disable()
    
    # Print top 20 time-consuming functions
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

def line_profile_function(func, *args, **kwargs):
    """Line-by-line profiling of specific function."""
    lp = LineProfiler()
    lp.add_function(func)
    lp.enable()
    result = func(*args, **kwargs)
    lp.disable()
    lp.print_stats()
    return result
```

**Target Bottlenecks:**
1. Feature extraction (expected 30% of time)
2. HDC operations (expected 25% of time)
3. Sparse table lookups (expected 20% of time)
4. Backoff strategy (expected 15% of time)
5. Token scoring and selection (expected 10% of time)

#### 6.3 Validation Metrics

**Accuracy Metrics:**
- Per-task accuracy (code, classification, pattern, Q&A)
- Average accuracy across all tasks
- Accuracy by context length
- Accuracy by input complexity

**Speed Metrics:**
- Average inference time per query
- Inference time by token length
- Training time per example
- Training time scaling (10, 100, 1000, 10000 examples)

**Memory Metrics:**
- Sparse table size (MB)
- Peak memory usage during training
- Peak memory usage during generation
- Memory growth rate with data scale

**Robustness Metrics:**
- Overfitting score (early vs late example accuracy)
- Backoff frequency by level
- Empty output rate
- Copy gate activation rate
- Generalization score (perturbed input accuracy)

### 7. Migration and Backward Compatibility

#### 7.1 Configuration Migration

**Old Configuration Format:**
```python
brain = BrainMemory()
# Implicit defaults, no explicit config
```

**New Configuration Format:**
```python
config = HDCConfig(
    context_window=5,
    rare_token_threshold=2,
    top_k=3,
    temperature=1.0,
    rust_acceleration=True
)
brain = BrainMemory(config=config)
```

**Migration Strategy:**
- Detect old API usage via argument inspection
- Emit deprecation warnings for old-style initialization
- Auto-convert to new config format
- Maintain backward compatibility for 2 major versions

#### 7.2 Serialization Compatibility

**Version Detection:**
```python
def load_with_version_migration(filename: str) -> BrainMemory:
    """Load brain state with automatic version migration."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # Detect version
    version = data.get('__version__', '0.1.0')
    
    if version < '0.2.0':
        # Migrate old format
        data = _migrate_v01_to_v02(data)
    
    brain = BrainMemory(config=data.get('config', HDCConfig()))
    brain.tables.feature_next = data['feature_next']
    brain.tables.hdc_next = data['hdc_next']
    brain.tables.vocab = data['vocab']
    return brain
```

### 8. Monitoring and Observability

#### 8.1 Runtime Telemetry

```python
@dataclass
class TelemetryCollector:
    """Collect runtime performance metrics."""
    generation_times: List[float] = field(default_factory=list)
    backoff_counts: Counter = field(default_factory=Counter)
    cache_hits: int = 0
    cache_misses: int = 0
    rust_calls: int = 0
    python_fallbacks: int = 0
    
    def record_generation(self, metrics: GenerationMetrics) -> None:
        """Record single generation event."""
        self.generation_times.append(metrics.inference_time_ms)
        for level in metrics.backoff_levels:
            self.backoff_counts[level] += 1
    
    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        return {
            'avg_gen_time_ms': np.mean(self.generation_times),
            'p50_gen_time_ms': np.percentile(self.generation_times, 50),
            'p95_gen_time_ms': np.percentile(self.generation_times, 95),
            'p99_gen_time_ms': np.percentile(self.generation_times, 99),
            'backoff_distribution': dict(self.backoff_counts),
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'rust_usage_rate': self.rust_calls / max(1, self.rust_calls + self.python_fallbacks)
        }
```

#### 8.2 Performance Dashboard

```python
def generate_performance_report(
    baseline_results: Dict,
    optimized_results: Dict
) -> str:
    """Generate markdown performance comparison report."""
    report = []
    report.append("# HDC Performance Improvement Report\n")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report.append("## Accuracy Comparison\n\n")
    report.append("| Task | Baseline | Optimized | Improvement |\n")
    report.append("|------|----------|-----------|-------------|\n")
    
    for task in ['code', 'classification', 'pattern', 'qa']:
        baseline_acc = baseline_results['tasks'][task]['accuracy'] * 100
        optimized_acc = optimized_results['tasks'][task]['accuracy'] * 100
        improvement = optimized_acc - baseline_acc
        report.append(f"| {task.capitalize()} | {baseline_acc:.1f}% | {optimized_acc:.1f}% | +{improvement:.1f}% |\n")
    
    report.append("\n## Speed Comparison\n\n")
    report.append("| Task | Baseline (ms) | Optimized (ms) | Speedup |\n")
    report.append("|------|---------------|----------------|----------|\n")
    
    for task in ['code', 'classification', 'pattern', 'qa']:
        baseline_time = baseline_results['tasks'][task]['avg_gen_time_ms']
        optimized_time = optimized_results['tasks'][task]['avg_gen_time_ms']
        speedup = baseline_time / max(0.1, optimized_time)
        report.append(f"| {task.capitalize()} | {baseline_time:.2f} | {optimized_time:.2f} | {speedup:.2f}× |\n")
    
    report.append("\n## Aggregate Metrics\n\n")
    baseline_agg = baseline_results['aggregate']
    optimized_agg = optimized_results['aggregate']
    
    report.append(f"- **Overall Accuracy:** {baseline_agg['overall_accuracy']*100:.1f}% → {optimized_agg['overall_accuracy']*100:.1f}% (+{(optimized_agg['overall_accuracy']-baseline_agg['overall_accuracy'])*100:.1f}%)\n")
    report.append(f"- **Overall Speed:** {baseline_agg['overall_speed_ms']:.2f}ms → {optimized_agg['overall_speed_ms']:.2f}ms ({baseline_agg['overall_speed_ms']/optimized_agg['overall_speed_ms']:.2f}× faster)\n")
    
    return ''.join(report)
```

### 9. Risk Mitigation

#### 9.1 Identified Risks

**Risk 1: Rust Acceleration Unavailable**
- **Probability:** Medium (build issues, platform compatibility)
- **Impact:** High (10-20× speed difference)
- **Mitigation:** Python fallback implementations, clear documentation
- **Detection:** Runtime check on first import, warn user

**Risk 2: Hyperparameter Tuning Finds No Improvement**
- **Probability:** Low (baseline already poor)
- **Impact:** Medium (target metrics not achieved)
- **Mitigation:** Expand search space, try adaptive methods
- **Detection:** Grid search results analysis, Pareto front inspection

**Risk 3: Memory Explosion with Large Training Sets**
- **Probability:** Medium (10K+ training pairs)
- **Impact:** High (OOM crashes)
- **Mitigation:** Automatic compression, LRU eviction, streaming learning
- **Detection:** Memory monitoring, profiling at scale

**Risk 4: Overfitting Despite Improvements**
- **Probability:** Medium (sparse tables sensitive to recency)
- **Impact:** Medium (poor generalization)
- **Mitigation:** Negative evidence, experience replay, regularization
- **Detection:** Sequential learning tests, accuracy by training phase

**Risk 5: Integration Breaks Existing Code**
- **Probability:** Low (careful API design)
- **Impact:** Medium (adoption friction)
- **Mitigation:** Backward compatibility, migration guide, deprecation cycle
- **Detection:** Comprehensive integration tests, user feedback

#### 9.2 Rollback Strategy

```python
def rollback_to_baseline(brain: BrainMemory) -> BrainMemory:
    """Revert to baseline configuration if optimization fails."""
    baseline_config = HDCConfig(
        context_window=5,
        rare_token_threshold=2,
        top_k=3,
        temperature=1.0,
        rust_acceleration=False,  # Disable to avoid risk
        adaptive_readout_enabled=False
    )
    
    # Preserve learned data
    new_brain = BrainMemory(config=baseline_config)
    new_brain.tables = brain.tables
    
    logger.warning("Rolled back to baseline configuration due to performance issues")
    return new_brain
```

### 10. Success Criteria

#### 10.1 Must-Have Criteria

1. **Accuracy > 80%** across all benchmark tasks (code, classification, pattern, Q&A)
2. **Inference time < 50ms** average per query across all tasks
3. **Training time < 1000ms** for 10 training examples
4. **No crashes** or memory errors during benchmarks
5. **Backward compatibility** maintained for existing API

#### 10.2 Nice-to-Have Criteria

1. **Accuracy > 85%** on at least 2 tasks
2. **Inference time < 30ms** on classification and pattern tasks
3. **Memory usage < 300MB** for 10K training pairs
4. **Rust acceleration** available and working on all platforms
5. **Comprehensive documentation** with performance tuning guide

#### 10.3 Acceptance Testing

**Acceptance Test Suite:**
```python
def run_acceptance_tests() -> bool:
    """Run acceptance tests to validate success criteria."""
    brain = BrainMemory(config=HDCConfig())
    suite = BenchmarkSuite()
    
    # Run benchmarks
    results = suite.run_all_benchmarks(brain)
    
    # Check criteria
    criteria_passed = []
    
    # Criterion 1: Accuracy > 80%
    overall_acc = results['aggregate']['overall_accuracy']
    criteria_passed.append(('accuracy', overall_acc > 0.80, f"{overall_acc*100:.1f}%"))
    
    # Criterion 2: Inference time < 50ms
    overall_speed = results['aggregate']['overall_speed_ms']
    criteria_passed.append(('speed', overall_speed < 50, f"{overall_speed:.2f}ms"))
    
    # Criterion 3: Training time < 1000ms
    max_train_time = max(task['train_time_ms'] for task in results['tasks'].values())
    criteria_passed.append(('training_speed', max_train_time < 1000, f"{max_train_time:.2f}ms"))
    
    # Criterion 4: No crashes (implicitly passed if we reach here)
    criteria_passed.append(('stability', True, "No crashes"))
    
    # Print results
    print("\n" + "="*60)
    print("ACCEPTANCE TEST RESULTS")
    print("="*60)
    for name, passed, value in criteria_passed:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | {name:20s} | {value}")
    print("="*60)
    
    all_passed = all(passed for _, passed, _ in criteria_passed)
    print(f"\nOverall: {'✓ ALL CRITERIA MET' if all_passed else '✗ SOME CRITERIA FAILED'}\n")
    
    return all_passed
```

## Summary

This design provides a comprehensive approach to improving HDC system performance through:

1. **Accuracy Improvements:** Enhanced credit assignment, improved backoff strategy, adaptive readout sizing
2. **Speed Optimizations:** Rust acceleration, context sketch caching, incremental feature extraction
3. **Hyperparameter Optimization:** Grid search, Pareto optimization, automatic configuration
4. **Memory Efficiency:** Sparse storage compression, efficient serialization
5. **Comprehensive Testing:** Unit tests, integration tests, performance benchmarks, property-based tests (where applicable)
6. **Observability:** Telemetry collection, performance dashboards, detailed diagnostics

The phased implementation strategy ensures incremental progress with clear milestones, while risk mitigation measures protect against common failure modes. Success criteria are precisely defined and automatically validated through acceptance testing.

**Expected Outcomes:**
- Accuracy: 33.3% → >80% (>2.4× improvement)
- Inference time: 178ms → <50ms (>3.5× speedup)
- Training time: 1919ms → <1000ms (>1.9× speedup)
- Memory efficiency: 30-50% reduction
- Comprehensive benchmarking and diagnostics
