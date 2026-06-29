# PUHL-LUCK Brain Memory

**P50: Sparse Autoregressive Next-Token Prediction with Hyperdimensional Memory**

## Requirements

- **Python**: 3.9+ (tested on 3.14)
- **Rust** (optional, for performance): 
  - rustc 1.70+
  - cargo 1.70+
- **Python packages**: numpy

## Installation

From the repository root:

```bash
python -m pip install -e packages/puhl_luck
```

### Optional: Build Rust acceleration

For 10-20× speedup on transition search and operator induction:

```bash
pip install maturin
cd packages/puhl_luck
python -m maturin build --release
pip install rust_core/target/wheels/*.whl
```

**Note**: Rust is optional. The system works with Python fallback if Rust is not available.

## Architecture

### P50: Sparse Autoregressive Generator

**Core Innovation**: True next-token prediction using sparse transition tables and context sketching.

#### Generation Pipeline

```
Query → Tokenize → [BOS] input [SEP]
  ↓
Autoregressive Loop:
  ├─ Context Sketch: hash(last K tokens + field features)
  ├─ Sparse Table Query: sketch → {token: count}
  ├─ Backoff Strategy: K → K/2 → K/4 → unigram → field_only
  ├─ Copy Gate: extract rare tokens from input
  ├─ Select Next Token: weighted by counts
  └─ Update State: append token, repeat
  ↓
Until [EOS] or max_tokens
  ↓
Detokenize → Output
```

#### Learning Pipeline

```
expose_pair(input, target)
  ↓
Sequence: [BOS] + input_tokens + [SEP] + target_tokens + [EOS]
  ↓
For each position i:
  ├─ context = tokens[:i]
  ├─ next_token = tokens[i]
  ├─ sketch = BLAKE2b_128bit(context + field + position)
  └─ sparse_table[sketch][next_token] += 1
```

### Key Components

1. **SparseNextTokenTable** — Context sketch → token distribution storage
2. **ContextSketch** — 128-bit BLAKE2b hashing with position encoding and 5-level backoff
3. **CopyGate** — Extracts and scores rare tokens from input for direct copying
4. **AutoregressiveLoop** — Token-by-token generation with state updates
5. **SparseAutoregressiveGenerator** — Main orchestrator integrating all components

### Configuration (Tuned Parameters)

```python
SparseGeneratorConfig(
    context_k=5,              # Context window size (tuned)
    rare_token_threshold=2,   # Copy gate threshold (tuned)
    max_tokens=512,          # Maximum generation length
    top_k=3                  # Candidate pool size (tuned)
)
```

**Tuning Results**: Optimal parameters found via grid search on code generation tasks.

### Tokenization

**Punctuation-Preserving Tokenizer**:
```python
Input:  "def add(a, b):"
Tokens: ["def", "add", "(", "a", ",", "b", ")", ":"]
```

This preserves code syntax structure for better pattern learning.

### 9-Layer Hyperdimensional Memory (Foundation)

The sparse autoregressive generator is built on top of HDC-based memory:

1. **Exposure Events Memory** — stores observations
2. **Cognitive Field Formation** — activates relevant memories
3. **Operator Memory Layer** — learned transformation patterns (for metrics only)
4. **Transition Memory Layer** — partial → complete state transitions ⚡ (Rust accelerated)
5. **Candidate Emergence** — generates tension-reducing candidates
6. **Free-Energy Field Scoring** — scores candidates by energy reduction
7. **Recursive Stabilization** — iterative field updates
8. **Surface Realization** — converts states to text/code
9. **Feedback Correction** — learns from errors

**Rust Acceleration** (optional):
- Transition search: `find_transitions_by_features_rust()` ⚡
- Operator clustering: `greedy_cluster_transitions_rust()` ⚡
- Batch operations: feature extraction, similarity ⚡

## Usage

### Training saves learned files separately from code:

```text
brain_data/
  brain_memory.pkl
  brain_rank_micro.pmr
  brain_meta.json
  interaction_log.jsonl
```

### Common commands:

```bash
puhl-luck                    # Interactive mode
puhl-luck brain_data         # Load from directory
puhl-luck t data             # Train mode
puhl-luck s data             # Store mode
puhl-luck c --no-learn       # Chat without learning
puhl-luck q "question" "A,B,C,D"  # Question answering
puhl-luck b ask.tsv --quiet  # Batch processing
puhl-luck recall "query"     # Memory recall
puhl-luck status             # System status
```

## API Example

### Basic Usage

```python
from puhl_luck import BrainMemory

# Initialize
brain = BrainMemory()

# Learn from (input, target) pairs
brain.expose_pair(
    partial="def add(a, b):",
    complete="return a + b",
    domain="code"
)

# Generate code
output = brain.generate(
    "def multiply(x, y):",
    max_new_tokens=20,
    domain="code"
)
print(output)  # "return x * y"
```

### Advanced: Metrics and Monitoring

```python
# Generate with metrics
output, metrics = brain.generate(
    "def subtract(a, b):",
    max_new_tokens=20,
    domain="code",
    return_metrics=True
)

print(f"Generated: {output}")
print(f"Method: {metrics.generation_method}")  # "sparse_autoregressive"
print(f"Tokens: {metrics.tokens_generated}")
print(f"Backoff levels: {metrics.backoff_levels}")
print(f"Copy activations: {metrics.copy_gate_activations}")
print(f"Empty output: {metrics.empty_output}")
```

### Domain Support

P50 supports multiple domains with unified generation:

```python
# Code generation
brain.expose_pair("def func():", "pass", domain="code")

# Classification
brain.expose_pair("Text: positive review", "Label: positive", domain="classification")

# Question answering
brain.expose_pair("Q: What is Python?", "A: A programming language", domain="qa")
```

### Statistics

```python
# Get generator statistics
stats = brain._sparse_generator.get_statistics()
print(f"Pairs learned: {stats['pairs_learned']}")
print(f"Total transitions: {stats['total_transitions']}")
print(f"Unique contexts: {stats['total_contexts']}")
print(f"Unique tokens: {stats['total_unique_tokens']}")
```

## Testing

```bash
cd packages/puhl_luck
python -m pytest tests/ -v
```

**Test coverage**:
- ✅ 32 sparse table tests
- ✅ 47 context sketch tests
- ✅ 33 copy gate tests
- ✅ 22 autoregressive loop tests
- ✅ 25 sparse generator e2e tests
- ✅ 19 integration tests
- ✅ 8 brain integration tests
- ✅ 11 benchmark tests
- ✅ 313 total tests passing

## Benchmarks

### Code Generation (MBPP-style)

```bash
python benchmarks/mbpp_bench.py
```

### Code Generation (HumanEval-style)

```bash
python benchmarks/humaneval_bench.py
```

### Parameter Tuning

```bash
python benchmarks/tune_parameters.py
```

## Migration from P48

### What Changed

**P48 (Operator-based pattern retrieval)**:
```python
Query → Field signature → Operator graph selection → Template → Output
Problem: Returns empty when no operator match found
```

**P50 (Sparse autoregressive generation)**:
```python
Query → Sparse table → Next token prediction → Autoregressive loop → Output
Benefit: Generates based on learned token transitions, not pattern retrieval
```

### Breaking Changes

1. **No more `use_sparse_generator` parameter** - Sparse is now the only path
2. **Tokenization changed** - Now preserves punctuation/operators as separate tokens
3. **Metrics updated** - New fields: `backoff_levels`, `copy_gate_activations`, `empty_output`

### Migration Steps

```python
# P48 code (still works)
brain.expose_pair(input, output, domain="code")
result = brain.generate(query, domain="code")

# P50 code (same API, different implementation)
brain.expose_pair(input, output, domain="code")  # Now learns sparse transitions
result = brain.generate(query, domain="code")    # Now uses sparse autoregressive

# Access new metrics
result, metrics = brain.generate(query, return_metrics=True)
print(metrics.generation_method)  # "sparse_autoregressive"
```

## Performance

### Tuned Configuration

After parameter tuning on code generation tasks:
- Context window: K=5 (optimal for balance)
- Rare token threshold: 2 (optimal for copy gate)
- Top-K candidates: 3 (optimal for diversity)

### Expected Characteristics

- **Empty output rate**: <10% with sufficient training
- **Copy rate**: 0% (generates, doesn't copy)
- **Backoff usage**: Progressive degradation when exact match not found
- **Generation speed**: O(K × V) per token where K=context size, V=vocabulary size

## Paper

Original HDC research: https://zenodo.org/records/20851529

Sparse autoregressive generator: P50 implementation (this package)

## Version Info

- **Package**: 0.1.0 (P50)
- **Python**: >=3.9 (tested: 3.14)
- **Rust**: 2021 edition (optional)
- **pyo3**: 0.23
- **numpy**: 2.5.0
- **ndarray**: 0.15
- **rayon**: 1.10

## Architecture Comparison

| Feature | P48 (Operator) | P50 (Sparse) |
|---------|---------------|--------------|
| **Generation** | Pattern retrieval | Next-token prediction |
| **Learning** | Operator graphs | Token transitions |
| **Empty outputs** | High (no match) | Low (backoff) |
| **Copy rate** | Variable | 0% (generates) |
| **Domains** | Code-focused | Universal (code/text/QA) |
| **Tokenization** | Whitespace split | Punctuation-preserving |

See the repository root `README.md` for the full user guide.
