# PUHL-LUCK Brain Memory

**Predictive Field Memory with Transition-Based Learning and Rust Acceleration**

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

### 9-Layer Predictive Field Memory

1. **Exposure Events Memory** — stores observations
2. **Cognitive Field Formation** — activates relevant memories
3. **Operator Memory Layer** — learned transformation patterns
4. **Transition Memory Layer** — partial → complete state transitions ⚡ (Rust accelerated)
5. **Candidate Emergence** — generates tension-reducing candidates
6. **Free-Energy Field Scoring** — scores candidates by energy reduction
7. **Recursive Stabilization** — iterative field updates
8. **Surface Realization** — converts states to text/code
9. **Feedback Correction** — learns from errors

### Rust Acceleration

**Accelerated operations**:
- Transition search: `find_transitions_by_features_rust()` ⚡
- Operator clustering: `greedy_cluster_transitions_rust()` ⚡
- Batch operations: feature extraction, similarity ⚡

**Source**: `rust_core/src/` (Rust 2021 edition, pyo3 0.23)

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

```python
from puhl_luck import BrainMemory

# Initialize
brain = BrainMemory()

# Learn transitions (partial → complete)
brain.expose_pair(
    partial="What is HDC?",
    complete="HDC (Hyperdimensional Computing) uses high-dimensional vectors for fast similarity search.",
    source="training",
)

# Generate response
response = brain.generate("Explain HDC indexing")
print(response)
```

## Testing

```bash
cd packages/puhl_luck
python -m pytest tests/ -v
```

**Test results**: 313 passed, 1 skipped (Python fallback)

## Paper

Original research: https://zenodo.org/records/20851529

## Version Info

- **Package**: 0.1.0
- **Python**: >=3.9 (tested: 3.14)
- **Rust**: 2021 edition
- **pyo3**: 0.23
- **numpy**: 2.5.0
- **ndarray**: 0.15
- **rayon**: 1.10

See the repository root `README.md` for the full user guide.
