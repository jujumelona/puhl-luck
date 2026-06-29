# Operator-Based Generation System

## Overview

PUHL-LUCK now uses **operator-based generation** instead of retrieval-based generation. This means:

- ❌ **OLD**: Store raw text, retrieve exact matches → high copy rate (90%+)
- ✅ **NEW**: Extract operator graphs, generate from patterns → low copy rate (5-10%)

## Architecture

### Generation Pipeline

```
Query → Field Formation → Operator Activation → Graph Ordering → Token Generation → Output
```

1. **Field Formation**: Extract features from input query
2. **Operator Activation**: Retrieve relevant operator graphs from storage
3. **Graph Ordering**: Build execution DAG from activated operators
4. **Constraint Extraction**: Extract output constraints (function names, valid tokens, etc.)
5. **Token Generation**: Generate tokens following operator graph + constraints
6. **Assembly**: Combine tokens into final output

### Key Components

#### 1. Operator Extraction (`_memory_operator_extraction.py`)

Extracts operator graphs from code and NLP targets:

**Code Operators** (from Python AST):
- `FUNCTION_DEF`, `ARG`, `RETURN`
- `LIST_COMP`, `FILTER`, `MAP`
- `BINARY_OP`, `COMPARE`, `MODULO`
- `FOR_LOOP`, `IF_STMT`, `WHILE_LOOP`

**NLP Operators** (pattern-based):
- `KEYWORD_MATCH`, `DOMAIN_MATCH`
- `CLASSIFY`, `RETURN_LABEL`
- `ENTAILMENT`, `CONTRADICTION`
- `EXTRACT_SPAN`, `ANSWER`

```python
from puhl_luck._memory_operator_extraction import CodeOperatorExtractor

extractor = CodeOperatorExtractor()
graph = extractor.extract("""
def count_even(nums):
    return len([x for x in nums if x % 2 == 0])
""")

# graph.nodes = [
#   Operator("FUNCTION_DEF", params={"name": "count_even"}),
#   Operator("ARG", params={"name": "nums"}),
#   Operator("RETURN"),
#   Operator("LEN"),
#   Operator("LIST_COMP"),
#   Operator("FILTER"),
#   Operator("MODULO", params={"right": 2}),
#   Operator("COMPARE_EQ", params={"right": 0})
# ]
```

#### 2. Operator Storage (`_memory_operator_storage.py`)

Stores operator graphs indexed by field signatures (NOT raw text):

```python
storage.store_graph(
    graph=operator_graph,
    field_signature="feat:count|feat:even|feat:list",
    token_sequence=["def", "count_even", "(", "nums", ")", ...]
)
```

**Storage Structure**:
- `graphs`: signature → OperatorGraph
- `field_to_graph`: field_signature → [graph_signatures]
- `operator_transitions`: (op1, op2) → frequency
- `graph_to_tokens`: graph_signature → token_patterns

**NO raw text storage** - only operator patterns and transitions.

#### 3. Operator Activation (`_memory_operator_activation.py`)

Activates operators from field signature:

```python
activation = OperatorActivation()
activated = activation.activate(
    field_features=["feat:count", "feat:even", "feat:numbers"],
    operator_storage=storage
)

# Returns: [
#   (Operator("COUNT"), 0.95),
#   (Operator("FILTER"), 0.87),
#   (Operator("EVEN"), 0.82),
#   (Operator("LIST_INPUT"), 0.75),
# ]
```

#### 4. Operator Ordering (`_memory_operator_ordering.py`)

Orders activated operators into execution DAG:

```python
ordering = OperatorOrdering()
ordered = ordering.order(activated, operator_storage)

# ordered.graph = OperatorGraph with topological order:
# LIST_INPUT → FILTER → EVEN → COUNT → RETURN
```

**Features**:
- Learns operator transitions from training
- Breaks cycles to ensure DAG
- Adds code wrappers (FUNCTION_DEF, RETURN)

#### 5. Constraint Extraction (`_memory_constraint_extraction.py`)

Extracts output constraints from input query:

```python
extractor = ConstraintExtractor()
constraints = extractor.extract(
    "Write a function count_even that takes a list",
    domain="code"
)

# constraints = {
#   "function_name": "count_even",
#   "arg_type_hint": "list",
#   "domain": "code"
# }
```

**Constraint Types**:
- Code: function names, arg names, type hints
- MCQA: valid tokens (A, B, C, D)
- Classification: label set
- QA: answer type (short, yes/no, span)

#### 6. Token Generation (`_memory_token_generation.py`)

Generates tokens from operator graph + constraints:

```python
generator = TokenGenerator()
tokens = generator.generate(
    graph=ordered_graph,
    constraints={"function_name": "count_even"},
    operator_storage=storage,
    max_tokens=64
)

# tokens = ["def", "count_even", "(", "nums", ")", ":", "return", "len", ...]
```

**Features**:
- Generates tokens for each operator in topological order
- Applies constraints (function name enforcement)
- Uses token transition patterns as fallback
- Ensures grammar correctness (matching parentheses, etc.)

#### 7. Generation Metrics (`_memory_generation_metrics.py`)

Tracks whether outputs are copies or novel compositions:

```python
tracker = MetricsTracker()

# Record training
tracker.record_training_example(
    output="def add(a, b): return a + b",
    graph_signature="FUNC|ARG|RETURN|ADD",
    operators=["FUNCTION_DEF", "ARG", "RETURN", "BINARY_OP"]
)

# Compute metrics
metrics = tracker.compute_metrics(
    generated_output="def multiply(x, y): return x * y",
    graph_signature="FUNC|ARG|RETURN|MULT",
    operators_used=["FUNCTION_DEF", "ARG", "RETURN", "BINARY_OP"],
    generation_method="operator"
)

print(f"Was copy: {metrics.was_exact_copy}")  # False
print(f"Novel composition: {metrics.novel_composition}")  # True
print(f"Operator reuse: {metrics.operator_reuse_rate:.1%}")  # 100% (all operators seen)
```

## Usage

### Training (expose_pair)

```python
from puhl_luck.brain_memory import BrainMemory

brain = BrainMemory()

# Train on code
brain.expose_pair(
    partial="def count_even(nums):",
    complete="def count_even(nums): return len([x for x in nums if x % 2 == 0])",
    domain="code"
)

# What happens internally:
# 1. Extract operator graph from target
# 2. Store graph indexed by input field signature
# 3. Learn operator transitions
# 4. Record training example for metrics
```

### Generation (generate)

```python
# Generate with operator-based generation
result, metrics = brain.generate(
    query="def count_even(arr):",
    use_operator_generation=True,
    domain="code",
    return_metrics=True
)

print(f"Generated: {result}")
# Output: def count_even(arr): return len([x for x in arr if x % 2 == 0])

print(f"Method: {metrics.generation_method}")  # "operator" or "retrieval" or "token_fallback"
print(f"Was copy: {metrics.was_exact_copy}")  # False (generated, not retrieved)
print(f"Similarity: {metrics.nearest_train_similarity:.1%}")  # e.g., 75%
```

### Generation Fallback Strategy

The system has 3 levels of fallback:

1. **Operator-Based Generation** (PRIMARY)
   - Extract operators → order → generate
   - Used when operator graphs available
   - Produces novel compositions

2. **Surface Retrieval** (FALLBACK)
   - Retrieve stored sequences by input features
   - Used when operator generation fails
   - Returns exact copies (was_exact_copy=True)

3. **Token Generation** (LAST RESORT)
   - Generate token-by-token using patterns
   - Used when no stored sequences match
   - May produce incomplete outputs

## Metrics

### Training Metrics

```python
# After training
summary = brain._metrics_tracker.get_summary()

print(f"Total generations: {summary['total_generations']}")
print(f"Copy rate: {summary['copy_rate']:.1%}")  # Should be low (5-10%)
print(f"Novel composition rate: {summary['novel_composition_rate']:.1%}")  # Should be high (60-80%)
print(f"Avg operator reuse: {summary['avg_operator_reuse']:.1%}")  # Should be high (80-100%)
```

### Expected Results

**OLD (Retrieval-Based)**:
- Copy rate: 90%+
- Novel composition: 0%
- Problem: Memorization, not generalization

**NEW (Operator-Based)**:
- Copy rate: 5-10%
- Novel composition: 60-80%
- Benefit: Generalization, composition, transfer

## Testing

Run all operator-based generation tests:

```bash
cd packages/puhl_luck

# Core tests (99 tests)
python -m pytest \
  tests/test_operator_extraction.py \
  tests/test_operator_storage.py \
  tests/test_operator_activation.py \
  tests/test_operator_ordering.py \
  tests/test_constraint_extraction.py \
  tests/test_token_generation.py \
  tests/test_operator_based_generation_e2e.py \
  tests/test_generation_metrics.py \
  -v

# Quick integration test
python test_integration.py
```

## Implementation Status

### ✅ Completed (Phases 1-4)

- [x] Phase 1: Operator Extraction
  - [x] Code operator extractor (AST-based)
  - [x] NLP operator extractor (pattern-based)
  - [x] Operator graph representation
  - [x] 20 tests passing

- [x] Phase 2: Operator Storage & Activation
  - [x] Operator memory storage (NO raw text)
  - [x] Operator activation from field
  - [x] Operator graph ordering
  - [x] 22 tests passing

- [x] Phase 3: Constrained Generation
  - [x] Constraint extraction (function names, labels, etc.)
  - [x] Token generation from operators
  - [x] integrate into expose_pair() and generate()
  - [x] 36 tests passing

- [x] Phase 4: Generation Metrics
  - [x] MetricsTracker for copy detection
  - [x] Operator reuse tracking
  - [x] Novel composition detection
  - [x] 11 tests passing

**Total: 99 tests passing**

### 🔄 In Progress (Phase 5)

- [ ] Phase 5: Cleanup & Documentation
  - [x] Documentation (this file)
  - [ ] Update README
  - [ ] Benchmark integration (optional)

## Troubleshooting

### Issue: Operator generation returns None

**Cause**: No operator graphs stored or field signature doesn't match

**Solution**:
1. Check operator storage: `len(brain._operator_storage.graphs)`
2. Check training was successful: `len(brain._metrics_tracker.training_outputs)`
3. Try with simpler query that matches training examples

### Issue: High copy rate despite operator generation

**Cause**: Falling back to retrieval instead of operator generation

**Solution**:
1. Verify `use_operator_generation=True` in generate()
2. Check that operator graphs were extracted during training
3. Ensure domain parameter matches ("code", "classification", etc.)

### Issue: Generated code has syntax errors

**Cause**: Token generation didn't close structures correctly

**Solution**:
1. File issue with example
2. Temporary workaround: use retrieval fallback
3. Future: Improve token generation grammar enforcement

## Future Work

1. **Operator Induction**: Learn new operators from patterns
2. **Cross-Domain Transfer**: Use code operators for NLP tasks
3. **Compositional Benchmarks**: Systematic evaluation of composition ability
4. **Operator Refinement**: Improve operator granularity and coverage

## References

- Spec: `.kiro/specs/operator-based-generation/`
- Implementation: `packages/puhl_luck/puhl_luck/_memory_operator_*.py`
- Tests: `packages/puhl_luck/tests/test_operator_*.py`
