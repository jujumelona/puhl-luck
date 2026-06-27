# Design Document: Predictive Field Memory

## Overview

### System Purpose

The Predictive Field Memory redesigns PUHL from a retrieval-then-rank memory system into a **cognitive field-based generative system**. Instead of searching for events and scoring candidates, the system forms **simultaneous activation fields** where memories, concepts, and operators interact dynamically. Outputs emerge from field tension rather than selection from lists.

### Core Paradigm Shift

**Original PUHL (Retrieval-Based)**:
```
Query → Retrieve Events → Score Candidates → Select Best → Output
```

**Predictive Field Memory (Field-Based)**:
```
Input → Field Formation (Simultaneous Activation) → 
Candidate Emergence (from tension) → Field Update → 
Recursive Stabilization (until convergence)
```

### Key Architectural Principles

1. **Simultaneous Activation**: Input activates multiple memory layers at once, creating an interactive field
2. **Emergence over Selection**: Candidates emerge from field dynamics rather than being retrieved and ranked
3. **Tension-Driven Generation**: High field energy (incompleteness, conflicts) drives output generation
4. **Recursive Stabilization**: Field updates iteratively until reaching low-energy stable state
5. **Universal State Completion**: Same algorithm applies to conversation, code, documents, reasoning

### Design Philosophy

The system treats generation as **state field stabilization** guided by free energy minimization. The cognitive field contains simultaneous activations that interact through resonance and dissonance. Output generation reduces field tension by resolving conflicts, satisfying goals, and completing partial states.

This transforms PUHL from a specialized memory retrieval tool into a **general-purpose cognitive architecture** based on predictive processing and active inference principles.

## Architecture

### Four-Layer Memory Organization

The system organizes memory into four distinct functional layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    Input (Query/Context)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │   Field Formation Process        │
        │  (Simultaneous Activation)       │
        └──────────┬──────────────────────┘
                   │
    ┌──────────────┼──────────────┬─────────────┐
    ▼              ▼              ▼             ▼
┌────────┐   ┌────────┐    ┌──────────┐  ┌──────────┐
│Layer 1 │   │Layer 2 │    │ Layer 3  │  │ Layer 4  │
│Exposure│──▶│ State  │◀──▶│ Operator │  │Transition│
│ Events │   │ Field  │    │  Memory  │  │  Memory  │
└────────┘   └───┬────┘    └──────────┘  └──────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ Candidate Emergence │
        │ (Tension Reduction) │
        └────────┬────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  Output Generation  │
        └────────┬────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │   Field Update &    │
        │Recursive Stabilize  │
        └─────────────────────┘
```

### Layer Descriptions

**Layer 1: Exposure Events Layer**
- Preserves current PUHL event storage structure
- Stores raw observational data: events, co-activation edges, HDC features
- Maintains backward compatibility with existing brain_memory.pkl files
- Functions as the episodic memory foundation

**Layer 2: State Field Layer**
- The active cognitive field representing current system state
- Contains:
  - Query feature representations
  - Activated events (with activation strengths)
  - Activated concepts (with activation strengths)
  - Activated operators (with activation strengths)
  - Conflict markers (with conflict strengths)
  - Goal states (with satisfaction levels)
  - Partial output representations
- Computes field energy from component interactions
- Supports resonance between simultaneously activated memories

**Layer 3: Operator Memory Layer**
- Stores learned state transformation patterns
- Operator types:
  - `completion_operator`: incomplete → complete
  - `repair_operator`: error → correction
  - `explanation_operator`: question → answer
  - `comparison_operator`: entity → comparison
  - `transformation_operator`: problem → solution
  - `composition_operator`: part → whole
- Enables generalization beyond specific event examples
- Learned through operator induction from repeated patterns

**Layer 4: Transition Memory Layer**
- Stores partial-to-complete state transitions (S_partial → S_complete)
- NOT before/after pairs, but incomplete/complete pairs
- Examples:
  - Conversation: incomplete_context → completed_context
  - Code: incomplete_code → completed_code
  - Reasoning: incomplete_reasoning → completed_reasoning
- Supports learning of completion dynamics

### Information Flow

1. **Input Reception**: Query/context arrives
2. **Field Formation**: Simultaneous activation of all layers
3. **Resonance Computation**: Activated memories interact
4. **Energy Calculation**: Field energy = conflicts - evidence
5. **Candidate Emergence**: Generate tension-reducing continuations
6. **Output Selection**: Choose energy-minimizing candidate
7. **Field Update**: Incorporate output into field state
8. **Convergence Check**: If stable, stop; else repeat from step 2

### Integration with Original PUHL

The design preserves successful PUHL components while adding field-based dynamics:

**Preserved Components**:
- Event storage structure (Layer 1)
- HDC-based feature extraction
- Co-activation edge tracking
- Micro-rank artifact functionality
- Novelty detection
- Targeted forgetting

**New Components**:
- State Field Layer (Layer 2)
- Operator Memory Layer (Layer 3)
- Transition Memory Layer (Layer 4)
- Field formation algorithm
- Candidate emergence from tension
- Recursive stabilization loop

## Components and Interfaces

### Core Components

#### 1. CognitiveField

Main orchestrator managing all four layers and the generation loop.

**Responsibilities**:
- Initialize and maintain all four memory layers
- Coordinate field formation process
- Execute recursive stabilization loop
- Provide backward-compatible API adapter

**Key Methods**:
```python
class CognitiveField:
    def __init__(self, window_size: int, decay: float)
    def form_field(self, input_context: InputContext) -> StateField
    def generate(self, context: str, max_iterations: int) -> GenerationResult
    def expose_text(self, text: str, metadata: Dict) -> None  # Backward compat
    def expose_file(self, path: str, metadata: Dict) -> None  # Backward compat
    def rank(self, query: str, candidates: List[str]) -> List[float]  # Backward compat
```

#### 2. ExposureEventsLayer

Layer 1: Stores raw observational data using current PUHL structure.

**Responsibilities**:
- Store events with HDC feature vectors
- Maintain co-activation edge graph
- Track feature frequencies and statistics
- Support HDC-based similarity queries

**Key Methods**:
```python
class ExposureEventsLayer:
    def store_event(self, event: EventRecord) -> str
    def get_event(self, event_id: str) -> EventRecord
    def find_similar_events(self, query_hv: np.ndarray, top_k: int) -> List[Tuple[str, float]]
    def get_coactivated_events(self, event_ids: List[str]) -> List[Tuple[str, float]]
    def compute_event_features(self, content: str, modality: str) -> List[str]
```

**Data Structures**:
- `events: Dict[str, EventRecord]` - Event ID → event data
- `edges: Dict[Tuple[int, int], float]` - Co-activation edges with weights
- `feature_to_events: Dict[int, Counter[str]]` - Feature → events containing it
- `event_hv: Dict[str, np.ndarray]` - Event ID → HDC hypervector

#### 3. StateFieldLayer

Layer 2: Represents the currently activated cognitive field.

**Responsibilities**:
- Maintain activation states for events, concepts, operators
- Compute field energy from component interactions
- Calculate resonance between activated memories
- Track conflicts, goals, and partial outputs

**Key Methods**:
```python
class StateFieldLayer:
    def activate_from_input(self, input_context: InputContext, 
                           events_layer: ExposureEventsLayer,
                           operators_layer: OperatorMemoryLayer) -> None
    def compute_field_energy(self) -> FieldEnergy
    def compute_resonance(self, mem1: str, mem2: str) -> float
    def add_conflict(self, conflict: ConflictMarker) -> None
    def add_goal(self, goal: GoalState) -> None
    def get_tension_sources(self) -> List[TensionSource]
    def update_with_output(self, output: str) -> None
```

**Data Structures**:
```python
@dataclass
class StateField:
    query_features: List[str]
    activated_events: Dict[str, float]  # event_id → activation_strength
    activated_concepts: Dict[str, float]  # concept_id → activation_strength
    activated_operators: Dict[str, float]  # operator_id → activation_strength
    conflict_markers: List[ConflictMarker]
    goal_states: List[GoalState]
    partial_outputs: List[str]
    
@dataclass
class FieldEnergy:
    total: float
    evidence: float
    conflicts: float
    evidence_breakdown: Dict[str, float]
    conflict_breakdown: Dict[str, float]
```

#### 4. OperatorMemoryLayer

Layer 3: Stores learned transformation patterns.

**Responsibilities**:
- Store operators as state transformation rules
- Match operators to current field state
- Instantiate operators with context-specific parameters
- Support operator induction from repeated patterns

**Key Methods**:
```python
class OperatorMemoryLayer:
    def store_operator(self, operator: OperatorRecord) -> str
    def find_applicable_operators(self, field_state: StateField) -> List[Tuple[str, float]]
    def instantiate_operator(self, operator_id: str, context: StateField) -> OperatorInstance
    def induce_operators(self, transition_history: List[StateTransition]) -> List[OperatorRecord]
```

**Data Structures**:
```python
@dataclass
class OperatorRecord:
    operator_id: str
    operator_type: OperatorType  # completion, repair, explanation, etc.
    pattern: StatePattern  # Abstract pattern matching structure
    transformation: TransformationRule
    confidence: float
    usage_count: int
    
class OperatorType(Enum):
    COMPLETION = "completion"
    REPAIR = "repair"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    TRANSFORMATION = "transformation"
    COMPOSITION = "composition"
```

#### 5. TransitionMemoryLayer

Layer 4: Stores partial-to-complete state transitions.

**Responsibilities**:
- Store S_partial → S_complete transitions
- Retrieve relevant transitions for current incomplete state
- Support learning of completion dynamics

**Key Methods**:
```python
class TransitionMemoryLayer:
    def store_transition(self, partial: StateField, complete: StateField) -> str
    def find_similar_transitions(self, current_partial: StateField, top_k: int) -> List[Tuple[str, float]]
    def get_completion_pattern(self, transition_id: str) -> CompletionPattern
```

**Data Structures**:
```python
@dataclass
class StateTransition:
    transition_id: str
    partial_state: StateField
    complete_state: StateField
    completion_vector: np.ndarray  # HDC representation of completion
    modality: str
    timestamp: float
```

#### 6. FieldFormation

Process that creates the cognitive field from input.

**Responsibilities**:
- Activate relevant events from Layer 1
- Activate relevant concepts from concept memory
- Activate relevant operators from Layer 3
- Compute initial resonance patterns
- Initialize field state in Layer 2

**Key Methods**:
```python
class FieldFormation:
    def form_field(self, 
                   input_context: InputContext,
                   events_layer: ExposureEventsLayer,
                   operators_layer: OperatorMemoryLayer,
                   previous_field: Optional[StateField] = None) -> StateField
    
    def activate_events(self, query_features: List[str], 
                       events_layer: ExposureEventsLayer) -> Dict[str, float]
    
    def activate_concepts(self, query_features: List[str]) -> Dict[str, float]
    
    def activate_operators(self, field_state: StateField,
                          operators_layer: OperatorMemoryLayer) -> Dict[str, float]
    
    def compute_initial_resonance(self, field: StateField) -> None
```

#### 7. CandidateEmergence

Process that generates outputs from field tension.

**Responsibilities**:
- Identify tension sources in current field
- Generate continuations that reduce tension
- Apply activated operators to generate candidates
- Use transition memory to guide completion
- Compute energy reduction for each candidate

**Key Methods**:
```python
class CandidateEmergence:
    def generate_candidates(self,
                           field: StateField,
                           operators_layer: OperatorMemoryLayer,
                           transitions_layer: TransitionMemoryLayer,
                           num_candidates: int) -> List[Candidate]
    
    def identify_tension_sources(self, field: StateField) -> List[TensionSource]
    
    def generate_from_operators(self, field: StateField,
                                operators: List[OperatorInstance]) -> List[Candidate]
    
    def generate_from_transitions(self, field: StateField,
                                  transitions: List[StateTransition]) -> List[Candidate]
    
    def compute_energy_reduction(self, candidate: Candidate, 
                                field: StateField) -> float
```

**Data Structures**:
```python
@dataclass
class Candidate:
    content: str
    energy_reduction: float
    tension_addressed: List[TensionSource]
    source: CandidateSource  # operator, transition, or hybrid
    
@dataclass
class TensionSource:
    type: TensionType  # conflict, unsatisfied_goal, incomplete_output
    location: str  # Where in the field
    strength: float
    description: str
```

#### 8. RecursiveStabilization

Loop that iteratively updates field until convergence.

**Responsibilities**:
- Execute generation loop: form field → emerge candidates → select → update
- Monitor field energy across iterations
- Detect convergence and oscillation
- Apply damping if oscillation detected
- Enforce maximum iteration limits

**Key Methods**:
```python
class RecursiveStabilization:
    def stabilize(self,
                  initial_context: InputContext,
                  cognitive_field: CognitiveField,
                  max_iterations: int,
                  convergence_threshold: float) -> StabilizationResult
    
    def detect_convergence(self, energy_history: List[float]) -> bool
    
    def detect_oscillation(self, energy_history: List[float]) -> bool
    
    def apply_damping(self, field: StateField, damping_factor: float) -> StateField
```

**Data Structures**:
```python
@dataclass
class StabilizationResult:
    final_output: str
    iterations: int
    converged: bool
    final_energy: float
    energy_history: List[float]
    convergence_stats: ConvergenceStats
```

#### 9. FreeEnergyMinimization

Computes field energy and guides stabilization.

**Responsibilities**:
- Compute total field energy as conflicts - evidence
- Break down energy into components
- Compute evidence from resonance, goals, constraints
- Compute conflicts from contradictions, violations, repetitions
- Guide candidate selection toward energy reduction

**Key Methods**:
```python
class FreeEnergyMinimization:
    def compute_field_energy(self, field: StateField) -> FieldEnergy
    
    def compute_evidence(self, field: StateField) -> Tuple[float, Dict[str, float]]
    
    def compute_conflicts(self, field: StateField) -> Tuple[float, Dict[str, float]]
    
    def predict_energy_after_update(self, field: StateField, 
                                    candidate: Candidate) -> float
```

**Energy Components**:

**Evidence (decreases energy)**:
- Memory resonance strength: How well activated memories support each other
- Goal satisfaction: How well current state satisfies active goals
- Constraint satisfaction: How well current state satisfies constraints
- Memory support: How strongly memories support current state

**Conflicts (increases energy)**:
- Contradictions: Activated memories that conflict with each other
- Constraint violations: Violations of hard constraints
- Repetition patterns: Unwanted repetitive structures
- Incompleteness: Partial or missing information

#### 10. OperatorInduction

Learns operators from repeated patterns.

**Responsibilities**:
- Identify repeated transformation patterns across exposures
- Abstract patterns into operators
- Generalize operators across surface forms
- Store operators in Layer 3

**Key Methods**:
```python
class OperatorInduction:
    def induce_from_history(self,
                           transition_history: List[StateTransition],
                           min_pattern_count: int) -> List[OperatorRecord]
    
    def identify_repeated_patterns(self, 
                                  transitions: List[StateTransition]) -> List[PatternCluster]
    
    def abstract_pattern(self, transitions: List[StateTransition]) -> StatePattern
    
    def generalize_transformation(self, 
                                 transitions: List[StateTransition]) -> TransformationRule
```

### Interface Specifications

#### InputContext Interface

```python
@dataclass
class InputContext:
    text: str
    modality: str
    metadata: Dict[str, Any]
    goals: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
```

#### StateField Interface (Detailed)

```python
@dataclass
class ConflictMarker:
    conflict_id: str
    conflict_type: ConflictType
    involved_memories: List[str]
    strength: float
    description: str

@dataclass
class GoalState:
    goal_id: str
    goal_description: str
    satisfaction_level: float  # 0.0 to 1.0
    constraints: List[str]

class ConflictType(Enum):
    CONTRADICTION = "contradiction"
    CONSTRAINT_VIOLATION = "constraint_violation"
    REPETITION = "repetition"
```

#### Backward Compatibility Adapter

```python
class PUHLCompatibilityAdapter:
    """Translates Original PUHL API calls to field operations"""
    
    def __init__(self, cognitive_field: CognitiveField):
        self.cognitive_field = cognitive_field
    
    def expose_text(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Original PUHL expose_text API"""
        context = InputContext(text=text, modality="text", metadata=metadata or {})
        self.cognitive_field.events_layer.store_event_from_context(context)
    
    def expose_file(self, filepath: str, metadata: Optional[Dict] = None) -> None:
        """Original PUHL expose_file API"""
        # Reads file and stores in events layer
    
    def rank(self, query: str, candidates: List[str], mode: str = "event") -> List[float]:
        """Original PUHL rank API using field-based scoring"""
        field = self.cognitive_field.form_field(InputContext(text=query, modality="text", metadata={}))
        scores = []
        for candidate in candidates:
            # Score using field energy reduction
            candidate_field = self.cognitive_field.form_field(
                InputContext(text=candidate, modality="text", metadata={})
            )
            energy_reduction = self.compute_compatibility(field, candidate_field)
            scores.append(energy_reduction)
        return scores
```

## Data Models

### Core Data Structures

#### EventRecord (Preserved from Original PUHL)

```python
@dataclass
class EventRecord:
    event_id: str
    content: str
    modality: str
    labels: Set[str]
    features: List[str]
    hdc_vector: np.ndarray
    timestamp: float
    exposure_count: int
    novelty_score: float
    content_hash: str
```

#### StateField (Core to Layer 2)

```python
@dataclass
class StateField:
    # Input representation
    query_features: List[str]
    query_hv: np.ndarray
    
    # Activated memories with strengths
    activated_events: Dict[str, float]  # event_id → activation
    activated_concepts: Dict[str, float]  # concept_id → activation
    activated_operators: Dict[str, float]  # operator_id → activation
    
    # Field dynamics
    conflict_markers: List[ConflictMarker]
    goal_states: List[GoalState]
    partial_outputs: List[str]
    
    # Resonance matrix
    resonance: Dict[Tuple[str, str], float]  # (mem1, mem2) → resonance_strength
    
    # Energy state
    field_energy: Optional[FieldEnergy] = None
    
    # History for recursive updates
    previous_outputs: List[str] = field(default_factory=list)
    iteration: int = 0
```

#### OperatorRecord (Core to Layer 3)

```python
@dataclass
class OperatorRecord:
    operator_id: str
    operator_type: OperatorType
    
    # Pattern matching
    pattern: StatePattern  # What field states this applies to
    preconditions: List[str]  # Required field properties
    
    # Transformation
    transformation: TransformationRule  # How to generate output
    completion_template: str
    
    # Statistics
    confidence: float
    usage_count: int
    success_rate: float
    generalization_level: int  # How abstract the operator is
    
    # Provenance
    induced_from: List[str]  # Transition IDs this was learned from
    timestamp: float

@dataclass
class StatePattern:
    """Abstract pattern that matches field states"""
    required_features: Set[str]
    required_concepts: Set[str]
    incompleteness_markers: List[str]
    goal_patterns: List[str]
    
@dataclass
class TransformationRule:
    """How to transform matched state"""
    rule_type: str  # "template", "learned_transition", "feature_combination"
    parameters: Dict[str, Any]
    confidence_threshold: float
```

#### StateTransition (Core to Layer 4)

```python
@dataclass
class StateTransition:
    transition_id: str
    
    # The incomplete → complete state pair
    partial_state: StateField
    complete_state: StateField
    
    # Difference representation
    completion_vector: np.ndarray  # HDC vector of what was added
    completion_features: List[str]
    
    # Context
    modality: str
    domain: str  # conversation, code, document, reasoning
    
    # Statistics
    timestamp: float
    relevance_count: int  # How often this transition was useful
```

#### FieldEnergy (Core to Free Energy Minimization)

```python
@dataclass
class FieldEnergy:
    # Total energy (conflicts - evidence)
    total: float
    
    # Components
    evidence: float
    conflicts: float
    
    # Detailed breakdown
    evidence_breakdown: Dict[str, float]  # source → contribution
    conflict_breakdown: Dict[str, float]  # source → contribution
    
    # Interpretation
    dominant_evidence_sources: List[str]
    dominant_conflict_sources: List[str]
    tension_level: float  # Normalized 0-1

# Evidence sources
EVIDENCE_SOURCES = [
    "memory_resonance",      # Activated memories support each other
    "goal_satisfaction",     # Goals are being satisfied
    "constraint_satisfaction",  # Constraints are met
    "memory_support",        # Strong backing from memory
    "coherence",            # Internal consistency
]

# Conflict sources
CONFLICT_SOURCES = [
    "contradiction",         # Memories conflict
    "constraint_violation",  # Constraints violated
    "repetition",           # Unwanted repetition
    "incompleteness",       # Missing information
    "goal_conflict",        # Goals conflict with each other
]
```

#### Candidate (Generated Output Candidate)

```python
@dataclass
class Candidate:
    # Content
    content: str
    tokens: List[str]
    
    # Energy analysis
    energy_reduction: float
    predicted_energy_after: float
    
    # Source tracking
    source: CandidateSource
    source_operators: List[str]  # Which operators contributed
    source_transitions: List[str]  # Which transitions contributed
    
    # Tension resolution
    tensions_addressed: List[TensionSource]
    tensions_resolved_count: int
    
    # Confidence
    confidence: float

class CandidateSource(Enum):
    OPERATOR_BASED = "operator"
    TRANSITION_BASED = "transition"
    HYBRID = "hybrid"
    FIELD_DYNAMICS = "field_dynamics"
```

### Persistence Models

#### CognitiveFieldSnapshot (For Serialization)

```python
@dataclass
class CognitiveFieldSnapshot:
    version: str
    timestamp: float
    
    # Layer 1: Exposure Events (preserved PUHL structure)
    events: Dict[str, EventRecord]
    edges: Dict[Tuple[int, int], float]
    feature_to_id: Dict[str, int]
    id_to_feature: List[str]
    event_hv: Dict[str, np.ndarray]
    
    # Layer 3: Operator Memory
    operators: Dict[str, OperatorRecord]
    
    # Layer 4: Transition Memory
    transitions: Dict[str, StateTransition]
    
    # Statistics
    total_exposures: int
    total_operators_induced: int
    total_transitions_stored: int
    
    # Configuration
    window_size: int
    decay: float
    hdc_dimensions: int
```

### Database Schema (If Using External Storage)

For production deployments, the system may use a database for Layer 3 and Layer 4:

**Operators Table**:
```sql
CREATE TABLE operators (
    operator_id TEXT PRIMARY KEY,
    operator_type TEXT NOT NULL,
    pattern_json TEXT NOT NULL,
    transformation_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    usage_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    generalization_level INTEGER DEFAULT 0,
    induced_from TEXT,  -- JSON array of transition IDs
    created_at REAL NOT NULL
);

CREATE INDEX idx_operators_type ON operators(operator_type);
CREATE INDEX idx_operators_confidence ON operators(confidence DESC);
```

**Transitions Table**:
```sql
CREATE TABLE transitions (
    transition_id TEXT PRIMARY KEY,
    partial_state_json TEXT NOT NULL,
    complete_state_json TEXT NOT NULL,
    completion_vector BLOB NOT NULL,  -- Serialized numpy array
    modality TEXT NOT NULL,
    domain TEXT NOT NULL,
    relevance_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE INDEX idx_transitions_modality ON transitions(modality);
CREATE INDEX idx_transitions_domain ON transitions(domain);
CREATE INDEX idx_transitions_relevance ON transitions(relevance_count DESC);
```

### Memory Management

#### Event Pruning (Inherited from Original PUHL)

- Preserve targeted forgetting mechanism
- Prune low-activation events periodically
- Maintain novelty-based retention

#### Operator Pruning

- Remove operators with low confidence after sufficient trials
- Consolidate similar operators
- Retain high-generalization operators preferentially

#### Transition Pruning

- Remove transitions with low relevance_count
- Keep diverse transition examples
- Maintain representative samples per domain

### Data Flow Example

**Example: Answering a Question**

1. **Input**: "What is the capital of France?"
   ```python
   context = InputContext(
       text="What is the capital of France?",
       modality="text",
       metadata={},
       goals=["answer_question"]
   )
   ```

2. **Field Formation**:
   ```python
   field = StateField(
       query_features=["what", "capital", "france", "question"],
       activated_events={
           "event_123": 0.85,  # Contains "Paris is the capital of France"
           "event_456": 0.62,  # Contains "France is in Europe"
       },
       activated_concepts={
           "geography": 0.78,
           "capital_cities": 0.92
       },
       activated_operators={
           "explanation_operator_7": 0.88  # question → answer pattern
       },
       goal_states=[
           GoalState(goal_id="g1", goal_description="answer_question", 
                    satisfaction_level=0.0, constraints=[])
       ]
   )
   ```

3. **Energy Computation**:
   ```python
   energy = FieldEnergy(
       total=0.75,  # High tension (incomplete)
       evidence=0.25,  # Some relevant memories
       conflicts=0.0,  # No contradictions
       evidence_breakdown={
           "memory_resonance": 0.15,
           "memory_support": 0.10
       },
       conflict_breakdown={},
       tension_level=0.75
   )
   ```

4. **Candidate Emergence**:
   ```python
   candidates = [
       Candidate(
           content="Paris",
           energy_reduction=0.70,
           source=CandidateSource.OPERATOR_BASED,
           source_operators=["explanation_operator_7"],
           tensions_addressed=[
               TensionSource(type=TensionType.UNSATISFIED_GOAL, 
                           location="goal_g1", strength=1.0)
           ],
           confidence=0.92
       )
   ]
   ```

5. **Field Update**:
   ```python
   updated_field = StateField(
       query_features=["what", "capital", "france", "question"],
       activated_events={...},  # Same
       activated_concepts={...},  # Same
       activated_operators={...},  # Same
       goal_states=[
           GoalState(goal_id="g1", goal_description="answer_question",
                    satisfaction_level=0.92, constraints=[])
       ],
       partial_outputs=["Paris"],
       previous_outputs=["Paris"]
   )
   ```

6. **Convergence**: Field energy dropped significantly (0.75 → 0.05), converged.


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Transition Incompleteness-to-Completeness Semantics

*For any* state transition stored in the Transition_Memory_Layer, the partial state SHALL have measurably higher incompleteness score than the complete state.

**Validates: Requirements 1.8, 5.1**

### Property 2: Simultaneous Multi-Layer Activation

*For any* input context that has relevant memories in multiple layers (events, concepts, operators), the Field_Formation process SHALL activate memories from all applicable layers simultaneously (not sequentially), and the resulting StateField SHALL contain non-empty activation dictionaries for each applicable layer.

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 3: Activation Strength Assignment

*For any* memory activation operation, every activated memory (event, concept, or operator) SHALL have an associated activation strength value in the range [0, 1].

**Validates: Requirements 2.5**

### Property 4: Resonance Correlation with Feature Overlap

*For any* two simultaneously activated memories in the StateField, if they share feature overlap above a threshold, their resonance SHALL be positive; if they contain conflicting features (contradiction markers), their resonance SHALL be negative.

**Validates: Requirements 2.6, 14.1, 14.2, 14.3, 14.4**

### Property 5: High Incompleteness Implies High Energy

*For any* StateField with incompleteness markers (unsatisfied goals, conflicts, or missing information), the computed field energy SHALL exceed a baseline threshold.

**Validates: Requirements 3.2, 13.1**

### Property 6: Candidate Energy Reduction

*For any* high-tension StateField (energy above threshold), all generated candidates SHALL have negative energy_reduction values, meaning they reduce total field energy when applied.

**Validates: Requirements 3.3, 13.2**

### Property 7: Conflict Reduction Through Candidates

*For any* StateField containing conflict markers, the generated candidates SHALL address at least one conflict, and applying the selected candidate SHALL result in a StateField with reduced conflict count or reduced total conflict strength.

**Validates: Requirements 3.5, 13.3**

### Property 8: Evidence Accumulation Through Candidates

*For any* StateField, the generated candidates SHALL increase the evidence score component of field energy when applied.

**Validates: Requirements 3.6**

### Property 9: Goal Satisfaction Increase

*For any* StateField with unsatisfied goals (satisfaction_level < 1.0), the generated candidates SHALL increase at least one goal's satisfaction level.

**Validates: Requirements 3.7, 13.4**

### Property 10: Operator Induction from Repeated Patterns

*For any* set of state transitions that repeat a similar transformation structure at least N times (where N is the minimum pattern count threshold), the Operator_Induction process SHALL identify the pattern and create an operator of the appropriate type (completion, repair, explanation, comparison, transformation, or composition).

**Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6**

### Property 11: Operator Application to Novel Inputs

*For any* operator stored in Operator_Memory_Layer and any novel input (not in the operator's training set) that matches the operator's preconditions and pattern, the Candidate_Emergence process SHALL apply that operator to generate candidates.

**Validates: Requirements 4.7, 9.1, 9.2**

### Property 12: Operator Generalization Across Surface Forms

*For any* operator induced from a set of transitions, the operator SHALL successfully match and apply to inputs that have the same structural transformation pattern but different surface content (different words, different variable names, different entities) from the training examples.

**Validates: Requirements 4.8, 9.3, 9.4, 9.5, 9.6**

### Property 13: Universal Completion Algorithm

*For any* domain (conversation, code, documents, reasoning), the State_Completion process SHALL invoke the same core completion function/algorithm, differing only in surface representation (tokenization, syntax) but not in completion dynamics (field formation, energy minimization, candidate emergence).

**Validates: Requirements 5.6, 8.1, 8.2, 8.3, 8.4, 8.5**

### Property 14: Field Update After Output

*For any* output generated by Candidate_Emergence, the Recursive_Stabilization process SHALL update the StateField to include that output before the next iteration.

**Validates: Requirements 6.1**

### Property 15: Activation Recomputation After Field Update

*For any* StateField update (adding output, changing goals, resolving conflicts), the Field_Formation process SHALL recompute activation strengths for all layers, and at least one activation strength SHALL change from its previous value.

**Validates: Requirements 6.2**

### Property 16: Recursive Loop Sequence Ordering

*For any* execution of the Recursive_Stabilization process, the operations SHALL occur in the following order: field formation → candidate emergence → output selection → field update → convergence check, and this sequence SHALL repeat until convergence or maximum iterations.

**Validates: Requirements 6.4**

### Property 17: Convergence to Low Energy State

*For any* Recursive_Stabilization execution that converges (not hitting max iterations), the final StateField energy SHALL be below the convergence threshold.

**Validates: Requirements 6.5, 20.2**

### Property 18: Stability Detection from Energy Changes

*For any* Recursive_Stabilization execution where consecutive energy changes fall below the convergence threshold for K consecutive iterations (where K is the stability window), the process SHALL detect convergence and terminate.

**Validates: Requirements 6.6, 20.2**

### Property 19: Energy Formula Consistency

*For any* StateField, the computed total field energy SHALL equal (total_conflicts - total_evidence), where total_conflicts and total_evidence are the sums of their respective component breakdowns.

**Validates: Requirements 7.1, 17.1, 17.2**

### Property 20: Evidence from Memory Resonance

*For any* StateField where activated memories have positive resonance values (mutually supporting), the evidence_breakdown component "memory_resonance" SHALL be positive and SHALL increase monotonically with the sum of positive resonance values.

**Validates: Requirements 7.2, 17.3**

### Property 21: Evidence from Goal Satisfaction

*For any* StateField containing goals, the evidence_breakdown component "goal_satisfaction" SHALL increase monotonically with the average satisfaction_level of all goals in the field.

**Validates: Requirements 7.3, 17.4**

### Property 22: Conflict from Contradictions

*For any* StateField containing ConflictMarkers of type CONTRADICTION, the conflict_breakdown component "contradiction" SHALL be positive and SHALL increase monotonically with the count and strength of contradiction markers.

**Validates: Requirements 7.5, 17.6**

### Property 23: Conflict from Constraint Violations

*For any* StateField containing ConflictMarkers of type CONSTRAINT_VIOLATION, the conflict_breakdown component "constraint_violation" SHALL be positive.

**Validates: Requirements 7.6, 17.7**

### Property 24: Conflict from Repetition

*For any* StateField containing ConflictMarkers of type REPETITION, the conflict_breakdown component "repetition" SHALL be positive.

**Validates: Requirements 7.7, 17.8**

### Property 25: Oscillation Detection and Damping

*For any* Recursive_Stabilization execution where field energy alternates between two similar values for M consecutive iterations (where M is the oscillation detection window), the process SHALL detect oscillation and apply damping to reduce oscillation amplitude.

**Validates: Requirements 20.3, 20.4**

### Property 26: Maximum Iteration Termination

*For any* Recursive_Stabilization execution, if the maximum iteration limit is reached without convergence, the process SHALL terminate and return the lowest-energy StateField encountered during the execution.

**Validates: Requirements 20.1, 20.5**

### Property 27: Multi-Modal Field Representation

*For any* input context containing multiple modalities (text + code, text + structured data, etc.), the Field_Formation process SHALL activate memories from all present modalities simultaneously, and the resulting StateField SHALL contain activated_events from multiple modality types.

**Validates: Requirements 19.4, 19.5, 19.6**


## Error Handling

### Input Validation Errors

**Empty or Invalid Input Context**:
- Detection: Check if input text is empty or None
- Handling: Return error with message "Input context cannot be empty"
- Recovery: Prompt user for valid input

**Malformed Metadata**:
- Detection: Validate metadata dictionary structure
- Handling: Log warning, use default metadata values
- Recovery: Continue with validated metadata

**Unsupported Modality**:
- Detection: Check if modality is in supported list
- Handling: Log warning, default to "text" modality
- Recovery: Process as text with feature extraction

### Field Formation Errors

**No Memories Activated**:
- Detection: All activation dictionaries are empty after field formation
- Handling: Log warning, create minimal field with query features only
- Recovery: Proceed with field dynamics using query features

**HDC Vector Computation Failure**:
- Detection: Exception during hyperdimensional computing operations
- Handling: Fall back to feature-based similarity
- Recovery: Use string matching for activation

**Memory Corruption**:
- Detection: Event or operator record fails validation
- Handling: Skip corrupted memory, log error with memory ID
- Recovery: Continue with remaining valid memories

### Energy Computation Errors

**NaN or Infinite Energy Values**:
- Detection: Check for math.isnan() or math.isinf()
- Handling: Log error with field state details
- Recovery: Reset energy to default high value, continue

**Energy Component Mismatch**:
- Detection: Sum of breakdown components doesn't match total
- Handling: Log warning with discrepancy details
- Recovery: Use component sum as authoritative value

**Negative Evidence or Conflict Scores**:
- Detection: Check if evidence < 0 or conflicts < 0
- Handling: Clamp to zero, log warning
- Recovery: Continue with clamped values

### Candidate Generation Errors

**No Candidates Generated**:
- Detection: Candidate list is empty after emergence
- Handling: Generate fallback candidate from most activated operator
- Recovery: If still empty, return empty string with error flag

**All Candidates Increase Energy**:
- Detection: All energy_reduction values are positive
- Handling: Select candidate with smallest energy increase
- Recovery: Log warning about inability to reduce energy

**Candidate Generation Timeout**:
- Detection: Generation exceeds time limit
- Handling: Return partial candidates generated so far
- Recovery: Use best available candidate, log timeout

### Recursive Stabilization Errors

**Infinite Loop (Max Iterations Exceeded)**:
- Detection: Iteration count reaches maximum
- Handling: Return lowest-energy state encountered
- Recovery: Log convergence statistics, mark as non-converged

**Oscillation Without Convergence**:
- Detection: Energy oscillates for multiple cycles
- Handling: Apply damping, if still oscillating return average state
- Recovery: Log oscillation pattern for analysis

**Energy Divergence (Increasing Energy)**:
- Detection: Energy increases for K consecutive iterations
- Handling: Revert to previous lowest-energy state
- Recovery: Terminate stabilization, log divergence

**Stack Overflow in Recursive Updates**:
- Detection: RecursionError exception
- Handling: Convert to iterative loop if detected
- Recovery: Continue with iterative implementation

### Operator Induction Errors

**Insufficient Transitions for Induction**:
- Detection: Transition count below minimum threshold
- Handling: Skip induction, log warning
- Recovery: Continue without creating operator

**Pattern Extraction Failure**:
- Detection: Cannot identify common pattern structure
- Handling: Log failed transitions, skip operator creation
- Recovery: Continue processing other transition groups

**Operator Conflict (Duplicate Patterns)**:
- Detection: New operator pattern matches existing operator
- Handling: Merge operators, increase confidence of existing
- Recovery: Update existing operator statistics

### Persistence Errors

**File I/O Errors**:
- Detection: Exception during save/load operations
- Handling: Retry with exponential backoff (3 attempts)
- Recovery: If all retries fail, raise error to user

**Pickle Deserialization Errors**:
- Detection: Exception during unpickling
- Handling: Attempt to load with compatibility mode
- Recovery: If fails, create new memory from scratch

**Disk Space Exhaustion**:
- Detection: OSError with ENOSPC error code
- Handling: Trigger emergency pruning of low-value memories
- Recovery: Retry save after pruning

**Corrupted Memory File**:
- Detection: Checksum mismatch or malformed data
- Handling: Attempt partial recovery of valid sections
- Recovery: Rebuild indices from recovered data

### Backward Compatibility Errors

**Legacy API Call with New-Only Features**:
- Detection: Legacy expose_text call with field-specific metadata
- Handling: Ignore unsupported metadata, log warning
- Recovery: Process with supported subset

**Old Brain Memory File Format**:
- Detection: Missing new layer data in loaded file
- Handling: Initialize missing layers as empty
- Recovery: Migrate data to new format on save

**Type Mismatch in Loaded Data**:
- Detection: Data type doesn't match expected schema
- Handling: Attempt type coercion, fallback to default
- Recovery: Log conversion details, continue

### Resource Exhaustion

**Memory Overflow (Too Many Activations)**:
- Detection: Activation dictionary size exceeds limit
- Handling: Prune lowest-activation memories
- Recovery: Continue with top-K activations

**Computation Timeout**:
- Detection: Operation exceeds allocated time budget
- Handling: Return partial result with timeout flag
- Recovery: Log operation details for optimization

**GPU/Accelerator Errors**:
- Detection: CUDA errors or accelerator unavailability
- Handling: Fall back to CPU computation
- Recovery: Log performance impact, continue

### Monitoring and Logging

**Error Logging Strategy**:
- All errors logged with timestamp, error type, context
- Stack traces captured for unexpected exceptions
- Error rates tracked for anomaly detection

**Metrics Collection**:
- Track error frequency by type
- Monitor recovery success rates
- Alert on error rate spikes

**Debug Mode**:
- Enable verbose logging of field states
- Dump problematic memory contents
- Trace energy computation steps

## Testing Strategy

### Overview

The Predictive Field Memory system requires a comprehensive testing strategy that validates both the correctness of field-based dynamics and the preservation of Original PUHL functionality. Testing combines property-based testing for universal correctness properties, unit tests for specific components, integration tests for layer interactions, and performance benchmarks.

### Property-Based Testing

**Library Selection**: Use `Hypothesis` (Python) for property-based test generation.

**Configuration**:
- Minimum 100 iterations per property test
- Timeout: 60 seconds per property
- Verbosity: Full counterexample reporting
- Seed: Fixed seed for reproducibility

**Property Test Structure**:

Each correctness property from the design document maps to one property-based test:

```python
from hypothesis import given, strategies as st
import pytest

@given(
    partial_state=st.builds(generate_incomplete_state_field),
    complete_state=st.builds(generate_complete_state_field)
)
def test_property_1_transition_incompleteness_semantics(partial_state, complete_state):
    """
    Feature: predictive-field-memory, Property 1: Transition Incompleteness-to-Completeness Semantics
    
    For any state transition, partial state has higher incompleteness than complete state.
    """
    transition = StateTransition(
        transition_id=generate_id(),
        partial_state=partial_state,
        complete_state=complete_state,
        completion_vector=np.random.rand(8192),
        modality="text",
        timestamp=time.time()
    )
    
    partial_incompleteness = compute_incompleteness_score(partial_state)
    complete_incompleteness = compute_incompleteness_score(complete_state)
    
    assert partial_incompleteness > complete_incompleteness, \
        f"Partial incompleteness ({partial_incompleteness}) must exceed complete ({complete_incompleteness})"
```

**Custom Generators**:

```python
@st.composite
def generate_incomplete_state_field(draw):
    """Generate StateField with incompleteness markers"""
    return StateField(
        query_features=draw(st.lists(st.text(min_size=1), min_size=1, max_size=10)),
        query_hv=np.random.rand(8192),
        activated_events=draw(st.dictionaries(st.text(), st.floats(min_value=0, max_value=1))),
        activated_concepts=draw(st.dictionaries(st.text(), st.floats(min_value=0, max_value=1))),
        activated_operators={},
        conflict_markers=draw(st.lists(generate_conflict_marker(), min_size=1)),
        goal_states=draw(st.lists(generate_unsatisfied_goal(), min_size=1)),
        partial_outputs=[]
    )

@st.composite
def generate_conflict_marker(draw):
    return ConflictMarker(
        conflict_id=f"conflict_{draw(st.integers(min_value=0, max_value=1000))}",
        conflict_type=draw(st.sampled_from(list(ConflictType))),
        involved_memories=draw(st.lists(st.text(), min_size=2, max_size=5)),
        strength=draw(st.floats(min_value=0.1, max_value=1.0)),
        description="Generated conflict"
    )
```

**Property Test Tagging**:

Every property test includes a docstring comment with the format:
```python
"""
Feature: predictive-field-memory, Property {N}: {Property Title}

{Property statement from design document}
"""
```

### Unit Testing

**Component Tests**:

Each component has focused unit tests for specific behaviors:

**ExposureEventsLayer Tests**:
- Store and retrieve events
- HDC vector computation
- Feature extraction accuracy
- Co-activation edge tracking
- Similarity search correctness

**StateFieldLayer Tests**:
- Field initialization
- Activation strength management
- Resonance computation
- Energy calculation components
- Field update operations

**OperatorMemoryLayer Tests**:
- Operator storage and retrieval
- Pattern matching accuracy
- Operator instantiation
- Confidence tracking

**TransitionMemoryLayer Tests**:
- Transition storage
- Similarity search for transitions
- Completion pattern extraction

**FieldFormation Tests**:
- Multi-layer activation
- Simultaneous activation verification
- Initial resonance computation

**CandidateEmergence Tests**:
- Tension source identification
- Candidate generation from operators
- Candidate generation from transitions
- Energy reduction computation

**RecursiveStabilization Tests**:
- Convergence detection
- Oscillation detection
- Maximum iteration handling
- Damping application

**FreeEnergyMinimization Tests**:
- Energy formula correctness
- Evidence component calculation
- Conflict component calculation
- Energy decomposition

**Unit Test Example**:

```python
def test_state_field_energy_computation():
    """Test that field energy is computed as conflicts - evidence"""
    field = StateField(
        query_features=["test"],
        query_hv=np.random.rand(8192),
        activated_events={"event1": 0.8},
        activated_concepts={},
        activated_operators={},
        conflict_markers=[
            ConflictMarker("c1", ConflictType.CONTRADICTION, ["e1", "e2"], 0.7, "test")
        ],
        goal_states=[
            GoalState("g1", "test_goal", 0.5, [])
        ],
        partial_outputs=[]
    )
    
    fem = FreeEnergyMinimization()
    energy = fem.compute_field_energy(field)
    
    # Verify formula
    expected_energy = energy.conflicts - energy.evidence
    assert abs(energy.total - expected_energy) < 1e-6
    
    # Verify breakdown consistency
    assert abs(sum(energy.evidence_breakdown.values()) - energy.evidence) < 1e-6
    assert abs(sum(energy.conflict_breakdown.values()) - energy.conflicts) < 1e-6
```

### Integration Testing

**Layer Interaction Tests**:

Test that the four layers work together correctly:

```python
def test_four_layer_integration():
    """Test that all four layers integrate correctly"""
    cognitive_field = CognitiveField(window_size=12, decay=0.72)
    
    # Layer 1: Store events
    cognitive_field.events_layer.store_event(create_test_event("Paris is capital of France"))
    
    # Layer 3: Induce operator from transitions
    transitions = [create_test_transition("question", "answer") for _ in range(5)]
    operators = cognitive_field.operators_layer.induce_operators(transitions)
    assert len(operators) > 0
    
    # Layer 4: Store transitions
    for t in transitions:
        cognitive_field.transitions_layer.store_transition(t.partial_state, t.complete_state)
    
    # Layer 2: Form field from input
    context = InputContext(text="What is the capital of France?", modality="text", metadata={})
    field = cognitive_field.form_field(context)
    
    # Verify multi-layer activation
    assert len(field.activated_events) > 0, "Events should be activated"
    assert len(field.activated_operators) > 0, "Operators should be activated"
```

**End-to-End Generation Tests**:

Test complete generation workflow:

```python
def test_end_to_end_question_answering():
    """Test complete workflow from question to answer"""
    cognitive_field = CognitiveField(window_size=12, decay=0.72)
    
    # Train with QA pairs
    qa_pairs = [
        ("What is 2+2?", "4"),
        ("What is the capital of France?", "Paris"),
        ("What color is the sky?", "Blue")
    ]
    
    for question, answer in qa_pairs:
        cognitive_field.expose_text(f"{question} {answer}", {"type": "qa"})
    
    # Generate answer for new question
    result = cognitive_field.generate("What is the capital of France?", max_iterations=10)
    
    assert result.converged, "Generation should converge"
    assert "Paris" in result.final_output or "paris" in result.final_output.lower()
    assert result.final_energy < 0.5, "Final energy should be low"
```

**Backward Compatibility Tests**:

Verify that Original PUHL API still works:

```python
def test_backward_compatibility_expose_rank():
    """Test that original PUHL expose/rank API works"""
    cognitive_field = CognitiveField(window_size=12, decay=0.72)
    adapter = PUHLCompatibilityAdapter(cognitive_field)
    
    # Use old API
    adapter.expose_text("The sky is blue", {"source": "test"})
    adapter.expose_text("Grass is green", {"source": "test"})
    
    # Rank using old API
    scores = adapter.rank("What color is the sky?", ["blue", "green", "red"])
    
    assert len(scores) == 3
    assert scores[0] > scores[1], "Blue should score higher than green"
    assert scores[0] > scores[2], "Blue should score higher than red"
```

### Performance Benchmarking

**Benchmark Suite Execution**:

Run existing PUHL benchmarks to ensure performance preservation:

```bash
# All benchmarks must complete without degradation > 5%
python scratch/benchmark_generation_quality.py
python scratch/benchmark_multimodal_generalization.py
python scratch/benchmark_puhl_energy_modes.py
python scratch/benchmark_hopfield_continuous.py
python scratch/benchmark_repeated_exposure_stability.py
```

**Performance Metrics**:
- Inference speed: tokens/second for generation
- Learning speed: events/second for exposure
- Accuracy: task-specific correctness metrics
- Memory usage: peak RAM consumption
- Energy computation time: milliseconds per field

**Benchmark Comparison**:

```python
def test_performance_preservation():
    """Verify performance matches or exceeds Original PUHL baseline"""
    baseline = load_baseline_metrics()
    
    cognitive_field = CognitiveField(window_size=12, decay=0.72)
    metrics = run_benchmark_suite(cognitive_field)
    
    for benchmark_name, metric_value in metrics.items():
        baseline_value = baseline[benchmark_name]
        degradation = (baseline_value - metric_value) / baseline_value
        
        assert degradation < 0.05, \
            f"{benchmark_name}: degradation {degradation*100:.1f}% exceeds 5% threshold"
```

### Test Coverage Goals

**Code Coverage**: Minimum 85% line coverage, 75% branch coverage
**Property Coverage**: All 27 correctness properties have property tests
**Component Coverage**: Every component has unit tests
**Integration Coverage**: All layer interactions tested
**Benchmark Coverage**: All 5 benchmarks pass within 5% of baseline

### Continuous Integration

**Test Execution Order**:
1. Unit tests (fast, run on every commit)
2. Property tests (medium, run on every commit)
3. Integration tests (medium, run on every commit)
4. Benchmark tests (slow, run on PR and nightly)

**Failure Handling**:
- Unit/Property/Integration test failures block merge
- Benchmark degradation > 5% blocks merge
- Benchmark degradation 1-5% requires justification

### Test Data Generation

**Synthetic Data**:
- Random state fields with controlled properties
- Generated transitions with known patterns
- Artificial conflicts and goals

**Real Data**:
- Historical PUHL brain_memory.pkl files
- Sample conversations, code, documents
- Multi-modal test cases

**Adversarial Data**:
- Edge cases: empty inputs, maximum sizes
- Pathological patterns: infinite loops, oscillations
- Corrupted data: malformed structures

### Manual Testing Scenarios

While most testing is automated, some scenarios require manual validation:

1. **Emergent Behavior Evaluation**: Does field resonance create meaningful emergent patterns?
2. **Operator Generalization Quality**: Do induced operators generalize appropriately?
3. **Energy Interpretation**: Are energy decompositions interpretable and useful?
4. **Convergence Behavior**: Does stabilization produce sensible outputs?

These require human judgment and are tested through:
- Interactive notebooks with visualizations
- Qualitative case studies
- User acceptance testing
