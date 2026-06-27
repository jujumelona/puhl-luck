# Requirements Document

## Introduction

This document specifies requirements for redesigning PUHL as a **Predictive Field Memory** system—a fundamental paradigm shift from "search-then-select" to **living cognitive state fields**. Instead of retrieving events and scoring candidates, the system forms **simultaneous activation fields** where memories interact dynamically, generating outputs from field tension rather than selection from lists. This transforms PUHL from a memory retrieval system into a general-purpose AI based on state field dynamics.

## Glossary

- **Cognitive_Field**: A living state composed of simultaneously activated memories, features, concepts, and operators that interact to generate outputs
- **Exposure_Events_Layer**: Layer 1—stores raw observational data (current PUHL events, edges, HDC features)
- **State_Field_Layer**: Layer 2—the currently activated cognitive state formed by input, containing query features, activated events, activated concepts, conflicts, goals, and partial outputs
- **Operator_Memory_Layer**: Layer 3—stores repeated state transformation patterns (question→explanation, error→correction, incomplete→complete)
- **Transition_Memory_Layer**: Layer 4—stores partial-to-complete state transitions, not before/after pairs
- **Field_Formation**: Process where input simultaneously activates multiple memory layers creating a resonant cognitive field
- **Candidate_Emergence**: Process where candidates arise spontaneously from field tension rather than being retrieved
- **Free_Energy_Minimization**: The field naturally moves toward lower energy states by resolving conflicts and increasing evidence
- **Operator_Induction**: Learning repeated transformation patterns from experience
- **State_Completion**: The core operation—transforming incomplete states into complete states
- **Recursive_Stabilization**: After output, field is updated and process repeats until stable
- **Tension**: High energy in the cognitive field caused by incompleteness, conflicts, or unsatisfied goals
- **Resonance**: When multiple activated memories reinforce each other in the cognitive field
- **Original_PUHL**: The current event-based memory system with exposure/co-activation structure
- **Performance_Benchmark_Suite**: The collection of benchmarks validating that redesign maintains performance

## Requirements

### Requirement 1: Four-Layer Memory Architecture

**User Story:** As a memory system architect, I want memory organized in four distinct layers, so that raw experience, activated states, learned operators, and state transitions are properly separated.

#### Acceptance Criteria

1. THE Cognitive_Field SHALL implement Layer 1 as Exposure_Events_Layer storing raw observational data
2. THE Cognitive_Field SHALL implement Layer 2 as State_Field_Layer storing currently activated cognitive states
3. THE Cognitive_Field SHALL implement Layer 3 as Operator_Memory_Layer storing repeated transformation patterns
4. THE Cognitive_Field SHALL implement Layer 4 as Transition_Memory_Layer storing partial-to-complete state transitions
5. THE Exposure_Events_Layer SHALL preserve the Original_PUHL structure (events, edges, HDC features)
6. THE State_Field_Layer SHALL contain query features, activated events, activated concepts, conflicts, goals, and partial outputs
7. THE Operator_Memory_Layer SHALL store transformation types including completion_operator, repair_operator, explanation_operator, comparison_operator, transformation_operator
8. THE Transition_Memory_Layer SHALL store S_partial → S_complete pairs, not S_before → S_after pairs

### Requirement 2: Simultaneous Memory Activation

**User Story:** As a cognitive modeling expert, I want input to activate multiple memory layers simultaneously, so that the system forms an interactive state field rather than a search result list.

#### Acceptance Criteria

1. WHEN input is received, THE Field_Formation SHALL activate relevant events from Exposure_Events_Layer simultaneously
2. WHEN input is received, THE Field_Formation SHALL activate relevant concepts from concept memory simultaneously
3. WHEN input is received, THE Field_Formation SHALL activate relevant operators from Operator_Memory_Layer simultaneously
4. THE State_Field_Layer SHALL represent activations as an interactive cognitive field, not a ranked list
5. THE State_Field_Layer SHALL maintain activation strengths for all simultaneously activated memories
6. THE State_Field_Layer SHALL compute resonance between activated memories where mutually supporting memories reinforce each other
7. WHEN multiple memories are activated, THE Field_Formation SHALL allow them to interact, creating emergent patterns beyond individual activations

### Requirement 3: Candidate Emergence from Field Tension

**User Story:** As a generation system designer, I want candidates to emerge from field tension rather than being retrieved, so that outputs arise naturally from cognitive dynamics.

#### Acceptance Criteria

1. THE Candidate_Emergence SHALL generate candidates from incomplete cognitive field states, not from retrieval operations
2. WHEN the State_Field_Layer contains incomplete information, THE Cognitive_Field SHALL compute field tension as high energy
3. WHEN field tension is high, THE Candidate_Emergence SHALL identify tension-reducing continuations
4. THE Candidate_Emergence SHALL use Free_Energy_Minimization as the generation principle, not as a post-generation scoring function
5. THE Candidate_Emergence SHALL generate candidates that reduce conflict markers in the State_Field_Layer
6. THE Candidate_Emergence SHALL generate candidates that increase evidence accumulation in the State_Field_Layer
7. THE Candidate_Emergence SHALL generate candidates that satisfy active goals in the State_Field_Layer
8. THE Free_Energy_Minimization SHALL drive candidate generation at creation time, not at evaluation time

### Requirement 4: Operator Memory and Induction

**User Story:** As a generalization engineer, I want the system to learn repeated transformation patterns as operators, so that it can apply learned rules to novel situations.

#### Acceptance Criteria

1. THE Operator_Memory_Layer SHALL store patterns as state transformation operators, not as event templates
2. THE Operator_Induction SHALL identify repeated patterns across multiple exposures
3. WHEN a question→answer pattern repeats, THE Operator_Induction SHALL create an explanation_operator
4. WHEN an error→correction pattern repeats, THE Operator_Induction SHALL create a repair_operator
5. WHEN an incomplete→complete pattern repeats, THE Operator_Induction SHALL create a completion_operator
6. WHEN a problem→solution pattern repeats, THE Operator_Induction SHALL create a transformation_operator
7. THE Candidate_Emergence SHALL apply stored operators to novel inputs
8. THE Operator_Memory_Layer SHALL enable generalization beyond specific event examples

### Requirement 5: Partial-to-Complete State Transitions

**User Story:** As a learning system designer, I want all learning to focus on completing incomplete states, so that the system learns restoration rather than prediction.

#### Acceptance Criteria

1. THE Transition_Memory_Layer SHALL store transitions as S_partial → S_complete, not S_before → S_after
2. WHEN storing a conversation turn, THE Transition_Memory_Layer SHALL record incomplete_context → completed_context
3. WHEN storing a code generation, THE Transition_Memory_Layer SHALL record incomplete_code → completed_code
4. WHEN storing a reasoning step, THE Transition_Memory_Layer SHALL record incomplete_reasoning → completed_reasoning
5. THE State_Completion SHALL treat token prediction as surface behavior and state completion as internal behavior
6. THE State_Completion SHALL apply the same completion principle to conversation, code, documents, and reasoning
7. THE Cognitive_Field SHALL learn to identify what makes a state incomplete and how to complete it

### Requirement 6: Recursive Field Stabilization

**User Story:** As a dynamic system designer, I want the cognitive field to update after each output and repeat the process, so that generation proceeds through iterative stabilization.

#### Acceptance Criteria

1. WHEN the Cognitive_Field produces an output, THE Recursive_Stabilization SHALL update the State_Field_Layer with the new output
2. WHEN the State_Field_Layer is updated, THE Field_Formation SHALL recompute activations based on the new state
3. WHEN activations are recomputed, THE Candidate_Emergence SHALL generate new candidates from the updated field
4. THE Recursive_Stabilization SHALL follow the loop: context → field formation → candidate emergence → output → field update → next field formation
5. THE Recursive_Stabilization SHALL continue until the field reaches a stable low-energy state
6. THE Recursive_Stabilization SHALL detect stability when field energy changes fall below a threshold

### Requirement 7: Field-Based Free Energy Minimization

**User Story:** As a cognitive dynamics expert, I want free energy minimization to drive field stabilization, so that the system naturally moves toward coherent states.

#### Acceptance Criteria

1. THE Free_Energy_Minimization SHALL compute field energy as conflicts minus evidence
2. THE Free_Energy_Minimization SHALL increase evidence score WHEN activated memories resonate with current input
3. THE Free_Energy_Minimization SHALL increase evidence score WHEN the field contains consistent goal-satisfying patterns
4. THE Free_Energy_Minimization SHALL increase evidence score WHEN the field has strong memory support
5. THE Free_Energy_Minimization SHALL increase conflict score WHEN activated memories contradict each other
6. THE Free_Energy_Minimization SHALL increase conflict score WHEN field contains constraint violations
7. THE Free_Energy_Minimization SHALL increase conflict score WHEN field contains repetition patterns
8. THE Recursive_Stabilization SHALL drive the field toward lower energy states through iterative candidate emergence
9. THE Free_Energy_Minimization SHALL act as a state field property, not a candidate scoring function

### Requirement 8: Universal State Completion Principle

**User Story:** As a general AI developer, I want conversation, code, documents, and reasoning to use the same completion principle, so that the system is truly domain-agnostic.

#### Acceptance Criteria

1. THE State_Completion SHALL apply to conversation by completing incomplete dialogue contexts
2. THE State_Completion SHALL apply to code generation by completing incomplete program states
3. THE State_Completion SHALL apply to document writing by completing incomplete document structures
4. THE State_Completion SHALL apply to reasoning by completing incomplete inference chains
5. THE State_Completion SHALL use the same completion algorithm across all domains
6. THE State_Completion SHALL differ only in surface representation (tokens, syntax) but not in completion dynamics

### Requirement 9: Operator-Level Generalization

**User Story:** As a transfer learning researcher, I want operators to enable application of learned transformations to new contexts, so that the system generalizes beyond memorized examples.

#### Acceptance Criteria

1. WHEN a novel input arrives, THE Candidate_Emergence SHALL identify applicable operators from Operator_Memory_Layer
2. WHEN an operator matches the current field state, THE Candidate_Emergence SHALL instantiate the operator with context-appropriate parameters
3. THE Operator_Induction SHALL generalize across different surface forms with similar transformation structure
4. THE Operator_Memory_Layer SHALL enable the system to apply "question→explanation" to new question types never seen before
5. THE Operator_Memory_Layer SHALL enable the system to apply "error→repair" to new error types never seen before
6. THE Operator_Memory_Layer SHALL provide stronger generalization than feature cluster matching alone

### Requirement 10: Preservation of Original PUHL Components

**User Story:** As a PUHL maintainer, I want the redesign to preserve successful Original_PUHL components, so that proven capabilities are retained.

#### Acceptance Criteria

1. THE Exposure_Events_Layer SHALL preserve the Original_PUHL event storage structure
2. THE Exposure_Events_Layer SHALL preserve the Original_PUHL co-activation edge structure
3. THE Field_Formation SHALL use Original_PUHL HDC-based feature extraction
4. THE Free_Energy_Minimization SHALL use Original_PUHL energy-based scoring components
5. THE Cognitive_Field SHALL preserve Original_PUHL micro-rank artifact functionality
6. THE Cognitive_Field SHALL preserve Original_PUHL novelty detection
7. THE Cognitive_Field SHALL preserve Original_PUHL targeted forgetting
8. THE Exposure_Events_Layer SHALL maintain compatibility with existing brain_memory.pkl files

### Requirement 11: Performance Preservation

**User Story:** As a system maintainer, I want the redesigned system to maintain current performance levels, so that the paradigm shift does not degrade user experience.

#### Acceptance Criteria

1. WHEN running the Performance_Benchmark_Suite, THE Cognitive_Field SHALL achieve inference speed equal to or faster than the Original_PUHL
2. WHEN running the Performance_Benchmark_Suite, THE Cognitive_Field SHALL achieve learning speed equal to or faster than the Original_PUHL
3. WHEN running the Performance_Benchmark_Suite, THE Cognitive_Field SHALL achieve accuracy equal to or higher than the Original_PUHL on all benchmark tasks
4. THE Performance_Benchmark_Suite SHALL include benchmark_generation_quality.py
5. THE Performance_Benchmark_Suite SHALL include benchmark_multimodal_generalization.py
6. THE Performance_Benchmark_Suite SHALL include benchmark_puhl_energy_modes.py
7. THE Performance_Benchmark_Suite SHALL include benchmark_hopfield_continuous.py
8. THE Performance_Benchmark_Suite SHALL include benchmark_repeated_exposure_stability.py
9. IF any benchmark shows performance degradation exceeding 5% from baseline, THEN THE Cognitive_Field SHALL not replace the Original_PUHL until performance is recovered

### Requirement 12: Field State Representation

**User Story:** As a cognitive state modeler, I want the State_Field_Layer to represent all aspects of the current cognitive state, so that field dynamics can operate on complete information.

#### Acceptance Criteria

1. THE State_Field_Layer SHALL maintain query feature representations
2. THE State_Field_Layer SHALL maintain activated event identifiers with activation strengths
3. THE State_Field_Layer SHALL maintain activated concept identifiers with activation strengths
4. THE State_Field_Layer SHALL maintain activated operator identifiers with activation strengths
5. THE State_Field_Layer SHALL maintain conflict markers with conflict strengths
6. THE State_Field_Layer SHALL maintain goal states with satisfaction levels
7. THE State_Field_Layer SHALL maintain partial output representations
8. THE State_Field_Layer SHALL compute field energy from the combination of all maintained components

### Requirement 13: Tension-Driven Generation Principle

**User Story:** As a cognitive generation designer, I want high field tension to drive candidate generation, so that outputs emerge from system dynamics rather than selection.

#### Acceptance Criteria

1. THE Candidate_Emergence SHALL compute field tension from incomplete state markers
2. WHEN field tension exceeds a threshold, THE Candidate_Emergence SHALL generate tension-reducing continuations
3. THE Candidate_Emergence SHALL prioritize continuations that resolve conflicts in the State_Field_Layer
4. THE Candidate_Emergence SHALL prioritize continuations that satisfy unsatisfied goals in the State_Field_Layer
5. THE Candidate_Emergence SHALL prioritize continuations that complete partial outputs in the State_Field_Layer
6. THE Candidate_Emergence SHALL not perform candidate retrieval followed by ranking
7. THE Candidate_Emergence SHALL generate candidates directly from field dynamics

### Requirement 14: Resonance-Based Memory Interaction

**User Story:** As a memory interaction designer, I want simultaneously activated memories to interact through resonance, so that the field exhibits emergent behavior beyond individual memories.

#### Acceptance Criteria

1. WHEN multiple events are simultaneously activated, THE State_Field_Layer SHALL compute resonance between them
2. THE State_Field_Layer SHALL increase activation strength WHEN two memories mutually support each other
3. THE State_Field_Layer SHALL decrease activation strength WHEN two memories conflict with each other
4. THE State_Field_Layer SHALL detect resonance through feature overlap, semantic similarity, and co-activation history
5. THE Field_Formation SHALL allow resonance to propagate through the field, amplifying coherent patterns
6. THE Field_Formation SHALL allow dissonance to suppress incoherent patterns

### Requirement 15: Learning as State Completion Training

**User Story:** As a machine learning engineer, I want the training objective to be "complete incomplete states", so that learning focuses on restoration rather than prediction.

#### Acceptance Criteria

1. THE Cognitive_Field SHALL train using "incomplete state → complete state" as the unified learning objective
2. WHEN training on conversation data, THE Cognitive_Field SHALL learn to complete incomplete dialogue states
3. WHEN training on code data, THE Cognitive_Field SHALL learn to complete incomplete program states
4. WHEN training on document data, THE Cognitive_Field SHALL learn to complete incomplete document states
5. WHEN training on reasoning data, THE Cognitive_Field SHALL learn to complete incomplete inference states
6. THE Cognitive_Field SHALL use completion accuracy as the primary training metric
7. THE Cognitive_Field SHALL not use next-token prediction as the primary training objective

### Requirement 16: Operator Type Taxonomy

**User Story:** As an operator memory designer, I want a clear taxonomy of operator types, so that different transformation patterns are properly categorized.

#### Acceptance Criteria

1. THE Operator_Memory_Layer SHALL support completion_operator for incomplete→complete transformations
2. THE Operator_Memory_Layer SHALL support repair_operator for error→correction transformations
3. THE Operator_Memory_Layer SHALL support explanation_operator for question→answer transformations
4. THE Operator_Memory_Layer SHALL support comparison_operator for entity→comparison transformations
5. THE Operator_Memory_Layer SHALL support transformation_operator for problem→solution transformations
6. THE Operator_Memory_Layer SHALL support composition_operator for part→whole transformations
7. THE Operator_Memory_Layer SHALL allow new operator types to be added without architectural changes

### Requirement 17: Field Energy Decomposition

**User Story:** As a debugging engineer, I want to decompose field energy into components, so that I can understand why the field is in high or low energy states.

#### Acceptance Criteria

1. THE Free_Energy_Minimization SHALL provide energy decomposition showing evidence components
2. THE Free_Energy_Minimization SHALL provide energy decomposition showing conflict components
3. THE Free_Energy_Minimization SHALL compute evidence from memory resonance strength
4. THE Free_Energy_Minimization SHALL compute evidence from goal satisfaction level
5. THE Free_Energy_Minimization SHALL compute evidence from constraint satisfaction level
6. THE Free_Energy_Minimization SHALL compute conflict from contradiction detection
7. THE Free_Energy_Minimization SHALL compute conflict from repetition detection
8. THE Free_Energy_Minimization SHALL compute conflict from constraint violation detection

### Requirement 18: Backward Compatibility with Original PUHL API

**User Story:** As an existing PUHL user, I want the new system to support existing API operations, so that my current workflows continue to function during migration.

#### Acceptance Criteria

1. THE Cognitive_Field SHALL provide an adapter that translates Original_PUHL API calls to field operations
2. WHEN the adapter receives an expose_text call, THE Cognitive_Field SHALL store it in Exposure_Events_Layer
3. WHEN the adapter receives an expose_file call, THE Cognitive_Field SHALL store it in Exposure_Events_Layer
4. WHEN the adapter receives a rank query, THE Cognitive_Field SHALL execute it using Field_Formation and Free_Energy_Minimization
5. THE Cognitive_Field SHALL maintain backward compatibility with existing brain_memory.pkl files

### Requirement 19: Multi-Modality State Completion

**User Story:** As a multimodal AI researcher, I want state completion to work across text, code, images, and structured data, so that the system handles diverse input types.

#### Acceptance Criteria

1. THE State_Completion SHALL support completing incomplete text contexts
2. THE State_Completion SHALL support completing incomplete code contexts
3. THE State_Completion SHALL support completing incomplete structured data contexts
4. THE State_Completion SHALL support completing incomplete multimodal contexts containing multiple modalities
5. THE State_Field_Layer SHALL represent multiple modalities simultaneously in the cognitive field
6. THE Field_Formation SHALL activate memories from multiple modalities simultaneously

### Requirement 20: Convergence Guarantees for Recursive Stabilization

**User Story:** As a system reliability engineer, I want recursive stabilization to have convergence guarantees, so that the system does not loop infinitely.

#### Acceptance Criteria

1. THE Recursive_Stabilization SHALL implement a maximum iteration limit to prevent infinite loops
2. THE Recursive_Stabilization SHALL detect convergence when field energy change falls below a threshold across consecutive iterations
3. THE Recursive_Stabilization SHALL detect oscillation when field energy alternates between similar values
4. WHEN oscillation is detected, THE Recursive_Stabilization SHALL apply damping to reduce oscillation amplitude
5. WHEN maximum iterations are reached without convergence, THE Recursive_Stabilization SHALL return the lowest-energy state encountered
6. THE Recursive_Stabilization SHALL log convergence statistics for analysis

