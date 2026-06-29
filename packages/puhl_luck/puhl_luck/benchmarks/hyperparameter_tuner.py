"""
Hyperparameter tuning infrastructure for HDC system optimization.

This module implements grid search and Pareto-optimal configuration selection
for the HDC (Hyperdimensional Computing) sparse autoregressive generator.

As per Requirements 12.1-12.5:
- Evaluates all combinations of context window (3-10), rare threshold (1-5), top-K (1-10)
- Measures both accuracy and speed metrics
- Identifies Pareto-optimal configurations
- Saves tuning results for analysis
- Recommends best configuration based on user priority
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from puhl_luck.brain_memory import BrainMemory


@dataclass
class HyperparameterConfig:
    """
    Configuration for hyperparameter tuning.
    
    Attributes:
        context_window: Number of tokens in context (K)
        rare_token_threshold: Frequency threshold for copy gate activation
        top_k: Number of top candidates to consider during generation
    """
    context_window: int
    rare_token_threshold: int
    top_k: int
    
    def to_dict(self) -> Dict[str, int]:
        """Convert config to dictionary."""
        return {
            'context_window': self.context_window,
            'rare_token_threshold': self.rare_token_threshold,
            'top_k': self.top_k,
        }
    
    def __hash__(self) -> int:
        """Make config hashable for use as dict key."""
        return hash((self.context_window, self.rare_token_threshold, self.top_k))


@dataclass
class TuningResult:
    """
    Results from evaluating a single hyperparameter configuration.
    
    Attributes:
        config: Hyperparameter configuration tested
        accuracy: Proportion of correct predictions (0-1)
        avg_inference_time_ms: Average inference time in milliseconds
        total_tests: Number of test cases evaluated
        passed: Number of correct predictions
        failed: Number of incorrect predictions
        empty_outputs: Number of empty generation results
        avg_backoff_level: Average backoff level used (0=exact match)
        copy_gate_activations: Number of copy gate activations
    """
    config: HyperparameterConfig
    accuracy: float
    avg_inference_time_ms: float
    total_tests: int
    passed: int
    failed: int
    empty_outputs: int
    avg_backoff_level: float
    copy_gate_activations: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'config': self.config.to_dict(),
            'accuracy': self.accuracy,
            'avg_inference_time_ms': self.avg_inference_time_ms,
            'total_tests': self.total_tests,
            'passed': self.passed,
            'failed': self.failed,
            'empty_outputs': self.empty_outputs,
            'avg_backoff_level': self.avg_backoff_level,
            'copy_gate_activations': self.copy_gate_activations,
        }


class HyperparameterTuner:
    """
    Automated hyperparameter tuning for HDC system.
    
    Implements grid search over hyperparameter space to find optimal configurations
    that balance accuracy and speed. Supports Pareto-optimal configuration selection
    and priority-based recommendations.
    
    As per Requirement 12.1: Evaluates all combinations of:
    - context_window: 3-10 tokens
    - rare_token_threshold: 1-5 occurrences
    - top_k: 1-10 candidates
    
    Usage:
        >>> tuner = HyperparameterTuner(train_data, test_data)
        >>> results = tuner.grid_search()
        >>> best_config = tuner.recommend_config(results, priority='balanced')
    """
    
    def __init__(
        self,
        train_data: List[Tuple[str, str]],
        test_data: List[Tuple[str, str]],
        domain: str = 'default'
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            train_data: List of (input, target) training pairs
            test_data: List of (input, expected_output) test pairs
            domain: Domain identifier for training/testing
            
        Requirements: 12.1, 12.2
        """
        self.train_data = train_data
        self.test_data = test_data
        self.domain = domain
        
        # Default search spaces (as per Requirements 5.1, 6.1, 7.1)
        self.context_windows: List[int] = [3, 4, 5, 6, 7, 8, 10]
        self.rare_thresholds: List[int] = [1, 2, 3, 4, 5]
        self.top_k_values: List[int] = [1, 2, 3, 5, 8, 10]
        
        # Storage for results
        self.results: List[TuningResult] = []
        
    def set_search_space(
        self,
        context_windows: Optional[List[int]] = None,
        rare_thresholds: Optional[List[int]] = None,
        top_k_values: Optional[List[int]] = None
    ) -> None:
        """
        Configure custom search spaces for hyperparameters.
        
        Args:
            context_windows: List of context window sizes to evaluate
            rare_thresholds: List of rare token thresholds to evaluate
            top_k_values: List of top-K values to evaluate
            
        Requirements: 12.1
        """
        if context_windows is not None:
            self.context_windows = context_windows
        if rare_thresholds is not None:
            self.rare_thresholds = rare_thresholds
        if top_k_values is not None:
            self.top_k_values = top_k_values
    
    def _train_brain(
        self,
        brain: BrainMemory,
        config: HyperparameterConfig
    ) -> float:
        """
        Train brain with given configuration.
        
        Args:
            brain: BrainMemory instance to train
            config: Hyperparameter configuration
            
        Returns:
            Training time in milliseconds
            
        Requirements: 6.1, 6.2, 6.3, 6.4
        """
        start_time = time.time()
        
        # Apply configuration to the underlying sparse logit generator
        # The BrainMemory facade exposes the generator via _logit_generator
        if hasattr(brain, '_logit_generator') and brain._logit_generator is not None:
            lg = brain._logit_generator
            
            # Set top_k (number of candidates to consider during generation)
            # Requirement 7.1, 7.2: Top-K parameter optimization
            lg.top_k = config.top_k
            
            # Set repetition_window (context window size for repetition penalty)
            # Requirement 5.1, 5.2: Context window optimization
            # Note: This controls the repetition penalty window, not the context window
            # used for generation. The actual context window for feature extraction
            # is managed internally by the generator based on data scale.
            if hasattr(lg, 'scorer') and lg.scorer is not None:
                lg.scorer.repetition_window = config.context_window
            
            # Set rare_token_threshold for copy gate optimization
            # Requirement 6.1, 6.2, 6.3, 6.4: Tokens with frequency < threshold
            # are marked as copy candidates and prioritized for extraction
            lg.rare_token_threshold = config.rare_token_threshold
        
        # Train on all pairs
        for input_text, target_text in self.train_data:
            brain.expose_pair(
                partial=input_text,
                complete=target_text,
                domain=self.domain,
                modality=self.domain,
            )
        
        training_time_ms = (time.time() - start_time) * 1000
        return training_time_ms
    
    def _evaluate_config(
        self,
        config: HyperparameterConfig,
        max_new_tokens: int = 64
    ) -> TuningResult:
        """
        Evaluate a single hyperparameter configuration.
        
        Args:
            config: Hyperparameter configuration to evaluate
            max_new_tokens: Maximum tokens to generate per query
            
        Returns:
            TuningResult with performance metrics
            
        Requirements: 12.2
        """
        # Create fresh brain instance
        brain = BrainMemory()
        
        # Train with this configuration
        self._train_brain(brain, config)
        
        # Evaluate on test data
        total_tests = len(self.test_data)
        passed = 0
        failed = 0
        empty_outputs = 0
        inference_times = []
        backoff_levels_sum = 0
        backoff_count = 0
        copy_gate_activations = 0
        
        for input_text, expected_output in self.test_data:
            start_time = time.time()
            
            try:
                # Generate with metrics
                result = brain.generate(
                    query=input_text,
                    max_new_tokens=max_new_tokens,
                    domain=self.domain,
                    return_metrics=True,
                )
                
                # Unpack result
                if isinstance(result, tuple) and len(result) == 2:
                    generated, gen_metrics = result
                else:
                    generated = result
                    gen_metrics = None
                
                inference_time_ms = (time.time() - start_time) * 1000
                inference_times.append(inference_time_ms)
                
                # Check if output is empty
                is_empty = len(generated.strip()) == 0
                if is_empty:
                    empty_outputs += 1
                
                # Extract metrics if available
                if gen_metrics:
                    if hasattr(gen_metrics, 'backoff_levels'):
                        for level, count in gen_metrics.backoff_levels.items():
                            backoff_levels_sum += level * count
                            backoff_count += count
                    
                    if hasattr(gen_metrics, 'copy_gate_activations'):
                        copy_gate_activations += gen_metrics.copy_gate_activations
                
                # Evaluate correctness (simple substring match)
                correct = expected_output.lower() in generated.lower()
                
                if correct:
                    passed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                # Handle generation failures
                inference_time_ms = (time.time() - start_time) * 1000
                inference_times.append(inference_time_ms)
                failed += 1
                empty_outputs += 1
        
        # Calculate aggregate metrics
        accuracy = passed / total_tests if total_tests > 0 else 0.0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
        avg_backoff = backoff_levels_sum / backoff_count if backoff_count > 0 else 0.0
        
        return TuningResult(
            config=config,
            accuracy=accuracy,
            avg_inference_time_ms=avg_inference_time,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            empty_outputs=empty_outputs,
            avg_backoff_level=avg_backoff,
            copy_gate_activations=copy_gate_activations,
        )
    
    def grid_search(
        self,
        context_windows: Optional[List[int]] = None,
        rare_thresholds: Optional[List[int]] = None,
        top_k_values: Optional[List[int]] = None,
        max_new_tokens: int = 64,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform exhaustive grid search over hyperparameter space.
        
        As per Requirement 12.1: Evaluates all combinations of context window,
        rare token threshold, and top-K values.
        
        Args:
            context_windows: List of context window sizes (default: [3-10])
            rare_thresholds: List of rare token thresholds (default: [1-5])
            top_k_values: List of top-K values (default: [1-10])
            max_new_tokens: Maximum tokens to generate per query
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing:
            - all_results: List of all TuningResult objects
            - best_accuracy_config: Config with highest accuracy
            - best_speed_config: Config with lowest inference time
            - pareto_front: List of Pareto-optimal configs
            - total_evaluations: Number of configurations tested
            - total_time_ms: Total evaluation time
            
        Requirements: 12.1, 12.2, 12.3
        """
        # Update search spaces if provided
        if context_windows is not None:
            self.context_windows = context_windows
        if rare_thresholds is not None:
            self.rare_thresholds = rare_thresholds
        if top_k_values is not None:
            self.top_k_values = top_k_values
        
        if verbose:
            print("=" * 70)
            print("HYPERPARAMETER GRID SEARCH")
            print("=" * 70)
            print(f"Context windows: {self.context_windows}")
            print(f"Rare thresholds: {self.rare_thresholds}")
            print(f"Top-K values: {self.top_k_values}")
            total_configs = len(self.context_windows) * len(self.rare_thresholds) * len(self.top_k_values)
            print(f"Total configurations: {total_configs}")
            print()
        
        # Clear previous results
        self.results = []
        total_start_time = time.time()
        
        # Grid search over all combinations
        config_count = 0
        for context_window in self.context_windows:
            for rare_threshold in self.rare_thresholds:
                for top_k in self.top_k_values:
                    config_count += 1
                    
                    config = HyperparameterConfig(
                        context_window=context_window,
                        rare_token_threshold=rare_threshold,
                        top_k=top_k,
                    )
                    
                    if verbose:
                        print(f"[{config_count}/{total_configs}] Evaluating: "
                              f"K={context_window}, rare={rare_threshold}, top_k={top_k}")
                    
                    # Evaluate configuration
                    result = self._evaluate_config(config, max_new_tokens)
                    self.results.append(result)
                    
                    if verbose:
                        print(f"  Accuracy: {result.accuracy * 100:.1f}%, "
                              f"Speed: {result.avg_inference_time_ms:.2f}ms")
        
        total_time_ms = (time.time() - total_start_time) * 1000
        
        if verbose:
            print()
            print(f"Grid search completed in {total_time_ms / 1000:.2f}s")
            print()
        
        # Find best configurations
        best_accuracy_result = max(self.results, key=lambda r: r.accuracy)
        best_speed_result = min(self.results, key=lambda r: r.avg_inference_time_ms)
        
        # Identify Pareto front (Requirement 12.3)
        pareto_front = self._identify_pareto_front(self.results)
        
        results_dict = {
            'all_results': [r.to_dict() for r in self.results],
            'best_accuracy_config': {
                'config': best_accuracy_result.config.to_dict(),
                'accuracy': best_accuracy_result.accuracy,
                'avg_inference_time_ms': best_accuracy_result.avg_inference_time_ms,
            },
            'best_speed_config': {
                'config': best_speed_result.config.to_dict(),
                'accuracy': best_speed_result.accuracy,
                'avg_inference_time_ms': best_speed_result.avg_inference_time_ms,
            },
            'pareto_front': [r.to_dict() for r in pareto_front],
            'total_evaluations': len(self.results),
            'total_time_ms': total_time_ms,
        }
        
        if verbose:
            print("=" * 70)
            print("GRID SEARCH RESULTS")
            print("=" * 70)
            print(f"Best accuracy: {best_accuracy_result.accuracy * 100:.1f}% "
                  f"(K={best_accuracy_result.config.context_window}, "
                  f"rare={best_accuracy_result.config.rare_token_threshold}, "
                  f"top_k={best_accuracy_result.config.top_k})")
            print(f"Best speed: {best_speed_result.avg_inference_time_ms:.2f}ms "
                  f"(K={best_speed_result.config.context_window}, "
                  f"rare={best_speed_result.config.rare_token_threshold}, "
                  f"top_k={best_speed_result.config.top_k})")
            print(f"Pareto-optimal configs: {len(pareto_front)}")
            print("=" * 70)
        
        return results_dict
    
    def _identify_pareto_front(
        self,
        results: List[TuningResult]
    ) -> List[TuningResult]:
        """
        Identify Pareto-optimal configurations.
        
        A configuration is Pareto-optimal if no other configuration is strictly
        better in both accuracy and speed.
        
        Args:
            results: List of tuning results
            
        Returns:
            List of Pareto-optimal TuningResult objects
            
        Requirement: 12.3
        """
        pareto_front = []
        
        for candidate in results:
            is_dominated = False
            
            # Check if candidate is dominated by any other result
            for other in results:
                if other == candidate:
                    continue
                
                # Other dominates candidate if it's strictly better in both metrics
                # (higher accuracy AND lower inference time)
                if (other.accuracy >= candidate.accuracy and
                    other.avg_inference_time_ms <= candidate.avg_inference_time_ms and
                    (other.accuracy > candidate.accuracy or 
                     other.avg_inference_time_ms < candidate.avg_inference_time_ms)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def recommend_config(
        self,
        results: Optional[Dict[str, Any]] = None,
        priority: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Recommend best configuration based on user priority.
        
        As per Requirement 12.5: Recommends configuration based on priority:
        - 'accuracy': Maximize accuracy
        - 'speed': Minimize inference time
        - 'balanced': Balance accuracy and speed
        
        Args:
            results: Results from grid_search(). If None, uses last search results.
            priority: One of 'accuracy', 'speed', 'balanced'
            
        Returns:
            Dictionary with recommended configuration and metrics
            
        Requirement: 12.5
        """
        if results is None:
            if not self.results:
                raise ValueError("No results available. Run grid_search() first.")
            results_list = self.results
        else:
            results_list = [
                TuningResult(
                    config=HyperparameterConfig(**r['config']),
                    accuracy=r['accuracy'],
                    avg_inference_time_ms=r['avg_inference_time_ms'],
                    total_tests=r['total_tests'],
                    passed=r['passed'],
                    failed=r['failed'],
                    empty_outputs=r['empty_outputs'],
                    avg_backoff_level=r['avg_backoff_level'],
                    copy_gate_activations=r['copy_gate_activations'],
                )
                for r in results['all_results']
            ]
        
        if priority == 'accuracy':
            # Select configuration with highest accuracy
            best_result = max(results_list, key=lambda r: r.accuracy)
            
        elif priority == 'speed':
            # Select configuration with lowest inference time
            best_result = min(results_list, key=lambda r: r.avg_inference_time_ms)
            
        elif priority == 'balanced':
            # Select configuration with best balanced score
            # Normalize accuracy (0-1) and speed (invert and normalize)
            # Use weighted geometric mean
            max_time = max(r.avg_inference_time_ms for r in results_list)
            
            def balanced_score(r: TuningResult) -> float:
                # Normalize speed score (1 - normalized time)
                speed_score = 1 - (r.avg_inference_time_ms / max_time)
                # Geometric mean of accuracy and speed
                return (r.accuracy * speed_score) ** 0.5
            
            best_result = max(results_list, key=balanced_score)
            
        else:
            raise ValueError(f"Invalid priority: {priority}. "
                           f"Must be 'accuracy', 'speed', or 'balanced'.")
        
        return {
            'priority': priority,
            'recommended_config': best_result.config.to_dict(),
            'accuracy': best_result.accuracy,
            'avg_inference_time_ms': best_result.avg_inference_time_ms,
            'empty_outputs': best_result.empty_outputs,
            'avg_backoff_level': best_result.avg_backoff_level,
        }
    
    def save_tuning_results(
        self,
        results: Dict[str, Any],
        filename: str = 'hyperparameter_tuning_results.json'
    ) -> str:
        """
        Save all tested configurations to JSON for analysis.
        
        As per Requirement 12.4: Save tuning results including all tested
        configurations and their metrics for later analysis.
        
        Args:
            results: Results dictionary from grid_search()
            filename: Output filename (default: 'hyperparameter_tuning_results.json')
            
        Returns:
            Full path to saved file
            
        Requirement: 12.4, 12.5
        """
        # Add timestamp and metadata
        output = {
            'timestamp': datetime.now().isoformat(),
            'domain': self.domain,
            'training_examples': len(self.train_data),
            'test_examples': len(self.test_data),
            'search_space': {
                'context_windows': self.context_windows,
                'rare_thresholds': self.rare_thresholds,
                'top_k_values': self.top_k_values,
            },
            **results
        }
        
        # Save to file
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        return str(filepath.absolute())


__all__ = [
    'HyperparameterTuner',
    'HyperparameterConfig',
    'TuningResult',
]
