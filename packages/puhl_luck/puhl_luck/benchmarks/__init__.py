"""
Benchmark utilities for PUHL-LUCK memory system.

Provides comprehensive benchmarking across multiple task types:
- Code generation
- Classification
- Pattern matching
- Question answering
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from puhl_luck.brain_memory import BrainMemory


@dataclass
class BenchmarkMetrics:
    """Metrics for a single benchmark run."""
    task_type: str
    total_tests: int
    passed: int
    failed: int
    accuracy: float
    avg_inference_time_ms: float
    total_time_ms: float
    backoff_levels: Dict[int, int]  # level -> count
    copy_gate_activations: int
    empty_outputs: int
    empty_output_rate: float
    avg_backoff_level: float
    details: List[Dict[str, Any]]


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for HDC performance testing.
    
    Covers four task categories as per Requirement 11.1:
    - Code generation
    - Classification (sentiment, topic)
    - Pattern matching (sequence completion)
    - Question answering
    
    Tracks performance metrics as per Requirements 11.2-11.6:
    - Accuracy per task type
    - Inference speed
    - Backoff statistics
    - Copy gate activations
    - Empty output rates
    """
    
    def __init__(self):
        """Initialize the benchmark suite."""
        self.results_history: List[Dict[str, Any]] = []
        
    # ========================================================================
    # BENCHMARK DATA DEFINITIONS
    # ========================================================================
    
    @staticmethod
    def _get_code_generation_data() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Get code generation training and test data.
        
        Returns:
            (training_pairs, test_pairs) - each pair is (input, expected_output)
        """
        training = [
            ("def add(a, b):", "def add(a, b):\n    return a + b"),
            ("def subtract(x, y):", "def subtract(x, y):\n    return x - y"),
            ("def multiply(a, b):", "def multiply(a, b):\n    return a * b"),
            ("def square(n):", "def square(n):\n    return n * n"),
            ("def is_even(x):", "def is_even(x):\n    return x % 2 == 0"),
            ("def is_positive(n):", "def is_positive(n):\n    return n > 0"),
            ("def max_two(a, b):", "def max_two(a, b):\n    return a if a > b else b"),
            ("def min_two(a, b):", "def min_two(a, b):\n    return a if a < b else b"),
            ("def abs_val(x):", "def abs_val(x):\n    return x if x >= 0 else -x"),
            ("def double(n):", "def double(n):\n    return n * 2"),
        ]
        
        tests = [
            ("def divide(a, b):", "return a / b"),
            ("def modulo(x, y):", "return x % y"),
            ("def cube(n):", "return n * n * n"),
            ("def is_odd(x):", "return x % 2 == 1"),
            ("def negate(n):", "return -n"),
        ]
        
        return training, tests
    
    @staticmethod
    def _get_classification_data() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Get classification training and test data (sentiment analysis).
        
        Returns:
            (training_pairs, test_pairs) - each pair is (text, label)
        """
        training = [
            ("This movie was absolutely fantastic! I loved every moment.", "positive"),
            ("Great product, highly recommend to everyone.", "positive"),
            ("Amazing experience, will definitely come back again.", "positive"),
            ("Best purchase I've made this year, very satisfied.", "positive"),
            ("Excellent service and quality, exceeded expectations.", "positive"),
            ("Terrible experience, waste of time and money.", "negative"),
            ("Very disappointed with the quality, not worth it.", "negative"),
            ("Horrible service, will never use this again.", "negative"),
            ("Worst product ever, complete garbage.", "negative"),
            ("Awful quality, broke after one use.", "negative"),
            ("It's okay, nothing special but not bad either.", "neutral"),
            ("Average product, meets basic expectations.", "neutral"),
            ("Decent enough, could be better but acceptable.", "neutral"),
            ("Standard service, nothing remarkable.", "neutral"),
            ("Mediocre experience, neither good nor bad.", "neutral"),
        ]
        
        tests = [
            ("Wonderful experience, absolutely loved it!", "positive"),
            ("Great value for money, very pleased.", "positive"),
            ("Poor quality, very unhappy with purchase.", "negative"),
            ("Disappointing product, expected much better.", "negative"),
            ("It's fine, nothing impressive.", "neutral"),
        ]
        
        return training, tests
    
    @staticmethod
    def _get_pattern_matching_data() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Get pattern matching training and test data (sequence completion).
        
        Returns:
            (training_pairs, test_pairs) - each pair is (prefix, completion)
        """
        training = [
            ("1 2 3 4", "5"),
            ("2 4 6 8", "10"),
            ("1 3 5 7", "9"),
            ("10 20 30 40", "50"),
            ("5 10 15 20", "25"),
            ("a b c d", "e"),
            ("x y z", "a"),
            ("red blue green", "yellow"),
            ("monday tuesday wednesday", "thursday"),
            ("january february march", "april"),
            ("cat dog bird", "fish"),
            ("apple banana orange", "grape"),
            ("circle square triangle", "rectangle"),
            ("one two three", "four"),
            ("north south east", "west"),
        ]
        
        tests = [
            ("3 6 9 12", "15"),
            ("100 200 300", "400"),
            ("e f g h", "i"),
            ("summer autumn winter", "spring"),
            ("car bus train", "plane"),
        ]
        
        return training, tests
    
    @staticmethod
    def _get_qa_data() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Get Q&A training and test data.
        
        Returns:
            (training_pairs, test_pairs) - each pair is (question, answer)
        """
        training = [
            ("What is the capital of France?", "Paris"),
            ("What color is the sky?", "blue"),
            ("How many days in a week?", "seven"),
            ("What is 2 + 2?", "4"),
            ("What is the opposite of hot?", "cold"),
            ("What animal says meow?", "cat"),
            ("What do we use to write?", "pen"),
            ("What season comes after winter?", "spring"),
            ("What is the first month of the year?", "January"),
            ("What shape has three sides?", "triangle"),
            ("What is frozen water called?", "ice"),
            ("What do bees make?", "honey"),
        ]
        
        tests = [
            ("What is the capital of Italy?", "Rome"),
            ("What color are trees?", "green"),
            ("How many months in a year?", "twelve"),
            ("What is 5 + 5?", "10"),
            ("What is the opposite of big?", "small"),
            ("What animal says woof?", "dog"),
        ]
        
        return training, tests
    
    # ========================================================================
    # BENCHMARK EXECUTION
    # ========================================================================
    
    def _train_brain(
        self,
        brain: BrainMemory,
        training_data: List[Tuple[str, str]],
        domain: str
    ) -> float:
        """
        Train brain on dataset.
        
        Args:
            brain: BrainMemory instance
            training_data: List of (input, target) pairs
            domain: Domain identifier
            
        Returns:
            Training time in milliseconds
        """
        start_time = time.time()
        
        for input_text, target_text in training_data:
            brain.expose_pair(
                partial=input_text,
                complete=target_text,
                domain=domain,
                modality=domain,
            )
        
        training_time_ms = (time.time() - start_time) * 1000
        return training_time_ms
    
    def _evaluate_task(
        self,
        brain: BrainMemory,
        test_data: List[Tuple[str, str]],
        domain: str,
        task_type: str,
        max_new_tokens: int = 64
    ) -> BenchmarkMetrics:
        """
        Evaluate brain on test dataset.
        
        Args:
            brain: Trained BrainMemory instance
            test_data: List of (input, expected_output) pairs
            domain: Domain identifier
            task_type: Task type name
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            BenchmarkMetrics with detailed results
        """
        total_tests = len(test_data)
        passed = 0
        failed = 0
        inference_times = []
        backoff_levels = Counter()
        copy_gate_activations = 0
        empty_outputs = 0
        details = []
        
        for input_text, expected_output in test_data:
            start_time = time.time()
            
            try:
                # Generate output with metrics
                result = brain.generate(
                    query=input_text,
                    max_new_tokens=max_new_tokens,
                    domain=domain,
                    return_metrics=True,
                )
                
                # Unpack result (could be tuple or just string)
                if isinstance(result, tuple) and len(result) == 2:
                    generated, gen_metrics = result
                else:
                    generated = result
                    gen_metrics = None
                
                inference_time_ms = (time.time() - start_time) * 1000
                inference_times.append(inference_time_ms)
                
                # Extract metrics from generation
                is_empty = len(generated.strip()) == 0
                if is_empty:
                    empty_outputs += 1
                
                # Track backoff levels if available
                if gen_metrics and hasattr(gen_metrics, 'backoff_levels'):
                    for level, count in gen_metrics.backoff_levels.items():
                        backoff_levels[level] += count
                
                # Track copy gate activations if available
                if gen_metrics and hasattr(gen_metrics, 'copy_gate_activations'):
                    copy_gate_activations += gen_metrics.copy_gate_activations
                
                # Evaluate correctness
                # For classification/QA: check if expected is in generated (case-insensitive)
                # For code/patterns: check for key tokens or exact match
                correct = False
                
                if task_type in ["classification", "qa"]:
                    # Case-insensitive substring match
                    correct = expected_output.lower() in generated.lower()
                else:
                    # Token-based matching for code and patterns
                    expected_tokens = set(expected_output.lower().split())
                    generated_tokens = set(generated.lower().split())
                    overlap = len(expected_tokens & generated_tokens)
                    correct = overlap >= len(expected_tokens) * 0.5  # 50% token overlap
                
                if correct:
                    passed += 1
                else:
                    failed += 1
                
                details.append({
                    "input": input_text,
                    "expected": expected_output,
                    "generated": generated,
                    "correct": correct,
                    "inference_time_ms": inference_time_ms,
                    "empty": is_empty,
                })
                
            except Exception as e:
                inference_time_ms = (time.time() - start_time) * 1000
                inference_times.append(inference_time_ms)
                failed += 1
                empty_outputs += 1
                
                details.append({
                    "input": input_text,
                    "expected": expected_output,
                    "generated": "",
                    "correct": False,
                    "inference_time_ms": inference_time_ms,
                    "empty": True,
                    "error": str(e),
                })
        
        # Calculate metrics
        accuracy = passed / total_tests if total_tests > 0 else 0.0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
        total_time = sum(inference_times)
        empty_output_rate = empty_outputs / total_tests if total_tests > 0 else 0.0
        
        # Calculate average backoff level
        total_backoffs = sum(backoff_levels.values())
        avg_backoff = (
            sum(level * count for level, count in backoff_levels.items()) / total_backoffs
            if total_backoffs > 0
            else 0.0
        )
        
        return BenchmarkMetrics(
            task_type=task_type,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            accuracy=accuracy,
            avg_inference_time_ms=avg_inference_time,
            total_time_ms=total_time,
            backoff_levels=dict(backoff_levels),
            copy_gate_activations=copy_gate_activations,
            empty_outputs=empty_outputs,
            empty_output_rate=empty_output_rate,
            avg_backoff_level=avg_backoff,
            details=details,
        )
    
    def run_all_benchmarks(
        self,
        tasks: Optional[List[str]] = None,
        max_new_tokens: int = 64,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite across all task types.
        
        As per Requirement 11.1: Covers code generation, classification,
        pattern matching, and Q&A tasks.
        
        Args:
            tasks: List of task types to run. If None, runs all tasks.
                   Options: ['code', 'classification', 'pattern', 'qa']
            max_new_tokens: Maximum tokens to generate per query
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing:
            - task_results: Dict[task_type, BenchmarkMetrics]
            - aggregate_metrics: Overall statistics
            - timestamp: ISO format timestamp
            
        Requirements: 11.1, 11.2, 11.3, 11.4
        """
        if tasks is None:
            tasks = ['code', 'classification', 'pattern', 'qa']
        
        if verbose:
            print("=" * 70)
            print("HDC PERFORMANCE BENCHMARK SUITE")
            print("=" * 70)
            print(f"Tasks: {', '.join(tasks)}")
            print(f"Max new tokens: {max_new_tokens}")
            print()
        
        task_results = {}
        total_start_time = time.time()
        
        # ====================================================================
        # CODE GENERATION BENCHMARK
        # ====================================================================
        if 'code' in tasks:
            if verbose:
                print("[1/4] Code Generation Benchmark")
                print("-" * 70)
            
            brain = BrainMemory()
            training_data, test_data = self._get_code_generation_data()
            
            if verbose:
                print(f"Training on {len(training_data)} examples...")
            train_time = self._train_brain(brain, training_data, 'code')
            
            if verbose:
                print(f"Training time: {train_time:.2f}ms")
                print(f"Evaluating on {len(test_data)} test cases...")
            
            metrics = self._evaluate_task(brain, test_data, 'code', 'code', max_new_tokens)
            task_results['code'] = metrics
            
            if verbose:
                print(f"Accuracy: {metrics.accuracy * 100:.1f}%")
                print(f"Avg inference: {metrics.avg_inference_time_ms:.2f}ms")
                print(f"Empty outputs: {metrics.empty_outputs}/{metrics.total_tests}")
                print()
        
        # ====================================================================
        # CLASSIFICATION BENCHMARK
        # ====================================================================
        if 'classification' in tasks:
            if verbose:
                print("[2/4] Classification (Sentiment) Benchmark")
                print("-" * 70)
            
            brain = BrainMemory()
            training_data, test_data = self._get_classification_data()
            
            if verbose:
                print(f"Training on {len(training_data)} examples...")
            train_time = self._train_brain(brain, training_data, 'sentiment')
            
            if verbose:
                print(f"Training time: {train_time:.2f}ms")
                print(f"Evaluating on {len(test_data)} test cases...")
            
            metrics = self._evaluate_task(brain, test_data, 'sentiment', 'classification', max_new_tokens)
            task_results['classification'] = metrics
            
            if verbose:
                print(f"Accuracy: {metrics.accuracy * 100:.1f}%")
                print(f"Avg inference: {metrics.avg_inference_time_ms:.2f}ms")
                print(f"Empty outputs: {metrics.empty_outputs}/{metrics.total_tests}")
                print()
        
        # ====================================================================
        # PATTERN MATCHING BENCHMARK
        # ====================================================================
        if 'pattern' in tasks:
            if verbose:
                print("[3/4] Pattern Matching (Sequence Completion) Benchmark")
                print("-" * 70)
            
            brain = BrainMemory()
            training_data, test_data = self._get_pattern_matching_data()
            
            if verbose:
                print(f"Training on {len(training_data)} examples...")
            train_time = self._train_brain(brain, training_data, 'pattern')
            
            if verbose:
                print(f"Training time: {train_time:.2f}ms")
                print(f"Evaluating on {len(test_data)} test cases...")
            
            metrics = self._evaluate_task(brain, test_data, 'pattern', 'pattern', max_new_tokens)
            task_results['pattern'] = metrics
            
            if verbose:
                print(f"Accuracy: {metrics.accuracy * 100:.1f}%")
                print(f"Avg inference: {metrics.avg_inference_time_ms:.2f}ms")
                print(f"Empty outputs: {metrics.empty_outputs}/{metrics.total_tests}")
                print()
        
        # ====================================================================
        # Q&A BENCHMARK
        # ====================================================================
        if 'qa' in tasks:
            if verbose:
                print("[4/4] Question Answering Benchmark")
                print("-" * 70)
            
            brain = BrainMemory()
            training_data, test_data = self._get_qa_data()
            
            if verbose:
                print(f"Training on {len(training_data)} examples...")
            train_time = self._train_brain(brain, training_data, 'qa')
            
            if verbose:
                print(f"Training time: {train_time:.2f}ms")
                print(f"Evaluating on {len(test_data)} test cases...")
            
            metrics = self._evaluate_task(brain, test_data, 'qa', 'qa', max_new_tokens)
            task_results['qa'] = metrics
            
            if verbose:
                print(f"Accuracy: {metrics.accuracy * 100:.1f}%")
                print(f"Avg inference: {metrics.avg_inference_time_ms:.2f}ms")
                print(f"Empty outputs: {metrics.empty_outputs}/{metrics.total_tests}")
                print()
        
        # ====================================================================
        # AGGREGATE METRICS
        # ====================================================================
        total_elapsed = (time.time() - total_start_time) * 1000
        
        # Calculate aggregate statistics
        total_tests = sum(m.total_tests for m in task_results.values())
        total_passed = sum(m.passed for m in task_results.values())
        overall_accuracy = total_passed / total_tests if total_tests > 0 else 0.0
        
        avg_inference_time = (
            sum(m.avg_inference_time_ms * m.total_tests for m in task_results.values()) / total_tests
            if total_tests > 0
            else 0.0
        )
        
        total_empty_outputs = sum(m.empty_outputs for m in task_results.values())
        overall_empty_rate = total_empty_outputs / total_tests if total_tests > 0 else 0.0
        
        # Aggregate backoff statistics
        total_backoff_levels = Counter()
        for metrics in task_results.values():
            for level, count in metrics.backoff_levels.items():
                total_backoff_levels[level] += count
        
        total_copy_activations = sum(m.copy_gate_activations for m in task_results.values())
        
        avg_backoff = (
            sum(level * count for level, count in total_backoff_levels.items())
            / sum(total_backoff_levels.values())
            if sum(total_backoff_levels.values()) > 0
            else 0.0
        )
        
        aggregate_metrics = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_tests - total_passed,
            "overall_accuracy": overall_accuracy,
            "avg_inference_time_ms": avg_inference_time,
            "total_execution_time_ms": total_elapsed,
            "total_empty_outputs": total_empty_outputs,
            "overall_empty_rate": overall_empty_rate,
            "total_backoff_levels": dict(total_backoff_levels),
            "avg_backoff_level": avg_backoff,
            "total_copy_gate_activations": total_copy_activations,
        }
        
        if verbose:
            print("=" * 70)
            print("AGGREGATE RESULTS")
            print("=" * 70)
            print(f"Total tests: {total_tests}")
            print(f"Passed: {total_passed}")
            print(f"Failed: {total_tests - total_passed}")
            print(f"Overall accuracy: {overall_accuracy * 100:.1f}%")
            print(f"Avg inference time: {avg_inference_time:.2f}ms")
            print(f"Total execution time: {total_elapsed / 1000:.2f}s")
            print(f"Empty output rate: {overall_empty_rate * 100:.1f}%")
            print(f"Avg backoff level: {avg_backoff:.2f}")
            print(f"Copy gate activations: {total_copy_activations}")
            print("=" * 70)
        
        # Compile results
        results = {
            "timestamp": datetime.now().isoformat(),
            "task_results": {
                task: asdict(metrics) for task, metrics in task_results.items()
            },
            "aggregate_metrics": aggregate_metrics,
        }
        
        return results
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Save benchmark results to JSON file with timestamp.
        
        As per Requirement 11.5: Save results in JSON format with timestamps
        for historical comparison.
        
        Args:
            results: Results dictionary from run_all_benchmarks()
            filename: Optional filename. If None, generates timestamped filename.
            
        Returns:
            Path to saved file
            
        Requirement: 11.5
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        # Ensure filename is a Path object
        filepath = Path(filename)
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return str(filepath)


__all__ = [
    "BenchmarkSuite",
    "BenchmarkMetrics",
]

# Import hyperparameter tuner classes for easy access
from puhl_luck.benchmarks.hyperparameter_tuner import (
    HyperparameterTuner,
    HyperparameterConfig,
    TuningResult,
)
