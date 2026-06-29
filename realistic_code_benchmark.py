"""
Realistic Code Completion Benchmark
Compares HDC against actual code completion tools (N-gram, Jedi)
"""

import sys
import time
import psutil
import os
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any
import json

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'puhl_luck'))

from puhl_luck.brain_memory import BrainMemory


# ============================================================================
# BASELINE: N-GRAM MODEL
# ============================================================================

class NGramModel:
    """Simple n-gram baseline for code completion"""
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngrams: Dict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        
    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization"""
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
        
    def train(self, examples: List[Tuple[str, str]]):
        """Train on input-target pairs"""
        for input_text, target_text in examples:
            tokens = self.tokenize(input_text + ' ' + target_text)
            for i in range(len(tokens) - self.n):
                context = tuple(tokens[i:i+self.n])
                next_token = tokens[i+self.n]
                self.ngrams[context][next_token] += 1
                
    def complete(self, input_text: str, max_tokens: int = 10) -> str:
        """Generate completion"""
        tokens = self.tokenize(input_text)
        output = []
        
        for _ in range(max_tokens):
            if len(tokens) < self.n:
                break
                
            context = tuple(tokens[-self.n:])
            if context not in self.ngrams:
                break
                
            # Get most common next token
            candidates = self.ngrams[context]
            if not candidates:
                break
                
            next_token = candidates.most_common(1)[0][0]
            output.append(next_token)
            tokens.append(next_token)
            
        return ' '.join(output)


# ============================================================================
# JEDI ADAPTER
# ============================================================================

class JediAdapter:
    """Adapter for Jedi Python completion"""
    
    def __init__(self):
        try:
            import jedi
            self.jedi = jedi
            self.available = True
        except ImportError:
            print("Warning: Jedi not available (pip install jedi)")
            self.available = False
            
    def train(self, examples: List[Tuple[str, str]]):
        """Jedi doesn't need training"""
        pass
        
    def complete(self, input_text: str, max_tokens: int = 10) -> str:
        """Get completion from Jedi"""
        if not self.available:
            return ""
            
        try:
            # Get completions at end of input
            script = self.jedi.Script(input_text)
            completions = script.complete(len(input_text.split('\n')), len(input_text.split('\n')[-1]))
            
            if completions:
                # Return first completion
                return completions[0].name
            return ""
        except Exception as e:
            return ""


# ============================================================================
# HDC ADAPTER
# ============================================================================

class HDCAdapter:
    """Adapter for HDC Sparse System"""
    
    def __init__(self):
        self.brain = BrainMemory()
        
    def train(self, examples: List[Tuple[str, str]]):
        """Train HDC on examples"""
        for input_text, target_text in examples:
            self.brain.expose_pair(input_text, target_text, domain='code', modality='code')
            
    def complete(self, input_text: str, max_tokens: int = 10) -> str:
        """Generate completion"""
        try:
            result, _ = self.brain.generate(input_text, max_new_tokens=max_tokens, domain='code')
            return str(result) if result else ""
        except Exception as e:
            return ""


# ============================================================================
# BENCHMARK ORCHESTRATOR
# ============================================================================

class BenchmarkOrchestrator:
    """Runs benchmark and collects metrics"""
    
    def __init__(self):
        self.results = {}
        
    def measure_metrics(self, tool, tool_name: str, train_data: List[Tuple[str, str]], 
                       test_data: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Measure all metrics for a tool"""
        
        # Measure training time and memory
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        train_start = time.time()
        tool.train(train_data)
        train_time = time.time() - train_start
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_mb = mem_after - mem_before
        
        # Measure inference latency and accuracy
        latencies = []
        correct = 0
        total = len(test_data)
        
        for input_text, expected_output in test_data:
            start = time.time()
            prediction = tool.complete(input_text, max_tokens=20)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
            # Check accuracy (token overlap)
            pred_tokens = set(prediction.split())
            expected_tokens = set(expected_output.split())
            if pred_tokens & expected_tokens:  # Any overlap counts as correct
                correct += 1
                
        accuracy = correct / total if total > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            'tool': tool_name,
            'train_time_s': train_time,
            'memory_mb': memory_mb,
            'avg_latency_ms': avg_latency,
            'accuracy': accuracy,
            'test_cases': total
        }
        
    def run(self, train_data: List[Tuple[str, str]], test_data: List[Tuple[str, str]]):
        """Run benchmark on all tools"""
        
        print("="*70)
        print("REALISTIC CODE COMPLETION BENCHMARK")
        print("="*70)
        print(f"Training examples: {len(train_data)}")
        print(f"Test examples: {len(test_data)}")
        print()
        
        tools = [
            (NGramModel(n=3), "N-gram (n=3)"),
            (NGramModel(n=5), "N-gram (n=5)"),
            (JediAdapter(), "Jedi (Python)"),
            (HDCAdapter(), "HDC Sparse System"),
        ]
        
        results = []
        for tool, name in tools:
            print(f"Testing {name}...")
            try:
                metrics = self.measure_metrics(tool, name, train_data, test_data)
                results.append(metrics)
                print(f"  ✓ Completed: {metrics['accuracy']*100:.1f}% accuracy, "
                      f"{metrics['avg_latency_ms']:.1f}ms latency")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                
        self.results = results
        return results


# ============================================================================
# METRICS REPORTER
# ============================================================================

class MetricsReporter:
    """Generate comparison reports"""
    
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        
    def print_table(self):
        """Print comparison table"""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        print(f"{'Tool':<25} {'Accuracy':<12} {'Latency (ms)':<15} {'Memory (MB)':<15} {'Train (s)':<12}")
        print("-"*70)
        
        for r in self.results:
            print(f"{r['tool']:<25} {r['accuracy']*100:>10.1f}% {r['avg_latency_ms']:>14.1f} "
                  f"{r['memory_mb']:>14.1f} {r['train_time_s']:>11.2f}")
                  
        print("="*70)
        
    def export_json(self, filename: str):
        """Export results to JSON"""
        with open(filename, 'w') as f:
            json.dump({
                'results': self.results,
                'summary': {
                    'best_accuracy': max(self.results, key=lambda x: x['accuracy'])['tool'],
                    'best_latency': min(self.results, key=lambda x: x['avg_latency_ms'])['tool'],
                    'best_memory': min(self.results, key=lambda x: x['memory_mb'])['tool'],
                }
            }, f, indent=2)
        print(f"\n✓ Results exported to {filename}")


# ============================================================================
# PYTHON CODE CORPUS
# ============================================================================

def get_python_corpus() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Generate Python code corpus"""
    
    # Training examples (10)
    train_data = [
        ("def add(a, b):", "return a + b"),
        ("def subtract(a, b):", "return a - b"),
        ("def multiply(a, b):", "return a * b"),
        ("def divide(a, b):", "return a / b"),
        ("def square(x):", "return x * x"),
        ("def is_even(n):", "return n % 2 == 0"),
        ("def get_max(a, b):", "return a if a > b else b"),
        ("def factorial(n):", "return 1 if n == 0 else n * factorial(n-1)"),
        ("class Person:\n    def __init__(self, name):", "self.name = name"),
        ("for i in range(10):", "print(i)"),
    ]
    
    # Test examples (20)
    test_data = [
        ("def power(x, n):", "return x ** n"),
        ("def is_odd(n):", "return n % 2 != 0"),
        ("def get_min(a, b):", "return a if a < b else b"),
        ("def absolute(x):", "return x if x >= 0 else -x"),
        ("def cube(x):", "return x * x * x"),
        ("class Animal:\n    def __init__(self, name):", "self.name = name"),
        ("def sum_list(lst):", "return sum(lst)"),
        ("def reverse_string(s):", "return s[::-1]"),
        ("def count_vowels(s):", "return sum(1 for c in s if c in 'aeiou')"),
        ("def fibonacci(n):", "return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
        ("while True:", "break"),
        ("if x > 0:", "print('positive')"),
        ("def greet(name):", "return f'Hello, {name}!'"),
        ("def double(x):", "return x * 2"),
        ("def half(x):", "return x / 2"),
        ("def increment(x):", "return x + 1"),
        ("def decrement(x):", "return x - 1"),
        ("def negate(x):", "return -x"),
        ("def is_positive(x):", "return x > 0"),
        ("def is_zero(x):", "return x == 0"),
    ]
    
    return train_data, test_data


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Get corpus
    train_data, test_data = get_python_corpus()
    
    # Run benchmark
    orchestrator = BenchmarkOrchestrator()
    results = orchestrator.run(train_data, test_data)
    
    # Generate report
    reporter = MetricsReporter(results)
    reporter.print_table()
    reporter.export_json('realistic_benchmark_results.json')
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("This benchmark compares HDC against tools that perform similar tasks:")
    print("- N-gram: Simple statistical baseline")
    print("- Jedi: Real-world Python completion engine")
    print("- HDC: Hyperdimensional Computing sparse system")
    print()
    print("Unlike GPT-2/CodeGen, these tools:")
    print("✓ Run locally without GPU")
    print("✓ Work offline")
    print("✓ Have minimal memory footprint")
    print("✓ Target code completion specifically")
    print("="*70)


if __name__ == '__main__':
    main()
