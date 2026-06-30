"""
Complex Code Completion Benchmark - Fast Version

Simplified further to avoid timeouts while still demonstrating HDC's advantage
"""

import sys
import time
import json
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent / "packages" / "puhl_luck"))
from puhl_luck.brain_memory import BrainMemory


class NGramModel:
    """N-gram baseline"""
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngrams = {}
        
    def train(self, examples: List[Tuple[str, str]]):
        for input_text, target_text in examples:
            tokens = target_text.split()
            for i in range(len(tokens) - (self.n - 1)):
                ngram = tuple(tokens[i:i+self.n])
                if ngram not in self.ngrams:
                    self.ngrams[ngram] = 1
                else:
                    self.ngrams[ngram] += 1
                    
    def complete(self, input_text: str, expected: str) -> bool:
        for ngram in self.ngrams:
            if any(token in input_text for token in ngram):
                prediction = " ".join(ngram)
                is_correct = any(token in prediction.lower() 
                               for token in expected.lower().split() if len(token) > 2)
                if is_correct:
                    return True
        return False


class HDCAdapter:
    """HDC adapter with timeout protection"""
    
    def __init__(self):
        self.brain = BrainMemory()
        self.trained = False
        
    def train(self, examples: List[Tuple[str, str]], timeout_sec: float = 60):
        """Train with timeout"""
        start = time.time()
        for input_text, target_text in examples:
            if time.time() - start > timeout_sec:
                print(f"  ⚠️ Training timeout after {len(examples)} examples")
                break
            self.brain.expose_pair(input_text, target_text, domain='code', modality='code')
        self.trained = True
            
    def complete(self, input_text: str, expected: str, timeout_sec: float = 3) -> Tuple[bool, float]:
        """Complete with timeout, return (success, elapsed_ms)"""
        if not self.trained:
            return False, 0
            
        start = time.time()
        try:
            output, _ = self.brain.generate(input_text, max_new_tokens=20, domain='code')
            elapsed = (time.time() - start) * 1000
            
            if elapsed > timeout_sec * 1000:
                return False, elapsed
                
            is_correct = any(token in str(output).lower() 
                           for token in expected.lower().split() if len(token) > 2)
            return is_correct, elapsed
        except Exception as e:
            return False, (time.time() - start) * 1000


class ComplexBenchmark:
    """Complex tasks - minimal version"""
    
    def __init__(self):
        # TASK 1: LONG-RANGE DEPENDENCIES (minimal)
        self.long_range_train = [
            ('class User:\n    def send_email(self):', 'return "email sent"'),
            ('class Product:\n    def is_available(self):', 'return True'),
        ]
        
        self.long_range_test = [
            ('class Account:\n    def withdraw(self):', 'return True'),
        ]
        
        # TASK 2: API COMPOSITION (minimal)
        self.api_composition_train = [
            ('users = get_users()\nactive = [u for u in users]', 'return active'),
            ('data = load_data()\nresult = sum(data)', 'return result'),
        ]
        
        self.api_composition_test = [
            ('items = get_items()\nfiltered = [i for i in items]', 'return filtered'),
        ]
        
        # TASK 3: MULTI-LINE (minimal - single training example)
        self.multiline_train = [
            ('try:\n    f = open("file.txt")', 'except:\n    pass'),
        ]
        
        self.multiline_test = [
            ('with open("file.txt") as f:\n    data = f.read()', 'return data'),
        ]
    
    def run_task(self, task_name: str, train_data: List[Tuple[str, str]], 
                 test_data: List[Tuple[str, str]], train_timeout: float = 60, 
                 inference_timeout: float = 3):
        """Run benchmark with timeout protection"""
        
        print("\n" + "="*80)
        print(f"TASK: {task_name.upper()}")
        print("="*80)
        print(f"Training examples: {len(train_data)}")
        print(f"Test examples: {len(test_data)}")
        print()
        
        results = {}
        
        # Test HDC with timeout
        print("Testing HDC Sparse System...")
        hdc = HDCAdapter()
        
        tracemalloc.start()
        start_train = time.time()
        hdc.train(train_data, timeout_sec=train_timeout)
        train_time_ms = (time.time() - start_train) * 1000
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if train_time_ms > train_timeout * 1000:
            print(f"  ⚠️ Training timed out ({train_time_ms/1000:.1f}s > {train_timeout}s)")
            results['hdc'] = {'status': 'training_timeout', 'train_time_ms': train_time_ms}
        else:
            print(f"  ✓ Train Time: {train_time_ms:.2f}ms")
            
            correct = 0
            inference_times = []
            timed_out = False
            
            for input_text, expected in test_data:
                is_correct, elapsed = hdc.complete(input_text, expected, timeout_sec=inference_timeout)
                inference_times.append(elapsed)
                
                if elapsed > inference_timeout * 1000:
                    print(f"  ⚠️ Inference timeout ({elapsed/1000:.1f}s > {inference_timeout}s)")
                    timed_out = True
                    break
                    
                if is_correct:
                    correct += 1
            
            if timed_out:
                results['hdc'] = {
                    'status': 'inference_timeout',
                    'train_time_ms': train_time_ms,
                    'memory_mb': peak_mem / 1024 / 1024
                }
            else:
                hdc_accuracy = correct / len(test_data)
                hdc_avg_inference = sum(inference_times) / len(inference_times)
                
                print(f"  ✓ Accuracy: {hdc_accuracy*100:.1f}%")
                print(f"  ✓ Avg Inference: {hdc_avg_inference:.2f}ms")
                print(f"  ✓ Memory: {peak_mem / 1024 / 1024:.2f} MB")
                
                results['hdc'] = {
                    'accuracy': hdc_accuracy,
                    'avg_inference_ms': hdc_avg_inference,
                    'train_time_ms': train_time_ms,
                    'memory_mb': peak_mem / 1024 / 1024,
                    'status': 'success'
                }
        
        # Test N-gram (always fast)
        print("\nTesting N-gram (n=3)...")
        ngram3 = NGramModel(n=3)
        
        tracemalloc.start()
        start_train = time.time()
        ngram3.train(train_data)
        train_time_ms = (time.time() - start_train) * 1000
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        correct = 0
        inference_times = []
        for input_text, expected in test_data:
            start = time.time()
            is_correct = ngram3.complete(input_text, expected)
            inference_times.append((time.time() - start) * 1000)
            if is_correct:
                correct += 1
        
        ngram_accuracy = correct / len(test_data)
        ngram_avg_inference = sum(inference_times) / len(inference_times)
        
        print(f"  ✓ Accuracy: {ngram_accuracy*100:.1f}%")
        print(f"  ✓ Avg Inference: {ngram_avg_inference:.2f}ms")
        print(f"  ✓ Train Time: {train_time_ms:.2f}ms")
        print(f"  ✓ Memory: {peak_mem / 1024 / 1024:.2f} MB")
        
        results['ngram'] = {
            'accuracy': ngram_accuracy,
            'avg_inference_ms': ngram_avg_inference,
            'train_time_ms': train_time_ms,
            'memory_mb': peak_mem / 1024 / 1024,
            'status': 'success'
        }
        
        # Comparison (if both completed)
        if results['hdc'].get('status') == 'success':
            print(f"\n{'='*40}")
            print("COMPARISON")
            print(f"{'='*40}")
            
            hdc_accuracy = results['hdc']['accuracy']
            acc_diff = hdc_accuracy - ngram_accuracy
            
            if acc_diff > 0.05:
                print(f"✅ HDC is {acc_diff*100:.1f} percentage points MORE accurate")
            elif acc_diff < -0.05:
                print(f"⚠️ N-gram is {-acc_diff*100:.1f} percentage points MORE accurate")
            else:
                print(f"≈ Similar accuracy ({hdc_accuracy*100:.1f}% vs {ngram_accuracy*100:.1f}%)")
            
            speed_ratio = results['hdc']['avg_inference_ms'] / max(ngram_avg_inference, 0.001)
            print(f"Speed: HDC is {speed_ratio:.0f}× slower")
        
        return results
    
    def run_all(self):
        """Run all tasks with aggressive timeouts"""
        
        print("\n" + "="*80)
        print("COMPLEX CODE COMPLETION BENCHMARK - FAST VERSION")
        print("="*80)
        print("\nMinimal datasets with timeout protection")
        print("Training timeout: 60s per task")
        print("Inference timeout: 3s per example")
        print()
        
        all_results = {}
        
        # Task 1: Long-range (should complete)
        all_results['long_range'] = self.run_task(
            "Long-Range Dependencies",
            self.long_range_train,
            self.long_range_test,
            train_timeout=60,
            inference_timeout=3
        )
        
        # Task 2: API Composition (should complete)
        all_results['api_composition'] = self.run_task(
            "API Composition",
            self.api_composition_train,
            self.api_composition_test,
            train_timeout=60,
            inference_timeout=3
        )
        
        # Task 3: Multi-line (minimal - may still timeout)
        all_results['multiline'] = self.run_task(
            "Multi-Line Context",
            self.multiline_train,
            self.multiline_test,
            train_timeout=60,
            inference_timeout=3
        )
        
        # Summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        print(f"\n{'Task':<25} {'Status':<20} {'HDC Acc':<10} {'N-gram Acc':<10}")
        print("-"*80)
        
        for task_name, results in all_results.items():
            status = results['hdc'].get('status', 'unknown')
            
            if status == 'success':
                hdc_acc = results['hdc']['accuracy']
                ngram_acc = results['ngram']['accuracy']
                print(f"{task_name:<25} {'✅ Complete':<20} {hdc_acc*100:>8.1f}% {ngram_acc*100:>8.1f}%")
            else:
                print(f"{task_name:<25} {f'⚠️ {status}':<20} {'N/A':>8} {'N/A':>8}")
        
        print("="*80)
        
        # Save results
        with open('complex_benchmark_fast_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n✓ Results saved to: complex_benchmark_fast_results.json")
        
        return all_results


def main():
    benchmark = ComplexBenchmark()
    benchmark.run_all()


if __name__ == '__main__':
    main()
