"""
Complex Code Completion Benchmark

Tests HDC vs N-gram on HARDER tasks where HDC should excel:
1. Long-range dependencies (>5 token context)
2. Multi-line code blocks
3. API composition patterns
4. Cross-function references
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
    """N-gram baseline (same as before)"""
    
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
    """HDC adapter (proper API)"""
    
    def __init__(self):
        self.brain = BrainMemory()
        
    def train(self, examples: List[Tuple[str, str]]):
        for input_text, target_text in examples:
            self.brain.expose_pair(input_text, target_text, 
                                  domain='code', modality='code')
            
    def complete(self, input_text: str, expected: str) -> bool:
        try:
            output, _ = self.brain.generate(input_text, max_new_tokens=50, domain='code')
            return any(token in str(output).lower() 
                      for token in expected.lower().split() if len(token) > 2)
        except:
            return False


class ComplexBenchmark:
    """Complex tasks where HDC should excel"""
    
    def __init__(self):
        # TASK 1: LONG-RANGE DEPENDENCIES (>10 token context) - SIMPLIFIED
        self.long_range_train = [
            ('class User:\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email\n    def send_email(self):',
             'return f"Sending email to {self.email}"'),
            
            ('class Product:\n    def __init__(self, name, price, stock):\n        self.name = name\n        self.price = price\n        self.stock = stock\n    def is_available(self):',
             'return self.stock > 0'),
            
            ('class Order:\n    def __init__(self, items, total):\n        self.items = items\n        self.total = total\n    def calculate_tax(self):',
             'return self.total * 0.1'),
        ]
        
        self.long_range_test = [
            ('class Customer:\n    def __init__(self, name, balance):\n        self.name = name\n        self.balance = balance\n    def can_afford(self, amount):',
             'return self.balance >= amount'),
            
            ('class Account:\n    def __init__(self, owner, balance):\n        self.owner = owner\n        self.balance = balance\n    def withdraw(self, amount):',
             'if self.balance >= amount:\n            self.balance -= amount\n            return True\n        return False'),
        ]
        
        # TASK 2: API COMPOSITION (chained method calls) - SIMPLIFIED
        self.api_composition_train = [
            ('users = get_users()\nactive_users = [u for u in users if u.active]',
             'user_count = len(active_users)\nreturn user_count'),
            
            ('data = load_data()\nfiltered = [x for x in data if x > 0]',
             'result = sum(filtered)\nreturn result'),
            
            ('items = fetch_items()\nsorted_items = sorted(items, key=lambda x: x.price)',
             'top_items = sorted_items[:10]\nreturn top_items'),
        ]
        
        self.api_composition_test = [
            ('products = get_products()\navailable = [p for p in products if p.stock > 0]',
             'cheap_products = [p for p in available if p.price < 100]\nreturn cheap_products'),
            
            ('orders = fetch_orders()\ncompleted = [o for o in orders if o.status == "done"]',
             'total = sum(o.amount for o in completed)\nreturn total'),
        ]
        
        # TASK 3: MULTI-LINE CONTEXT - SIMPLIFIED
        self.multiline_train = [
            ('try:\n    file = open("data.txt")\n    content = file.read()',
             'except FileNotFoundError:\n    return None\nfinally:\n    file.close()'),
            
            ('if user.is_admin:\n    if request.method == "POST":\n        form = Form(request.POST)',
             'if form.is_valid():\n            form.save()\n            return redirect("success")'),
            
            ('for item in items:\n    if item.active:\n        result = process(item)',
             'if result:\n            output.append(result)\nreturn output'),
        ]
        
        self.multiline_test = [
            ('with open("file.txt") as f:\n    data = f.read()\n    lines = data.split("\\n")',
             'for line in lines:\n    if line.strip():\n        process_line(line)\nreturn True'),
            
            ('async def fetch(url):\n    async with session.get(url) as response:\n        if response.status == 200:',
             'data = await response.json()\n            return data\n        return None'),
        ]
    
    def run_task(self, task_name: str, train_data: List[Tuple[str, str]], 
                 test_data: List[Tuple[str, str]]):
        """Run benchmark on a specific task"""
        
        print("\n" + "="*80)
        print(f"TASK: {task_name.upper()}")
        print("="*80)
        print(f"Training examples: {len(train_data)}")
        print(f"Test examples: {len(test_data)}")
        print()
        
        results = {}
        
        # Test HDC
        print("Testing HDC Sparse System...")
        hdc = HDCAdapter()
        
        tracemalloc.start()
        start_train = time.time()
        hdc.train(train_data)
        train_time_ms = (time.time() - start_train) * 1000
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        correct = 0
        inference_times = []
        for input_text, expected in test_data:
            start = time.time()
            is_correct = hdc.complete(input_text, expected)
            inference_times.append((time.time() - start) * 1000)
            if is_correct:
                correct += 1
        
        hdc_accuracy = correct / len(test_data)
        hdc_avg_inference = sum(inference_times) / len(inference_times)
        
        print(f"  ✓ Accuracy: {hdc_accuracy*100:.1f}%")
        print(f"  ✓ Avg Inference: {hdc_avg_inference:.2f}ms")
        print(f"  ✓ Train Time: {train_time_ms:.2f}ms")
        print(f"  ✓ Memory: {peak_mem / 1024 / 1024:.2f} MB")
        
        results['hdc'] = {
            'accuracy': hdc_accuracy,
            'avg_inference_ms': hdc_avg_inference,
            'train_time_ms': train_time_ms,
            'memory_mb': peak_mem / 1024 / 1024
        }
        
        # Test N-gram (n=3)
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
            'memory_mb': peak_mem / 1024 / 1024
        }
        
        # Comparison
        print(f"\n{'='*40}")
        print("COMPARISON")
        print(f"{'='*40}")
        
        acc_diff = hdc_accuracy - ngram_accuracy
        if acc_diff > 0.05:
            print(f"✅ HDC is {acc_diff*100:.1f} percentage points MORE accurate")
        elif acc_diff < -0.05:
            print(f"⚠️ N-gram is {-acc_diff*100:.1f} percentage points MORE accurate")
        else:
            print(f"≈ Similar accuracy ({hdc_accuracy*100:.1f}% vs {ngram_accuracy*100:.1f}%)")
        
        speed_ratio = hdc_avg_inference / max(ngram_avg_inference, 0.001)
        print(f"Speed: HDC is {speed_ratio:.0f}× slower")
        
        return results
    
    def run_all(self):
        """Run all complex tasks"""
        
        print("\n" + "="*80)
        print("COMPLEX CODE COMPLETION BENCHMARK")
        print("="*80)
        print("\nTesting HDC on HARDER tasks where it should excel:")
        print("  1. Long-range dependencies (>10 token context)")
        print("  2. API composition (chained method calls)")
        print("  3. Multi-line context (requires understanding previous lines)")
        print()
        
        all_results = {}
        
        # Task 1: Long-range
        all_results['long_range'] = self.run_task(
            "Long-Range Dependencies",
            self.long_range_train,
            self.long_range_test
        )
        
        # Task 2: API Composition
        all_results['api_composition'] = self.run_task(
            "API Composition",
            self.api_composition_train,
            self.api_composition_test
        )
        
        # Task 3: Multi-line
        all_results['multiline'] = self.run_task(
            "Multi-Line Context",
            self.multiline_train,
            self.multiline_test
        )
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        print(f"\n{'Task':<25} {'HDC Acc':<12} {'N-gram Acc':<12} {'Winner':<12}")
        print("-"*80)
        
        for task_name, results in all_results.items():
            hdc_acc = results['hdc']['accuracy']
            ngram_acc = results['ngram']['accuracy']
            
            if hdc_acc > ngram_acc + 0.05:
                winner = "HDC ✅"
            elif ngram_acc > hdc_acc + 0.05:
                winner = "N-gram ✅"
            else:
                winner = "Tie ≈"
            
            print(f"{task_name:<25} {hdc_acc*100:>10.1f}% {ngram_acc*100:>10.1f}% {winner:<12}")
        
        print("="*80)
        
        # Save results
        with open('complex_benchmark_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n✓ Results saved to: complex_benchmark_results.json")
        
        return all_results


def main():
    benchmark = ComplexBenchmark()
    benchmark.run_all()


if __name__ == '__main__':
    main()
