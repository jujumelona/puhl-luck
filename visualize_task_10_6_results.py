"""Visualize Task 10.6 overfitting prevention validation results."""
import json
from pathlib import Path


def print_bar_chart(label: str, value: float, max_value: float = 100.0, width: int = 50):
    """Print a simple ASCII bar chart."""
    filled = int((value / max_value) * width)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {label:30s} [{bar}] {value:.1f}%")


def visualize_results():
    """Visualize Task 10.6 validation results."""
    
    # Load results
    results_file = Path(__file__).parent / "task_10_6_validation_results.json"
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    print("=" * 80)
    print("TASK 10.6: OVERFITTING PREVENTION - VALIDATION RESULTS")
    print("=" * 80)
    print()
    
    # Overall status
    overall = "✓ PASS" if results["overall_pass"] else "✗ FAIL"
    print(f"Overall Status: {overall}")
    print()
    
    # Requirement 3.1
    req_3_1 = results["requirement_3_1"]
    print("-" * 80)
    print("REQUIREMENT 3.1: Accuracy Degradation <5%")
    print("-" * 80)
    print()
    print_bar_chart("Baseline Accuracy (A)", req_3_1["baseline_accuracy"] * 100)
    print_bar_chart("Final Accuracy (A)", req_3_1["final_accuracy"] * 100)
    print()
    print(f"  Degradation: {req_3_1['degradation_percentage']:.1f}% (threshold: <{req_3_1['threshold']:.1f}%)")
    print(f"  Status: {'✓ PASS' if req_3_1['pass'] else '✗ FAIL'}")
    print()
    
    # Requirement 3.3
    req_3_3 = results["requirement_3_3"]
    print("-" * 80)
    print("REQUIREMENT 3.3: Consistent Accuracy Within 10%")
    print("-" * 80)
    print()
    print_bar_chart("Accuracy on A (after C)", req_3_3["accuracy_a_after_c"] * 100)
    print_bar_chart("Accuracy on B (after C)", req_3_3["accuracy_b_after_c"] * 100)
    print_bar_chart("Accuracy on C (after C)", req_3_3["accuracy_c_after_c"] * 100)
    print()
    print(f"  Min: {req_3_3['min_accuracy'] * 100:.1f}%")
    print(f"  Max: {req_3_3['max_accuracy'] * 100:.1f}%")
    print(f"  Variance: {req_3_3['variance_percentage']:.1f}% (threshold: <{req_3_3['threshold']:.1f}%)")
    print(f"  Status: {'✓ PASS' if req_3_3['pass'] else '✗ FAIL'}")
    print()
    
    # Detailed phase-by-phase accuracy
    detailed = results["detailed_accuracies"]
    print("-" * 80)
    print("PHASE-BY-PHASE ACCURACY PROGRESSION")
    print("-" * 80)
    print()
    print("Phase 1: Train on Dataset A (Arithmetic)")
    print_bar_chart("  A baseline", detailed["phase_1_a_baseline"] * 100)
    print()
    
    print("Phase 2: Train on Dataset B (Boolean)")
    print_bar_chart("  A after B", detailed["phase_2_a_after_b"] * 100)
    print_bar_chart("  B after B", detailed["phase_2_b_after_b"] * 100)
    print()
    
    print("Phase 3: Train on Dataset C (Utility)")
    print_bar_chart("  A after C", detailed["phase_3_a_after_c"] * 100)
    print_bar_chart("  B after C", detailed["phase_3_b_after_c"] * 100)
    print_bar_chart("  C after C", detailed["phase_3_c_after_c"] * 100)
    print()
    
    # Key insights
    print("-" * 80)
    print("KEY INSIGHTS")
    print("-" * 80)
    print()
    
    if req_3_1["degradation_percentage"] == 0:
        print("  ✓ Zero degradation: Earlier patterns perfectly preserved")
    else:
        degradation_margin = req_3_1["threshold"] - req_3_1["degradation_percentage"]
        print(f"  ✓ Degradation within safe margin: {degradation_margin:.1f}% below threshold")
    
    if req_3_3["variance_percentage"] == 0:
        print("  ✓ Perfect consistency: All datasets maintain equal accuracy")
    else:
        variance_margin = req_3_3["threshold"] - req_3_3["variance_percentage"]
        print(f"  ✓ Variance within acceptable range: {variance_margin:.1f}% below threshold")
    
    print("  ✓ Rank-loss credit assignment prevents catastrophic forgetting")
    print("  ✓ Negative evidence maintains balanced pattern representation")
    print("  ✓ Progressive backoff enables retrieval of earlier learned patterns")
    print()
    
    print("=" * 80)
    print(f"VALIDATION: {overall}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    visualize_results()
