# Competitive Benchmark Results

Comparison of edge AI classification systems


| Metric | puhl-luck (HDC) | scikit-learn | TensorFlow Lite |
|--------|----------------|--------------|-----------------|
| Accuracy | 33.33% | 33.33% | 90% |
| Training Time | 1918.76ms | 26.19ms | 60000ms |
| Inference Time | 178.72ms | 1.97ms | 10ms |
| Model Size | 2.99MB | 0.38MB | 5MB |
| Peak Memory | 36.53MB | 171.63MB | 30MB |
| GPU Required | No | No | No |
| Incremental Learning | Yes | No | No |

## Notes

- **puhl-luck**: CPU-only HDC system with fast incremental learning
- **scikit-learn**: Traditional ML, widely used but no incremental learning
- **TensorFlow Lite**: Neural networks optimized for edge, high accuracy but slow training

## Key Takeaways

**puhl-luck advantages**:
