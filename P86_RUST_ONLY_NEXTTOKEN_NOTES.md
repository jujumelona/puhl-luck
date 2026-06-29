# P86 Rust-only next-token engine

Adds a frozen inference engine that keeps the next-token formulation but moves
runtime tokenization, prompt feature extraction, sparse CSR scoring, copy-token
resolution, autoregressive generation, and batch prediction into Rust when the
extension is available.

This is not a model-cap patch. It changes the runtime representation:
training builder -> freeze -> compact CSR artifact -> RustFrozenNextEngine.

The Python path remains only as fallback and for training/evaluation orchestration.
