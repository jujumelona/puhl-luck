# P87 Max Rust Speed Pass

This revision does not change model capacity from hardware information.  CPU/threads are execution only.

Changes:

- Single-prompt frozen scoring no longer uses Rayon per-row hash-map reduction by default.
- Added dense/touched accumulator for frozen CSR scoring.
- Added top-1 fast path without full score sorting.
- Batch prediction remains Rayon-parallel across prompts.
- Added Rust batch_generate_text for parallel multi-prompt generation.
- active row duplicate tracking uses u64 row IDs instead of String sets.
- Greedy generation uses top-1 scorer instead of sorting all token scores.

Goal:

- Keep P85/P86 frozen semantics.
- Reduce Python and Rust overhead for next-token decoding.
- Preserve next-token use for code, text, and creative generation.
