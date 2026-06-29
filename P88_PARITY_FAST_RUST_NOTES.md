# P88 Parity Fast Rust Notes

Goal: keep P87 speed-oriented Rust next-token runtime, but restore parity with the
P85 Python frozen feature path.

Root cause from P87 result:
- Rust next-token path was active and fast, but code_copy accuracy dropped.
- Python frozen path used data-scale feature budgets (`_data_feature_budget`) while
  Rust active feature extraction activated more prompt rows.
- Extra Rust rows pushed operator tokens such as `+`/`-` over learned copy tokens
  (`[COPYi]`) in code_copy prompts.

Fix:
- Store Python data-scale metadata in frozen artifact meta:
  `data_scale_events`, `data_scale_features`, `data_scale_rows`, `data_scale_vocab`.
- Pass that metadata to `RustFrozenNextEngine` through reserved `__p88_*` entries.
- Rust active row generation now uses the same data-derived budget formula as
  Python `_data_feature_budget`.
- Rust uses budgeted P0/P1/P2/P3/PB feature activation instead of activating every
  prompt row.
- Single prompt remains dense/touched top-1 fast path; batch remains Rayon parallel.

No hardware/CPU/thread information is used for model capacity.
