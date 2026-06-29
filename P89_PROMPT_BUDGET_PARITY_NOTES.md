# P89 PROMPT BUDGET PARITY FAST RUST

P88 still lost parity on code_copy because Rust next-token generated a different prompt feature set than Python frozen feature extraction.

P89 changes Rust prompt feature extraction to use the same data-scale dynamic budgets as Python `_prompt_features()` and `_active_features()`.

No hardware/CPU/thread counts enter model capacity. Rust remains execution backend only.
