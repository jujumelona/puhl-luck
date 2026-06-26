# PUHL-LUCK Brain Memory

This package installs the `puhl-luck` command.

Paper / archive link: https://zenodo.org/records/20851529

Install from the repository root:

```bat
python -m pip install -e packages\puhl_luck
```

Training saves learned files separately from the code:

```text
brain_data/
  brain_memory.pkl
  brain_rank_micro.pmr
  brain_meta.json
  interaction_log.jsonl
```

Common commands:

```bat
puhl-luck
puhl-luck brain_data
puhl-luck t data
puhl-luck s data
puhl-luck c --no-learn
puhl-luck q "question" "A,B,C,D"
puhl-luck b ask.tsv --quiet
puhl-luck recall "query text"
puhl-luck status
```

See the repository root `README.md` for the full user guide.
