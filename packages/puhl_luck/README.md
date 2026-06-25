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
puhl-luck train --data data --epochs 1
puhl-luck train --data data --micro-hash-bits 32
puhl-luck start
puhl-luck ask "question" --choices "A,B,C,D" --engine micro
puhl-luck ask "question" --choices "A,B,C,D" --engine micro --device gpu
puhl-luck ask-batch --input ask.tsv --format tsv --engine micro --quiet
puhl-luck chat --no-learn
puhl-luck recall "query text"
puhl-luck status
```

See the repository root `README.md` for the full user guide.
