# PUHL-LUCK

PUHL-LUCK is a local exposure-learning memory engine. It learns from text,
images, audio, and arbitrary files, then saves the learned brain outside the
code.

Paper / archive link: https://zenodo.org/records/20851529

## Install

Open CMD or PowerShell in this repository:

```bat
python -m pip install -e packages\puhl_luck
```

Check it:

```bat
puhl-luck --help
```

## Fast Start

Put files in `data/`, then run:

```bat
puhl-luck
```

Default behavior:

- If `brain_data/brain_memory.pkl` already exists, it opens chat.
- If no learned brain exists but `data/` exists, it trains `data/` and opens chat.
- Chat learning is on by default, so conversation turns are learned too.

Load a learned brain folder or learned file directly:

```bat
puhl-luck brain_data
puhl-luck brain_data\brain_memory.pkl
```

Ask a choice question without writing `ask`:

```bat
puhl-luck "machine learning language" "CSS,Python"
```

## Common Commands

Train:

```bat
puhl-luck t
puhl-luck t data
puhl-luck t data\notes.txt
```

Train and open chat:

```bat
puhl-luck s
puhl-luck s data
```

Open chat:

```bat
puhl-luck c
puhl-luck c --no-learn
```

Rank choices:

```bat
puhl-luck q "machine learning language" "CSS,Python"
```

Batch rank:

```bat
puhl-luck b ask.tsv --quiet
puhl-luck b ask.jsonl --output answers.jsonl
```

Recall learned memories:

```bat
puhl-luck recall "query text"
```

Show status:

```bat
puhl-luck status
```

## Data Folder

Example:

```text
data/
  notes.txt
  manual.md
  rows.csv
  rows.jsonl
  image.png
  sound.wav
  raw.bin
```

Supported text files:

```text
.txt .md .csv .jsonl .json
```

Supported file events:

```text
.png .jpg .jpeg .webp .gif .bmp
.wav .mp3 .flac .ogg .m4a
any other file as bytes
```

For `.jsonl`, PUHL-LUCK reads observation fields such as `text`, `content`,
`question`, `prompt`, and `input`.

## Learned Files

The default learned brain folder is `brain_data/`:

```text
brain_data/
  brain_memory.pkl
  brain_rank_micro.pmr
  brain_meta.json
  interaction_log.jsonl
```

Use another brain folder when you want a separate model:

```bat
puhl-luck --brain-dir my_brain t data
puhl-luck my_brain
```

## Chat

Inside chat:

```text
/learn on
/learn off
/quiz topic
/rank question | choice1,choice2
/save
/stats
/exit
```

Learning is on by default. When learning is on, the user message, generated
reply, and combined turn are stored back into the brain.

## Inspect And Edit Memory

Inspect:

```bat
puhl-luck inspect --source data
puhl-luck inspect --contains "bad phrase"
puhl-luck inspect --modality image
```

Forget selected memories:

```bat
puhl-luck forget --contains "bad phrase" --dry-run
puhl-luck forget --contains "bad phrase" --yes
puhl-luck forget --event-id EVENT_ID --yes
```

Explain a ranked answer:

```bat
puhl-luck explain "machine learning language" --choices "CSS,Python"
```

## Benchmarks

Generation quality:

```bat
python scratch\benchmark_generation_quality.py
```

Regression guard:

```bat
python scratch\check_generation_regression_guard.py
```

Multimodal ranking:

```bat
python scratch\benchmark_multimodal_generalization.py --repeats 50
```

Energy modes:

```bat
python scratch\benchmark_puhl_energy_modes.py
```
