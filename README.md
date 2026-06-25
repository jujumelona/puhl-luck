# PUHL-LUCK

PUHL-LUCK is a local exposure-learning memory engine. It trains from observed
text, image, audio, and byte files without dense learned weights, then saves the
learned brain as files outside the code.

Paper / archive link: https://zenodo.org/records/20851529

The fastest inference path is the micro rank model:

```text
brain_data/brain_rank_micro.pmr
```

That file is generated automatically during training and is used by default for
choice ranking when possible.

## Install

Open CMD or PowerShell in this repository:

```bat
cd C:\Users\kkk\Desktop\puhl-luck
python -m pip install -e packages\puhl_luck
```

Check the command:

```bat
puhl-luck --help
```

If the script is not on PATH, run the module directly:

```bat
python -m puhl_luck.cli --help
```

## Learned Files

Training writes a brain folder. The default folder is `brain_data`.

```text
brain_data/
  brain_memory.pkl       full training/chat/recall memory
  brain_rank_micro.pmr   tiny fast inference rank model
  brain_meta.json        metadata and stats
  interaction_log.jsonl  chat interaction log
```

Use a different folder when you want a separate learned brain:

```bat
puhl-luck --brain-dir my_brain train --data data --epochs 1
```

## Training Data

Create a data folder:

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

For `.jsonl`, training reads observation fields such as `text`, `content`,
`question`, `prompt`, and `input`. Answer fields are not used for training.

## Train

One-command mode: train `data/` and start interactive chat:

```bat
puhl-luck start
```

This learns files in `data/`, writes the model files, then opens chat. Chat
learning is on by default, so the conversation also becomes training data.

Start the same flow with interaction learning disabled:

```bat
puhl-luck start --no-learn
```

Train the default `data` folder:

```bat
puhl-luck train
```

Train from another folder:

```bat
puhl-luck train --data data --epochs 1
```

Build the micro model with 32-bit feature hashes when the dataset is large and
you want fewer hash collisions:

```bat
puhl-luck train --data data --micro-hash-bits 32
```

Train from one file:

```bat
puhl-luck train --data data\notes.txt --epochs 3
```

Train into a specific brain folder:

```bat
puhl-luck --brain-dir brain_data train --data data --epochs 1
```

Training saves both the full memory and the micro rank model. There is no fixed
KB cap in the default export; larger data can create a larger learned model.
The default micro format is PMR2 with varint lengths, so event counts and row
counts are not limited to 65535 by the file format. `--micro-hash-bits 16` is
the compact default; `--micro-hash-bits 32` stores wider feature hashes.

## Fast Choice Ranking

Rank choices with the default `auto` engine:

```bat
puhl-luck ask "machine learning language" --choices "CSS,Python"
```

Force the tiny micro model:

```bat
puhl-luck ask "machine learning language" --choices "CSS,Python" --engine micro
```

Use the CPU micro path explicitly:

```bat
puhl-luck ask "machine learning language" --choices "CSS,Python" --engine micro --device cpu
```

Use the optional GPU micro kernel path:

```bat
puhl-luck ask "machine learning language" --choices "CSS,Python" --engine micro --device gpu
```

`--device gpu` requires CuPy, a visible CUDA GPU, and a 16-bit micro model. The
default `--device auto` uses CPU for the current micro model because very small
hash/event scoring is usually faster on CPU than paying GPU kernel launch
overhead. Use GPU for large batch experiments or larger 16-bit micro rank files.

Force the full memory ranker:

```bat
puhl-luck ask "machine learning language" --choices "CSS,Python" --engine full
```

Print full energy details from the full ranker:

```bat
puhl-luck ask "machine learning language" --choices "CSS,Python" --engine full --explain
```

`--engine auto` uses `brain_rank_micro.pmr` for normal event ranking when that
file exists. It falls back to `brain_memory.pkl` only when micro inference is not
available or when a full-only option such as `--explain` is requested.

## Batch Inference

TSV input format:

```text
question<TAB>choice1,choice2,choice3
```

Run a TSV batch through the micro model:

```bat
puhl-luck ask-batch --input ask.tsv --format tsv --engine micro --quiet
```

Run the same batch through the optional GPU backend:

```bat
puhl-luck ask-batch --input ask.tsv --format tsv --engine micro --device gpu --quiet
```

JSONL input format:

```jsonl
{"question":"machine learning language","choices":["CSS","Python"]}
{"question":"web style layout","choices":["Python","CSS"]}
```

Run JSONL and write JSONL output:

```bat
puhl-luck ask-batch --input ask.jsonl --output answers.jsonl --engine micro
```

Stream questions after loading the model once:

```bat
type ask.tsv | puhl-luck ask-stdin --format tsv --engine micro
```

For TSV streaming without `--scores`, each output line is only the predicted
zero-based choice index.

## Chat

Start interactive chat without auto-training `data/` first:

```bat
puhl-luck chat
```

Chat commands:

```text
/learn on
/learn off
/quiz topic
/rank question | choice1,choice2
/save
/stats
/exit
```

Learning is on by default. When learning is on, the user message, the generated
reply, and the combined turn are stored back into the brain.

Make a question from learned memory inside chat:

```text
you> /quiz machine learning
quiz> Which learned memory is most related to: machine learning?
  1. ...
  2. ...
answer with the number.
you> 1
```

Rank choices inside chat:

```text
you> /rank machine learning language | CSS,Python
rank> answer: Python
```

Start with chat learning disabled:

```bat
puhl-luck chat --no-learn
```

## Recall And Status

Retrieve related learned events:

```bat
puhl-luck recall "query text" --limit 5
```

Inspect addressed memories before editing them:

```bat
puhl-luck inspect --source data
puhl-luck inspect --contains "bad phrase"
puhl-luck inspect --modality image --limit 20
```

Explain which events and features supported a ranked answer:

```bat
puhl-luck explain "machine learning language" --choices "CSS,Python"
```

Delete only selected learned memories, then rebuild the indexes and micro rank
file:

```bat
puhl-luck forget --source bad_data --dry-run
puhl-luck forget --source bad_data --yes
puhl-luck forget --event-id EVENT_ID --yes
puhl-luck forget --contains "bad phrase" --yes
```

Show files and memory stats:

```bat
puhl-luck status
```

## Benchmarks

Run the current energy and micro-rank ablation:

```bat
python scratch\benchmark_puhl_energy_modes.py
```

Run the multimodal generalization sanity benchmark. It creates temporary text,
BMP image, WAV audio, and binary data, trains only from observations, evaluates
choice matching, then deletes generated data by default:

```bat
python scratch\benchmark_multimodal_generalization.py --repeats 50
```

Keep generated benchmark assets and model files for inspection:

```bat
python scratch\benchmark_multimodal_generalization.py --keep-data scratch\bench_tmp
```

Run it on your own rows:

```bat
python scratch\benchmark_puhl_energy_modes.py --train train.jsonl --test test.jsonl
```

Training rows are learned from observation text only. Evaluation answers are
used only to score the benchmark result.

## Architecture

The full memory is used for learning, chat, recall, and analysis. It stores
event features, directed co-activation links, ordered event traces, dynamic HDC
vectors, working memory, concept traces, and surprisal values.

The micro rank model is used for fastest choice inference. It stores one shared
compact feature hash space across modalities:

```text
input data
  -> modality feature extraction
  -> shared compact feature hashes
  -> feature-to-event and event-to-feature evidence
  -> event support / alignment score
  -> final choice
```

It is not a label table and does not train dense weights. Text, image, audio,
and byte observations share the same compressed feature-event memory.
