from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "puhl_luck"))

from puhl_luck import BrainMemory
from puhl_luck.brain_memory import tokenize


TRAIN_TEXTS = [
    "mist valley opens obsidian gate under rain",
    "silent archive keeps amber key near river",
    "winter market lights broken compass at dawn",
    "stranger maps silver road beyond forest",
]

CASES = [
    ("seen_prefix", ["mist", "valley"], "long context seen during training"),
    ("backoff_prefix", ["unknown", "mist", "valley"], "unseen prefix, suffix exists"),
    ("mixed_context", ["mist", "archive"], "two seen words not adjacent together"),
    ("no_context", ["purple", "comet"], "no suffix context in training"),
]


def ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(0, max(0, len(tokens) - n + 1))}


def train_ngrams(n: int) -> Set[Tuple[str, ...]]:
    out: Set[Tuple[str, ...]] = set()
    for text in TRAIN_TEXTS:
        out.update(ngrams(tokenize(text), n))
    return out


def overlap_ratio(tokens: List[str], known: Set[Tuple[str, ...]], n: int) -> float:
    rows = ngrams(tokens, n)
    if not rows:
        return 0.0
    return len(rows & known) / len(rows)


def run() -> None:
    mem = BrainMemory()
    for text in TRAIN_TEXTS:
        mem.expose_text(text, source="generation_train")

    train_bigrams = train_ngrams(2)
    train_fourgrams = train_ngrams(4)
    print("| case | mode | empty | copy_4gram | novel_bigram | repeat_bigram | ms | output |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for name, prefix, _desc in CASES:
        for mode, fn, kwargs in (
            ("order_greedy", mem.graph_decode_text, {}),
            ("order_sample_t08", mem.graph_decode_text, {"temperature": 0.8}),
            ("order_beam3", mem.graph_decode_text, {"beam_size": 3}),
            ("energy_greedy", mem.memory_energy_decode_text, {}),
            ("energy_sample_t08", mem.memory_energy_decode_text, {"temperature": 0.8}),
            ("energy_beam3", mem.memory_energy_decode_text, {"beam_size": 3}),
        ):
            t0 = time.perf_counter()
            output = fn(" ".join(prefix), prefix, max_new_tokens=8, **kwargs)
            ms = (time.perf_counter() - t0) * 1000.0
            out_tokens = tokenize(output)
            copy4 = overlap_ratio(out_tokens, train_fourgrams, 4)
            novel2 = 1.0 - overlap_ratio(out_tokens, train_bigrams, 2) if out_tokens else 0.0
            bigrams = list(tuple(out_tokens[i:i + 2]) for i in range(max(0, len(out_tokens) - 1)))
            repeat_bigram = 1.0 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
            print(
                f"| {name} | {mode} | {int(not bool(output))} | {copy4:.4f} | {novel2:.4f} | {repeat_bigram:.4f} | {ms:.5f} | {output or '<empty>'} |"
            )


if __name__ == "__main__":
    run()
