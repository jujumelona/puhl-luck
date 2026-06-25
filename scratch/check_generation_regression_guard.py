from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "puhl_luck"))
sys.path.insert(0, str(ROOT / "scratch"))

from puhl_luck import BrainMemory, MicroRankModel
from puhl_luck.brain_memory import ENERGY_MODES, tokenize

import benchmark_multimodal_generalization as multimodal
import benchmark_repeated_exposure_stability as repeated


TRAIN_TEXTS = [
    "mist valley opens obsidian gate under rain",
    "silent archive keeps amber key near river",
    "winter market lights broken compass at dawn",
    "stranger maps silver road beyond forest",
]


def ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(0, max(0, len(tokens) - n + 1))}


def train_ngrams(n: int) -> Set[Tuple[str, ...]]:
    rows: Set[Tuple[str, ...]] = set()
    for text in TRAIN_TEXTS:
        rows.update(ngrams(tokenize(text), n))
    return rows


def overlap_ratio(tokens: List[str], known: Set[Tuple[str, ...]], n: int) -> float:
    rows = ngrams(tokens, n)
    if not rows:
        return 0.0
    return len(rows & known) / len(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def check_generation() -> None:
    mem = BrainMemory()
    for text in TRAIN_TEXTS:
        mem.expose_text(text, source="generation_guard")

    train_fourgrams = train_ngrams(4)

    t0 = time.perf_counter()
    order_mixed = mem.graph_decode_text("mist archive", ["mist", "archive"], max_new_tokens=8)
    order_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    energy_mixed = mem.memory_energy_decode_text("mist archive", ["mist", "archive"], max_new_tokens=8)
    energy_ms = (time.perf_counter() - t0) * 1000.0

    no_context = mem.memory_energy_decode_text("purple comet", ["purple", "comet"], max_new_tokens=8)
    order_copy = overlap_ratio(tokenize(order_mixed), train_fourgrams, 4)
    energy_tokens = tokenize(energy_mixed)
    energy_copy = overlap_ratio(energy_tokens, train_fourgrams, 4)
    energy_bigrams = list(zip(energy_tokens, energy_tokens[1:]))
    energy_repeat = 1.0 - (len(set(energy_bigrams)) / len(energy_bigrams)) if energy_bigrams else 0.0

    require(bool(energy_mixed), "energy generation produced empty mixed-context output")
    require(bool(no_context), "energy generation did not produce dense-style fallback output")
    require(energy_copy <= order_copy + 1e-9, "energy generation increased 4-gram copying")
    require(energy_repeat <= 1e-9, f"energy generation repeated bigrams: {energy_repeat:.4f}")
    require(energy_ms <= max(20.0, order_ms * 2.0), f"energy generation too slow: {energy_ms:.4f}ms vs order {order_ms:.4f}ms")
    print(
        "generation ok: "
        f"order_ms={order_ms:.4f} energy_ms={energy_ms:.4f} "
        f"order_copy4={order_copy:.4f} energy_copy4={energy_copy:.4f} "
        f"energy_repeat2={energy_repeat:.4f}"
    )


def check_repeated_exposure() -> None:
    mem = repeated.train_memory(50)
    full = repeated.evaluate_full(mem)
    micro = repeated.evaluate_micro(mem)
    require(len(mem.events) == 16, f"repeated exposure duplicated events: {len(mem.events)}")
    require(full["acc"] >= 1.0, f"repeated exposure full acc regressed: {full['acc']:.4f}")
    require(full["top_anchor"] >= 1.0, f"repeated exposure anchor recall regressed: {full['top_anchor']:.4f}")
    require(micro["acc"] >= 1.0, f"repeated exposure micro acc regressed: {micro['acc']:.4f}")
    print(f"repeated ok: events={len(mem.events)} full_acc={full['acc']:.4f} micro_acc={micro['acc']:.4f}")


def check_multimodal() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        train, cases = multimodal.build_suite(Path(tmp))
        mem, train_s = multimodal.train_memory(train, epochs=1)
        brain_path = Path(tmp) / "brain_memory.pkl"
        micro_path = Path(tmp) / "brain_rank_micro.pmr"
        mem.save(brain_path)
        mem.save_rank_micro_only(micro_path, hash_bits=16)
        micro = MicroRankModel.load(micro_path)

        rows = {mode: multimodal.evaluate_full(mem, cases, mode, repeats=50) for mode in ENERGY_MODES}
        micro_row = multimodal.evaluate_micro(micro, cases, repeats=50)
        require(rows["event"]["acc"] >= 1.0, f"multimodal event acc regressed: {rows['event']['acc']:.4f}")
        require(rows["free_energy"]["acc"] >= 1.0, f"multimodal free_energy acc regressed: {rows['free_energy']['acc']:.4f}")
        require(micro_row["acc"] >= 1.0, f"multimodal micro acc regressed: {micro_row['acc']:.4f}")
        require(rows["event"]["infer_ms"] <= 5.0, f"multimodal event inference too slow: {rows['event']['infer_ms']:.4f}ms")
        require(rows["free_energy"]["infer_ms"] <= 5.0, f"multimodal free_energy inference too slow: {rows['free_energy']['infer_ms']:.4f}ms")
        print(
            "multimodal ok: "
            f"train_s={train_s:.4f} event_ms={rows['event']['infer_ms']:.4f} "
            f"free_energy_ms={rows['free_energy']['infer_ms']:.4f} micro_ms={micro_row['infer_ms']:.4f}"
        )


def main() -> None:
    check_generation()
    check_repeated_exposure()
    check_multimodal()
    print("guard ok")


if __name__ == "__main__":
    main()
