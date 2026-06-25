from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "puhl_luck"))

from puhl_luck import BrainMemory


CARDS = [
    ("lumora_001", "winter echoes blue lantern"),
    ("naruvol_003", "summer echoes red tower"),
    ("caelith_007", "autumn mirror brass river"),
    ("velora_011", "spring mirror green archive"),
]


def build_memory(noise: int = 64) -> BrainMemory:
    mem = BrainMemory(window_size=6)
    for card_id, body in CARDS:
        mem.expose_text(f"memory card {card_id} {body}", source="cards")
    for i in range(noise):
        mem.expose_text(f"memory card filler_{i:03d} winter echoes memory archive card", source="noise")
    return mem


def eval_recall(mem: BrainMemory, mode: str) -> Dict[str, float]:
    correct = 0
    t0 = time.perf_counter()
    for card_id, body in CARDS:
        query = f"{card_id} {body.split()[0]} {body.split()[1]}"
        features = mem.features_for_query(query)
        if mode == "continuous":
            scores = mem.hopfield_recall_continuous(features, iterations=2, top_k=4)
        elif mode == "feature_continuous":
            scores = mem.hopfield_recall_feature_continuous(features, iterations=2, top_k=4)
        else:
            scores = mem.hopfield_recall(features, iterations=2, top_k=4)
        if scores:
            top_event = max(scores.items(), key=lambda item: item[1])[0]
            correct += int(card_id in mem.events[top_event].preview)
    elapsed = time.perf_counter() - t0
    return {"acc": correct / len(CARDS), "ms": elapsed * 1000.0 / len(CARDS)}


def run() -> None:
    print("| noise | mode | acc | recall_ms | events | hdc_bits |")
    print("|---:|---|---:|---:|---:|---:|")
    for noise in (0, 16, 64, 256):
        mem = build_memory(noise=noise)
        for mode in ("discrete", "continuous", "feature_continuous"):
            row = eval_recall(mem, mode)
            print(f"| {noise} | {mode} | {row['acc']:.4f} | {row['ms']:.5f} | {len(mem.events)} | {mem.hdc_bits} |")


if __name__ == "__main__":
    run()
