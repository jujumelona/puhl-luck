from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "puhl_luck"))

from puhl_luck import BrainMemory, MicroRankModel


CARDS = [
    ("lumora_001", "winter echoes blue lantern"),
    ("naruvol_003", "summer echoes red tower"),
    ("caelith_007", "autumn mirror brass river"),
    ("velora_011", "spring mirror green archive"),
]


def train_memory(repeats: int, noisy: bool = True) -> BrainMemory:
    mem = BrainMemory(window_size=6)
    for repeat in range(repeats):
        for card_id, body in CARDS:
            mem.expose_text(f"memory card {card_id} {body}", source=f"repeat:{repeat}")
        if noisy:
            for i in range(12):
                mem.expose_text(
                    f"memory card filler_{i:03d} winter echoes memory archive card",
                    source=f"noise:{repeat}",
                )
    return mem


def evaluate_full(mem: BrainMemory) -> Dict[str, float]:
    correct = 0
    top_anchor = 0
    rows = []
    for idx, (card_id, body) in enumerate(CARDS):
        query = f"{card_id} {body.split()[0]} {body.split()[1]}"
        choices = [other_id for other_id, _ in CARDS]
        pred, scores = mem.rank(query, choices, mode="event")
        correct += int(pred == idx)
        state = mem._rank_query_state(query)
        top_event = state["event_scores"].most_common(1)[0][0] if state["event_scores"] else ""
        top_preview = mem.events[top_event].preview if top_event else ""
        top_anchor += int(card_id in top_preview)
        rows.append((card_id, choices[pred], scores[idx], top_preview[:80]))
    return {
        "acc": correct / len(CARDS),
        "top_anchor": top_anchor / len(CARDS),
        "rows": rows,
    }


def evaluate_micro(mem: BrainMemory) -> Dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "brain_rank_micro.pmr"
        mem.save_rank_micro_only(path, hash_bits=16)
        model = MicroRankModel.load(path)
        correct = 0
        for idx, (card_id, body) in enumerate(CARDS):
            query = f"{card_id} {body.split()[0]} {body.split()[1]}"
            choices = [other_id for other_id, _ in CARDS]
            pred = model.predict(query, choices)
            correct += int(pred == idx)
        return {"acc": correct / len(CARDS), "kb": path.stat().st_size / 1024.0}


def run() -> None:
    print("| repeats | events | full_acc | top_anchor | micro_acc | micro_kb | train_s |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for repeats in (1, 2, 5, 10, 25, 50):
        t0 = time.perf_counter()
        mem = train_memory(repeats)
        train_s = time.perf_counter() - t0
        full = evaluate_full(mem)
        micro = evaluate_micro(mem)
        print(
            f"| {repeats} | {len(mem.events)} | {full['acc']:.4f} | {full['top_anchor']:.4f} | "
            f"{micro['acc']:.4f} | {micro['kb']:.2f} | {train_s:.4f} |"
        )
        if full["acc"] < 1.0 or full["top_anchor"] < 1.0:
            for card_id, pred, score, top_preview in full["rows"]:
                print(f"  fail-detail card={card_id} pred={pred} gold_score={score:.4f} top={top_preview}")


if __name__ == "__main__":
    run()
