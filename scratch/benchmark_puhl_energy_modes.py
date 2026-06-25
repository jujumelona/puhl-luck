from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "puhl_luck"))

from puhl_luck import BrainMemory, MicroRankModel
from puhl_luck.brain_memory import ENERGY_MODES, tokenize

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - benchmark fallback
    torch = None
    nn = None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    obj = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        data = obj.get("rows", obj.get("data", []))
        return [row for row in data if isinstance(row, dict)]
    return []


def row_text(row: Dict[str, Any]) -> str:
    parts = []
    for key in ("text", "content", "question", "prompt", "input"):
        value = row.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def row_choices(row: Dict[str, Any]) -> List[str]:
    choices = row.get("choices", row.get("options", row.get("candidates", [])))
    if isinstance(choices, str):
        return [c.strip() for c in choices.split(",") if c.strip()]
    if isinstance(choices, list):
        return [str(c).strip() for c in choices if str(c).strip()]
    return []


def row_question(row: Dict[str, Any]) -> str:
    return str(row.get("question", row.get("prompt", row.get("input", row.get("text", "")))))


def row_answer_index(row: Dict[str, Any], choices: List[str]) -> int | None:
    ans = row.get("answer", row.get("label", row.get("target")))
    if ans is None:
        return None
    try:
        idx = int(ans)
        if 0 <= idx < len(choices):
            return idx
    except (TypeError, ValueError):
        pass
    ans_s = str(ans).strip().lower()
    for i, choice in enumerate(choices):
        if str(choice).strip().lower() == ans_s:
            return i
    return None


def score_mode(breakdown: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(float(breakdown.get(name, 0.0)) * weight for name, weight in weights.items())


def evaluate_mode(mem: BrainMemory, rows: List[Dict[str, Any]], mode: str) -> Dict[str, float]:
    weights = ENERGY_MODES[mode]
    correct = 0
    total = 0
    t0 = time.perf_counter()
    for row in rows:
        choices = row_choices(row)
        gold = row_answer_index(row, choices)
        if gold is None or not choices:
            continue
        breakdown = mem.rank_energy_breakdown(row_question(row), choices)
        scores = [score_mode(item, weights) for item in breakdown]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == gold)
        total += 1
    elapsed = time.perf_counter() - t0
    return {
        "mode": mode,
        "acc": correct / max(1, total),
        "items": total,
        "infer_ms": elapsed * 1000.0 / max(1, total),
    }


def evaluate_ranker(model: Any, rows: List[Dict[str, Any]], name: str) -> Dict[str, float]:
    correct = 0
    total = 0
    t0 = time.perf_counter()
    for row in rows:
        choices = row_choices(row)
        gold = row_answer_index(row, choices)
        if gold is None or not choices:
            continue
        pred = model.predict(row_question(row), choices) if hasattr(model, "predict") else model.rank(row_question(row), choices)[0]
        correct += int(pred == gold)
        total += 1
    elapsed = time.perf_counter() - t0
    return {
        "mode": name,
        "acc": correct / max(1, total),
        "items": total,
        "infer_ms": elapsed * 1000.0 / max(1, total),
    }


def synthetic_rows() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = [
        {"text": "python machine learning language data model"},
        {"text": "css web page style layout color"},
        {"text": "tiger dangerous striped animal fast predator"},
        {"text": "house safe family shelter rooms home"},
        {"text": "water cold liquid drink clear"},
        {"text": "fire hot flame burns danger"},
    ]
    test = [
        {"question": "machine learning language", "choices": ["CSS", "Python"], "answer": 1},
        {"question": "web style layout", "choices": ["Python", "CSS"], "answer": 1},
        {"question": "dangerous striped animal", "choices": ["house", "tiger"], "answer": 1},
        {"question": "safe shelter rooms", "choices": ["tiger", "house"], "answer": 1},
        {"question": "hot flame danger", "choices": ["water", "fire"], "answer": 1},
        {"question": "cold liquid drink", "choices": ["fire", "water"], "answer": 1},
    ]
    return train, test


class DenseTinyPair(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def dense_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        choices = row_choices(row)
        if row_answer_index(row, choices) is not None:
            out.append(row)
    return out


def build_vocab(rows: Iterable[Dict[str, Any]], limit: int = 512) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        for token in tokenize(row_question(row), 128):
            counts[token] = counts.get(token, 0) + 1
        for choice in row_choices(row):
            for token in tokenize(choice, 64):
                counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    return {token: idx for idx, (token, _) in enumerate(ranked)}


def encode_pair(question: str, choice: str, vocab: Dict[str, int]) -> "torch.Tensor":
    x = torch.zeros(len(vocab), dtype=torch.float32)
    for token in tokenize(question, 128):
        idx = vocab.get(token)
        if idx is not None:
            x[idx] += 1.0
    for token in tokenize(choice, 64):
        idx = vocab.get(token)
        if idx is not None:
            x[idx] += 1.5
    norm = torch.linalg.vector_norm(x)
    return x / norm if norm > 0 else x


def train_dense_tiny(train_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]], epochs: int = 80) -> Dict[str, float]:
    if torch is None or not train_rows:
        return {"model": "DenseTiny", "acc": 0.0, "infer_ms": 0.0, "train_s": 0.0, "storage_kb": 0.0, "items": 0}
    vocab = build_vocab(train_rows)
    model = DenseTinyPair(max(1, len(vocab)), hidden=16)
    opt = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    pairs = []
    for row in train_rows:
        choices = row_choices(row)
        gold = row_answer_index(row, choices)
        if gold is None:
            continue
        for idx, choice in enumerate(choices):
            pairs.append((encode_pair(row_question(row), choice, vocab), 1.0 if idx == gold else 0.0))
    t0 = time.perf_counter()
    if pairs:
        x = torch.stack([p[0] for p in pairs])
        y = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
        for _ in range(max(1, epochs)):
            logits = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    train_s = time.perf_counter() - t0
    correct = 0
    total = 0
    t1 = time.perf_counter()
    with torch.no_grad():
        for row in test_rows:
            choices = row_choices(row)
            gold = row_answer_index(row, choices)
            if gold is None or not choices:
                continue
            scores = [float(model(encode_pair(row_question(row), choice, vocab).unsqueeze(0))[0]) for choice in choices]
            pred = max(range(len(scores)), key=lambda i: scores[i])
            correct += int(pred == gold)
            total += 1
    infer_s = time.perf_counter() - t1
    storage_kb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024.0
    return {
        "model": "DenseTiny",
        "acc": correct / max(1, total),
        "infer_ms": infer_s * 1000.0 / max(1, total),
        "train_s": train_s,
        "storage_kb": storage_kb,
        "items": total,
    }


def run(args: argparse.Namespace) -> None:
    if args.train:
        train_rows = load_rows(Path(args.train))
        test_rows = load_rows(Path(args.test))
    else:
        train_rows, test_rows = synthetic_rows()

    mem = BrainMemory(window_size=args.window_size)
    t0 = time.perf_counter()
    for _ in range(max(1, args.epochs)):
        for row in train_rows:
            text = row_text(row).strip()
            if text:
                mem.expose_text(text, source="bench")
    train_s = time.perf_counter() - t0

    results = [evaluate_mode(mem, test_rows, mode) for mode in ENERGY_MODES]
    best = max(results, key=lambda row: (row["acc"], -row["infer_ms"]))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "brain_memory.pkl"
        micro_rank_path = Path(tmp) / "brain_rank_micro.pmr"
        mem.save(path)
        mem.save_rank_micro_only(micro_rank_path)
        micro_model = MicroRankModel.load(micro_rank_path)
        micro_result = evaluate_ranker(micro_model, test_rows, "micro")
        storage_kb = path.stat().st_size / 1024.0
        micro_rank_storage_kb = micro_rank_path.stat().st_size / 1024.0
    print(
        f"train_items={len(train_rows)} test_items={len(test_rows)} train_s={train_s:.4f} "
        f"storage_kb={storage_kb:.1f} micro_rank_storage_kb={micro_rank_storage_kb:.2f}"
    )
    print("| mode | acc | infer_ms | items |")
    print("|---|---:|---:|---:|")
    for row in results:
        marker = "*" if row["mode"] == best["mode"] else " "
        print(f"| {marker}{row['mode']} | {row['acc']:.4f} | {row['infer_ms']:.5f} | {int(row['items'])} |")
    print("| ranker | acc | infer_ms | storage_kb | items |")
    print("|---|---:|---:|---:|---:|")
    print(f"| micro | {micro_result['acc']:.4f} | {micro_result['infer_ms']:.5f} | {micro_rank_storage_kb:.2f} | {int(micro_result['items'])} |")
    supervised_train = dense_rows(train_rows)
    if supervised_train:
        dense = train_dense_tiny(supervised_train, test_rows, epochs=args.dense_epochs)
        print("| baseline | acc | infer_ms | train_s | storage_kb |")
        print("|---|---:|---:|---:|---:|")
        print(f"| {dense['model']} | {dense['acc']:.4f} | {dense['infer_ms']:.5f} | {dense['train_s']:.4f} | {dense['storage_kb']:.2f} |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PUHL unsupervised energy scoring ablation benchmark")
    parser.add_argument("--train", help="train rows .jsonl/.json/.csv; answer is ignored for training")
    parser.add_argument("--test", help="test rows .jsonl/.json/.csv with answer only for evaluation")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dense-epochs", type=int, default=80)
    parser.add_argument("--window-size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
