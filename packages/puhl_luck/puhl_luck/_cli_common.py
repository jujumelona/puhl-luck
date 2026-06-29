from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .brain_memory import ENERGY_MODES, TEXT_SUFFIXES, BrainMemory, MicroRankModel


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_BRAIN_DIR = Path("brain_data")
DEFAULT_DATA_DIR = Path("data")
TRAIN_HINT = "put files in data/ and run: puhl-luck train"
COMMAND_NAMES = {
    "start", "s",
    "train", "t", "learn",
    "chat", "c",
    "ask", "q",
    "ask-batch", "batch", "b",
    "ask-stdin",
    "recall",
    "status",
    "inspect",
    "forget",
    "explain",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    return value


def read_text_rows(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [line.strip().lstrip("\ufeff") for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines() if line.strip()]
    if suffix == ".csv":
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            for row in csv.reader(f):
                text = " ".join(str(cell).strip() for cell in row if str(cell).strip())
                if text:
                    rows.append(text)
        return rows
    if suffix == ".jsonl":
        rows = []
        preferred = ("text", "content", "question", "prompt", "input")
        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    rows.append(line)
                    continue
                if isinstance(obj, dict):
                    parts = [str(obj[k]).strip() for k in preferred if k in obj and str(obj[k]).strip()]
                    rows.append(" ".join(parts) if parts else json.dumps(obj, ensure_ascii=False))
                else:
                    rows.append(str(obj))
        return rows
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
        if isinstance(obj, list):
            return [json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x for x in obj]
        if isinstance(obj, dict):
            return [json.dumps(obj, ensure_ascii=False)]
    return []


def resolve_training_data_path(data_arg: Optional[str]) -> Path:
    if data_arg:
        return Path(data_arg)
    if DEFAULT_DATA_DIR.exists():
        return DEFAULT_DATA_DIR
    raise SystemExit(f"no training data path provided and default folder does not exist: {DEFAULT_DATA_DIR}\n{TRAIN_HINT}")


def brain_dir_from_load_path(value: str) -> Optional[Path]:
    path = Path(value)
    if path.is_file() and path.name in {"brain_memory.pkl", "brain_rank_micro.pmr", "brain_meta.json"}:
        return path.parent
    if path.is_dir() and (
        (path / "brain_memory.pkl").exists()
        or (path / "brain_rank_micro.pmr").exists()
        or (path / "brain_meta.json").exists()
    ):
        return path
    return None


def normalize_default_argv(argv: List[str]) -> List[str]:
    if not argv:
        return ["chat"] if (DEFAULT_BRAIN_DIR / "brain_memory.pkl").exists() else ["start"]
    first = argv[0]
    if first.startswith("-") or first in COMMAND_NAMES:
        return argv
    brain_dir = brain_dir_from_load_path(first)
    if brain_dir is not None:
        return ["--brain-dir", str(brain_dir), "chat", *argv[1:]]
    if len(argv) >= 2:
        return ["ask", argv[0], argv[1], *argv[2:]]
    return ["chat"]


def iter_training_inputs(data_path: Path) -> Iterable[tuple[str, Any]]:
    if not data_path.exists():
        raise FileNotFoundError(f"data path not found: {data_path}")
    files = sorted(p for p in data_path.rglob("*") if p.is_file()) if data_path.is_dir() else [data_path]
    for path in files:
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            for text in read_text_rows(path):
                if text.strip():
                    yield "text", text
        else:
            yield "file", path
