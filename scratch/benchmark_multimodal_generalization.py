from __future__ import annotations

import argparse
import math
import struct
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "puhl_luck"))

from puhl_luck import BrainMemory, MicroRankModel
from puhl_luck.brain_memory import ENERGY_MODES


@dataclass(frozen=True)
class EvalCase:
    family: str
    query: str
    choices: Tuple[str, ...]
    answer: int


def write_bmp(path: Path, width: int, height: int, kind: str) -> None:
    row_stride = ((width * 3 + 3) // 4) * 4
    pixel_bytes = bytearray()
    for y in range(height - 1, -1, -1):
        row = bytearray()
        for x in range(width):
            if kind == "vertical":
                v = 230 if width // 3 <= x < (2 * width) // 3 else 30
            elif kind == "vertical_shift":
                v = 220 if width // 3 + 1 <= x < (2 * width) // 3 + 1 else 35
            elif kind == "horizontal":
                v = 230 if height // 3 <= y < (2 * height) // 3 else 30
            elif kind == "horizontal_shift":
                v = 220 if height // 3 + 1 <= y < (2 * height) // 3 + 1 else 35
            else:
                v = 220 if ((x // 4) + (y // 4)) % 2 == 0 else 35
            row.extend([v, v, v])
        row.extend(b"\x00" * (row_stride - len(row)))
        pixel_bytes.extend(row)
    file_size = 54 + len(pixel_bytes)
    header = bytearray()
    header.extend(b"BM")
    header.extend(struct.pack("<IHHI", file_size, 0, 0, 54))
    header.extend(struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(pixel_bytes), 2835, 2835, 0, 0))
    path.write_bytes(bytes(header) + bytes(pixel_bytes))


def write_wav(path: Path, frequency: float, seconds: float = 0.35, rate: int = 16000) -> None:
    frames = int(seconds * rate)
    samples = bytearray()
    for i in range(frames):
        value = int(math.sin(2.0 * math.pi * frequency * i / rate) * 12000)
        samples.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(samples))


def write_bytes(path: Path, header: bytes, body_seed: int) -> None:
    data = bytearray(header)
    state = body_seed & 0xFFFFFFFF
    for i in range(1536):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        data.append((state >> 16) & 0xFF)
        if i % 97 == 0:
            data.extend(header[:8])
    path.write_bytes(bytes(data))


def build_suite(root: Path) -> Tuple[List[Tuple[str, str]], List[EvalCase]]:
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    proto_python = "python machine learning vectors model training inference data pipeline"
    proto_css = "css layout selector cascade browser style page responsive color"
    proto_med = (
        "clinical triage patient symptom fever cough diagnosis treatment followup "
        "hospital note medication risk"
    )
    proto_legal = (
        "contract clause liability arbitration indemnity party notice breach remedy "
        "agreement jurisdiction"
    )

    vertical = assets / "image_vertical.bmp"
    horizontal = assets / "image_horizontal.bmp"
    vertical_q = assets / "image_vertical_query.bmp"
    horizontal_q = assets / "image_horizontal_query.bmp"
    write_bmp(vertical, 32, 32, "vertical")
    write_bmp(horizontal, 32, 32, "horizontal")
    write_bmp(vertical_q, 32, 32, "vertical_shift")
    write_bmp(horizontal_q, 32, 32, "horizontal_shift")

    low = assets / "audio_low.wav"
    high = assets / "audio_high.wav"
    low_q = assets / "audio_low_query.wav"
    high_q = assets / "audio_high_query.wav"
    write_wav(low, 440.0)
    write_wav(high, 880.0)
    write_wav(low_q, 455.0)
    write_wav(high_q, 910.0)

    bin_a = assets / "packet_alpha.bin"
    bin_b = assets / "packet_beta.bin"
    bin_a_q = assets / "packet_alpha_query.bin"
    bin_b_q = assets / "packet_beta_query.bin"
    write_bytes(bin_a, b"PUHL_ALPHA_STREAM\x00", 17)
    write_bytes(bin_b, b"PUHL_BETA_STREAM\xff", 93)
    write_bytes(bin_a_q, b"PUHL_ALPHA_STREAM\x01", 19)
    write_bytes(bin_b_q, b"PUHL_BETA_STREAM\xfe", 91)

    train: List[Tuple[str, str]] = [
        ("text", proto_python),
        ("text", proto_css),
        ("text", proto_med),
        ("text", proto_legal),
        ("file", str(vertical)),
        ("file", str(horizontal)),
        ("file", str(low)),
        ("file", str(high)),
        ("file", str(bin_a)),
        ("file", str(bin_b)),
    ]
    cases = [
        EvalCase("text_short", "learning inference data", (proto_css, proto_python), 1),
        EvalCase("text_short", "browser selector layout", (proto_python, proto_css), 1),
        EvalCase("text_long", "patient fever medication hospital diagnosis", (proto_legal, proto_med), 1),
        EvalCase("text_long", "arbitration breach contract notice", (proto_med, proto_legal), 1),
        EvalCase("image", str(vertical_q), (str(horizontal), str(vertical)), 1),
        EvalCase("image", str(horizontal_q), (str(vertical), str(horizontal)), 1),
        EvalCase("audio", str(low_q), (str(high), str(low)), 1),
        EvalCase("audio", str(high_q), (str(low), str(high)), 1),
        EvalCase("bytes", str(bin_a_q), (str(bin_b), str(bin_a)), 1),
        EvalCase("bytes", str(bin_b_q), (str(bin_a), str(bin_b)), 1),
    ]
    return train, cases


def train_memory(rows: Iterable[Tuple[str, str]], epochs: int) -> Tuple[BrainMemory, float]:
    mem = BrainMemory(window_size=4)
    t0 = time.perf_counter()
    for _ in range(max(1, epochs)):
        for kind, value in rows:
            if kind == "file":
                mem.expose_file(value)
            else:
                mem.expose_text(value, source="bench")
    return mem, time.perf_counter() - t0


def evaluate_full(mem: BrainMemory, cases: List[EvalCase], mode: str, repeats: int) -> Dict[str, float]:
    correct = 0
    total = 0
    by_family: Dict[str, List[int]] = {}
    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        for case in cases:
            pred, _ = mem.rank(case.query, list(case.choices), mode=mode)
            correct += int(pred == case.answer)
            total += 1
    elapsed = time.perf_counter() - t0
    for case in cases:
        pred, _ = mem.rank(case.query, list(case.choices), mode=mode)
        by_family.setdefault(case.family, []).append(int(pred == case.answer))
    return {
        "engine": f"full:{mode}",
        "acc": correct / max(1, total),
        "infer_ms": elapsed * 1000.0 / max(1, total),
        "items": len(cases),
        **{f"{name}_acc": sum(vals) / max(1, len(vals)) for name, vals in sorted(by_family.items())},
    }


def evaluate_micro(model: MicroRankModel, cases: List[EvalCase], repeats: int) -> Dict[str, float]:
    correct = 0
    total = 0
    by_family: Dict[str, List[int]] = {}
    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        for case in cases:
            pred = model.predict(case.query, list(case.choices))
            correct += int(pred == case.answer)
            total += 1
    elapsed = time.perf_counter() - t0
    for case in cases:
        pred = model.predict(case.query, list(case.choices))
        by_family.setdefault(case.family, []).append(int(pred == case.answer))
    return {
        "engine": "micro",
        "acc": correct / max(1, total),
        "infer_ms": elapsed * 1000.0 / max(1, total),
        "items": len(cases),
        **{f"{name}_acc": sum(vals) / max(1, len(vals)) for name, vals in sorted(by_family.items())},
    }


def print_table(rows: List[Dict[str, float]], storage_kb: float, micro_kb: float, train_s: float) -> None:
    families = ["text_short", "text_long", "image", "audio", "bytes"]
    print(f"train_s={train_s:.4f} storage_kb={storage_kb:.2f} micro_rank_storage_kb={micro_kb:.2f}")
    print("| engine | acc | infer_ms | text_short | text_long | image | audio | bytes | items |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        cells = [
            str(row["engine"]),
            f"{row['acc']:.4f}",
            f"{row['infer_ms']:.5f}",
            *[f"{row.get(f'{family}_acc', 0.0):.4f}" for family in families],
            str(int(row["items"])),
        ]
        print("| " + " | ".join(cells) + " |")


def run(args: argparse.Namespace) -> None:
    context = tempfile.TemporaryDirectory() if not args.keep_data else None
    root = Path(context.name if context else args.keep_data)
    root.mkdir(parents=True, exist_ok=True)
    try:
        train, cases = build_suite(root)
        mem, train_s = train_memory(train, args.epochs)
        brain_path = root / "brain_memory.pkl"
        micro_path = root / "brain_rank_micro.pmr"
        mem.save(brain_path)
        mem.save_rank_micro_only(micro_path, hash_bits=args.micro_hash_bits)
        micro = MicroRankModel.load(micro_path)

        rows = [evaluate_full(mem, cases, mode, args.repeats) for mode in ENERGY_MODES]
        rows.append(evaluate_micro(micro, cases, args.repeats))
        print_table(rows, brain_path.stat().st_size / 1024.0, micro_path.stat().st_size / 1024.0, train_s)
        if args.keep_data:
            print(f"kept_data={root}")
    finally:
        if context is not None:
            context.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PUHL multimodal unsupervised choice-matching benchmark")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--micro-hash-bits", type=int, choices=(16, 32), default=16)
    parser.add_argument("--keep-data", type=Path, help="optional folder for generated benchmark assets/models")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
