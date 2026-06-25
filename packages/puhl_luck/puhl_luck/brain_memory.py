from __future__ import annotations

import csv
import gzip
import hashlib
import json
import lzma
import zlib
import math
import pickle
import re
import struct
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


TEXT_SUFFIXES = {".txt", ".md", ".csv", ".jsonl", ".json"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
HDC_WORD_BITS = 64
ALL_FILE_SUFFIXES = TEXT_SUFFIXES | IMAGE_SUFFIXES | AUDIO_SUFFIXES
TOKEN_RE = re.compile(r"[a-z]+(?:_[a-z0-9]+)+|[a-z]+|\d+(?:\.\d+)?|[\uac00-\ud7a3]+|[+\-*/=<>]+")
ANCHOR_TOKEN_RE = re.compile(r"[a-z]+(?:_[a-z0-9]+)*_\d{2,}")
PRINTABLE_BYTES_RE = re.compile(rb"[A-Za-z][A-Za-z0-9_]{2,31}")
ENERGY_MODES: Dict[str, Dict[str, float]] = {
    "direct": {"weighted_overlap": 1.0},
    "event": {"event_support": 1.0, "alignment": 0.6},
    "contrast": {"weighted_overlap": 1.2, "contrast": 0.8, "conflict": -0.8},
    "free_energy": {
        "weighted_overlap": 2.2,
        "event_support": 1.6,
        "alignment": 1.2,
        "contrast": 0.7,
        "resonance": 1.0,
        "entropy": -0.2,
        "conflict": -1.0,
        "freq_penalty": -0.05,
    },
}


def stable_id(value: str, size: int = 16) -> str:
    return hashlib.blake2b(value.encode("utf-8", errors="ignore"), digest_size=size).hexdigest()


def byte_digest(data: bytes, size: int = 8) -> str:
    return hashlib.blake2b(data, digest_size=size).hexdigest()


def stable_u32(value: str) -> int:
    return zlib.crc32(str(value).encode("utf-8", errors="ignore")) & 0xffffffff


def stable_u16(value: str) -> int:
    return zlib.crc32(str(value).encode("utf-8", errors="ignore")) & 0xffff


def micro_hash(value: str, bits: int = 16) -> int:
    return stable_u32(value) if int(bits) == 32 else stable_u16(value)


def write_varuint(out: bytearray, value: int) -> None:
    n = max(0, int(value))
    while n >= 0x80:
        out.append((n & 0x7f) | 0x80)
        n >>= 7
    out.append(n)


def read_varuint(raw: bytes, offset: int) -> Tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if offset >= len(raw):
            raise ValueError("truncated varuint in micro rank file")
        byte = raw[offset]
        offset += 1
        value |= (byte & 0x7f) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("varuint is too large")


def tokenize(text: str, max_tokens: int = 512) -> List[str]:
    text = re.sub(r"\s+", " ", str(text)).strip().lower()
    return TOKEN_RE.findall(text)[:max_tokens]


def is_anchor_token(token: str) -> bool:
    return bool(ANCHOR_TOKEN_RE.fullmatch(token))


def text_feature_list(text: str) -> List[str]:
    tokens = tokenize(text)
    features = ["mod:text"]
    for token in tokens:
        features.append(f"text:tok:{token}")
        if is_anchor_token(token):
            features.append(f"text:id:{token}")
        features.extend(text_subword_features(token))
    for a, b in zip(tokens, tokens[1:]):
        features.append(f"text:bi:{a}_{b}")
    for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
        features.append(f"text:tri:{a}_{b}_{c}")
    for token in tokens:
        if len(token) >= 4:
            features.append(f"text:shape:{len(token)}:{token[:2]}:{token[-2:]}")
    return features


def text_sequence(text: str) -> List[str]:
    return [f"text:{token}" for token in tokenize(text)]


def text_content_hashes(text: str, bits: int = 16) -> List[int]:
    tokens = tokenize(text)
    hashes: List[int] = []
    for token in tokens:
        hashes.append(micro_hash(f"tok:{token}", bits))
        if is_anchor_token(token):
            hashes.append(micro_hash(f"id:{token}", bits))
        if len(token) >= 4:
            hashes.append(micro_hash(f"stem:{rough_stem(token)}", bits))
    for a, b in zip(tokens, tokens[1:]):
        hashes.append(micro_hash(f"bi:{a}_{b}", bits))
    for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
        hashes.append(micro_hash(f"tri:{a}_{b}_{c}", bits))
    return list(dict.fromkeys(hashes))


def looks_like_file_query(value: str) -> bool:
    text = str(value)
    if not text or len(text) > 512:
        return False
    if any(sep in text for sep in ("\\", "/", ":")):
        return True
    suffix = Path(text).suffix.lower()
    return suffix in ALL_FILE_SUFFIXES or bool(suffix and len(text.split()) == 1)


def size_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    return str(1 << max(0, min(40, int(math.log2(max(1, n))))))


def byte_histogram_features(data: bytes, prefix: str, bins: int = 16) -> List[str]:
    if not data:
        return [f"{prefix}:empty"]
    counts = [0 for _ in range(bins)]
    step = 256 // bins
    for b in data[:65536]:
        counts[min(b // step, bins - 1)] += 1
    total = max(1, sum(counts))
    return [f"{prefix}:bytebin:{i}:{int((c / total) * 10)}" for i, c in enumerate(counts) if c]


def chunk_hash_features(data: bytes, prefix: str, max_chunks: int = 16) -> List[str]:
    if not data:
        return []
    chunk_count = min(max_chunks, max(1, int(math.sqrt(len(data)))))
    stride = max(1, len(data) // chunk_count)
    out = []
    for i in range(0, len(data), stride):
        if len(out) >= max_chunks:
            break
        chunk = data[i:i + stride]
        if chunk:
            out.append(f"{prefix}:chunk:{byte_digest(chunk, 6)}")
    return out


def printable_byte_features(data: bytes, prefix: str, max_tokens: int = 12) -> List[str]:
    out = []
    seen = set()
    for raw in PRINTABLE_BYTES_RE.findall(data[:4096]):
        token = raw.decode("ascii", errors="ignore").lower()
        if token and token not in seen:
            seen.add(token)
            out.append(f"{prefix}:ascii:{token}")
            if len(token) >= 4:
                out.append(f"{prefix}:ascii_shape:{len(token)}:{token[:2]}:{token[-2:]}")
        if len(seen) >= max_tokens:
            break
    return out


def bmp_perceptual_features(data: bytes) -> List[str]:
    if not (data.startswith(b"BM") and len(data) >= 54):
        return []
    try:
        offset = int.from_bytes(data[10:14], "little", signed=False)
        width = int.from_bytes(data[18:22], "little", signed=True)
        height = int.from_bytes(data[22:26], "little", signed=True)
        bpp = int.from_bytes(data[28:30], "little", signed=False)
        compression = int.from_bytes(data[30:34], "little", signed=False)
    except Exception:
        return []
    if width <= 0 or height == 0 or bpp not in {24, 32} or compression != 0:
        return []
    height_abs = abs(height)
    bytes_per_pixel = bpp // 8
    row_stride = ((width * bytes_per_pixel + 3) // 4) * 4
    if offset + row_stride * height_abs > len(data):
        return []

    def gray_at(sample_x: int, sample_y: int) -> int:
        x = min(width - 1, max(0, sample_x))
        y = min(height_abs - 1, max(0, sample_y))
        file_y = height_abs - 1 - y if height > 0 else y
        pos = offset + file_y * row_stride + x * bytes_per_pixel
        b, g, r = data[pos], data[pos + 1], data[pos + 2]
        return int((int(r) * 30 + int(g) * 59 + int(b) * 11) / 100)

    vals8 = []
    for y in range(8):
        sy = int((y + 0.5) * height_abs / 8.0)
        for x in range(8):
            sx = int((x + 0.5) * width / 8.0)
            vals8.append(gray_at(sx, sy))
    avg = sum(vals8) / max(1, len(vals8))
    ahash = 0
    for value in vals8:
        ahash = (ahash << 1) | int(value >= avg)

    dhash = 0
    for y in range(8):
        sy = int((y + 0.5) * height_abs / 8.0)
        row = []
        for x in range(9):
            sx = int((x + 0.5) * width / 9.0)
            row.append(gray_at(sx, sy))
        for left, right in zip(row, row[1:]):
            dhash = (dhash << 1) | int(left > right)
    return [
        f"image:phash:a:{ahash >> 48:04x}:{(ahash >> 32) & 0xffff:04x}:{(ahash >> 16) & 0xffff:04x}:{ahash & 0xffff:04x}",
        f"image:phash:d:{dhash >> 48:04x}:{(dhash >> 32) & 0xffff:04x}:{(dhash >> 16) & 0xffff:04x}:{dhash & 0xffff:04x}",
    ]


def wav_bytes_features(data: bytes) -> List[str]:
    try:
        import io
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            duration_ms = int((frames / max(1, rate)) * 1000)
            out = [
                "audio:format:wav",
                f"audio:rate:{rate}",
                f"audio:channels:{channels}",
                f"audio:duration_ms:{size_bucket(duration_ms)}",
            ]
            out.extend(wav_signal_features(wf))
            return out
    except wave.Error:
        return ["audio:format:wav_unreadable"]


def image_header_features(data: bytes) -> List[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        out = ["image:format:png"]
        if len(data) >= 24:
            w = int.from_bytes(data[16:20], "big", signed=False)
            h = int.from_bytes(data[20:24], "big", signed=False)
            out.extend([f"image:w:{size_bucket(w)}", f"image:h:{size_bucket(h)}", f"image:aspect:{aspect_bucket(w, h)}"])
        return out
    if data.startswith(b"\xff\xd8"):
        return ["image:format:jpeg"]
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ["image:format:webp"]
    if data.startswith(b"BM") and len(data) >= 26:
        w = int.from_bytes(data[18:22], "little", signed=True)
        h = int.from_bytes(data[22:26], "little", signed=True)
        return ["image:format:bmp", f"image:w:{size_bucket(abs(w))}", f"image:h:{size_bucket(abs(h))}", f"image:aspect:{aspect_bucket(abs(w), abs(h))}"]
    return []


def micro_content_features_for_value(value: str) -> List[str]:
    text = str(value)
    if looks_like_file_query(text):
        path = Path(text)
        if path.exists() and path.is_file():
            suffix = path.suffix.lower()
            if suffix in TEXT_SUFFIXES:
                return content_features(text_feature_list(read_text_file_as_one_event(path)))
            data = path.read_bytes()
            if suffix in IMAGE_SUFFIXES:
                features = ["mod:image", f"image:suffix:{suffix}", f"image:size:{size_bucket(len(data))}"]
                features.extend(byte_histogram_features(data, "image"))
                features.extend(chunk_hash_features(data, "image"))
                features.extend(image_header_features(data))
                features.extend(bmp_perceptual_features(data))
                return content_features(features)
            if suffix in AUDIO_SUFFIXES:
                features = ["mod:audio", f"audio:suffix:{suffix}", f"audio:size:{size_bucket(len(data))}"]
                features.extend(byte_histogram_features(data, "audio"))
                features.extend(chunk_hash_features(data, "audio"))
                if suffix == ".wav":
                    features.extend(wav_bytes_features(data))
                return content_features(features)
            features = ["mod:bytes", f"bytes:suffix:{suffix}", f"bytes:size:{size_bucket(len(data))}"]
            features.extend(byte_histogram_features(data, "bytes"))
            features.extend(chunk_hash_features(data, "bytes"))
            features.extend(printable_byte_features(data, "bytes"))
            return content_features(features)
    return content_features(text_feature_list(text))


def micro_hashes_for_value(value: str, bits: int = 16) -> List[int]:
    text = str(value)
    if looks_like_file_query(text):
        path = Path(text)
        if path.exists() and path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            return text_content_hashes(read_text_file_as_one_event(path), bits=bits)
        if path.exists() and path.is_file():
            return list(dict.fromkeys(h for h in (micro_feature_hash(feature, bits=bits) for feature in micro_content_features_for_value(text)) if h is not None))
    return text_content_hashes(text, bits=bits)


def micro_feature_hash(feature: str, bits: int = 16) -> Optional[int]:
    name = canonical_feature(feature)
    if name.startswith("mod:") or name.startswith("c3:") or name.startswith("text:shape:"):
        return None
    return micro_hash(name, bits)


def dynamic_hdc_words(feature_count: int, event_count: int) -> int:
    scale = max(4, feature_count + 2 * event_count + 2)
    return max(2, math.ceil(math.log2(scale)))


def dynamic_hdc_bits(feature_count: int, event_count: int) -> int:
    return dynamic_hdc_words(feature_count, event_count) * HDC_WORD_BITS


def feature_hv(feature: str, words: int) -> np.ndarray:
    seed = hashlib.blake2b(feature.encode("utf-8", errors="ignore"), digest_size=16).digest()
    chunks = []
    block_i = 0
    while len(chunks) < words:
        block = hashlib.blake2b(seed + block_i.to_bytes(4, "little"), digest_size=32).digest()
        chunks.extend(struct.unpack("<4Q", block))
        block_i += 1
    return np.array(chunks[:words], dtype=np.uint64)


def rotate_hv(value: np.ndarray, amount: int) -> np.ndarray:
    if value.size == 0:
        return value.copy()
    bit_shift = amount % HDC_WORD_BITS
    if bit_shift == 0:
        return value.copy()
    left = value << np.uint64(bit_shift)
    right = value >> np.uint64(HDC_WORD_BITS - bit_shift)
    return (left | right).astype(np.uint64, copy=False)


def bundle_hv(features: Iterable[str], bits: Optional[int] = None) -> np.ndarray:
    feature_list = list(features)
    if bits is None:
        bits = dynamic_hdc_bits(len(feature_list), 1)
    words = max(1, bits // HDC_WORD_BITS)
    value = np.zeros(words, dtype=np.uint64)
    for i, feature in enumerate(feature_list):
        value ^= rotate_hv(feature_hv(feature, words), i)
    return value


def hv_similarity(a: np.ndarray, b: np.ndarray, bits: Optional[int] = None) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if bits is None:
        bits = min(a.size, b.size) * HDC_WORD_BITS
    words = max(1, min(a.size, b.size, bits // HDC_WORD_BITS))
    bits = words * HDC_WORD_BITS
    diff = np.bitwise_xor(a[:words], b[:words]).view(np.uint8)
    return 1.0 - (float(np.unpackbits(diff).sum()) / bits)


def hdc_band_count(words: int, event_count: int) -> int:
    return max(1, min(max(1, words), max(1, int(math.sqrt(max(1, event_count))) + 1)))


def hdc_bands(value: np.ndarray, event_count: int, start_word: int = 0) -> List[Tuple[int, int]]:
    bands = hdc_band_count(value.size, event_count)
    end = min(value.size, max(bands, start_word))
    return [(i, int(value[i])) for i in range(start_word, end)]


@dataclass
class EventRecord:
    event_id: str
    modality: str
    source: str
    label: Optional[str]
    features: List[str]
    sequence: List[str] = field(default_factory=list)
    preview: str = ""
    novelty: float = 1.0
    hv: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint64))
    created_at: int = 0
    last_accessed_at: int = 0


class MicroRankModel:
    """Tiny inference-only ranker backed by the PMR1 shared feature hash space."""

    def __init__(
        self,
        feature_rows: Dict[int, List[Tuple[int, int]]],
        event_features: Dict[int, Tuple[int, ...]],
        hash_bits: int = 16,
    ):
        self.feature_rows = feature_rows
        self.event_features = event_features
        self.hash_bits = 32 if int(hash_bits) == 32 else 16
        self.event_feature_sets = {eid: set(features) for eid, features in event_features.items()}
        self.feature_event_freq = {h: max(1, len(rows)) for h, rows in feature_rows.items()}
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        self._choice_cache: Dict[str, Dict[str, Any]] = {}
        self._rank_cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[int, List[float]]] = {}
        self._predict_cache: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self._gpu_checked = False
        self._gpu_module = None

    def hash_weight(self, feature_hash: int) -> float:
        return 1.0 / math.sqrt(float(self.feature_event_freq.get(feature_hash, 1)))

    def set_weight(self, feature_set: set[int]) -> float:
        return sum(self.hash_weight(h) for h in feature_set)

    def shared_weight(self, left: set[int], right: set[int]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(self.hash_weight(h) for h in left if h in right)

    @classmethod
    def load(cls, path: str | Path) -> "MicroRankModel":
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_bytes(cls, payload: bytes) -> "MicroRankModel":
        raw = zlib.decompress(payload)
        if raw[:4] == b"PMR2":
            offset = 4
            hash_bits, offset = read_varuint(raw, offset)
            top_count, offset = read_varuint(raw, offset)
            feature_rows: Dict[int, List[Tuple[int, int]]] = {}
            for _ in range(top_count):
                h, offset = read_varuint(raw, offset)
                row_count, offset = read_varuint(raw, offset)
                rows = []
                for _ in range(row_count):
                    eid, offset = read_varuint(raw, offset)
                    count, offset = read_varuint(raw, offset)
                    rows.append((eid, count))
                if rows:
                    feature_rows[h] = rows
            event_count, offset = read_varuint(raw, offset)
            event_features: Dict[int, Tuple[int, ...]] = {}
            for _ in range(event_count):
                eid, offset = read_varuint(raw, offset)
                feature_count, offset = read_varuint(raw, offset)
                features = []
                for _ in range(feature_count):
                    h, offset = read_varuint(raw, offset)
                    features.append(h)
                if features:
                    event_features[eid] = tuple(dict.fromkeys(features))
            return cls(feature_rows, event_features, hash_bits=hash_bits)
        if raw[:4] != b"PMR1":
            raise ValueError("micro rank file must start with PMR1")
        offset = 4
        top_count, event_count = struct.unpack_from("<HH", raw, offset)
        offset += 4
        feature_rows: Dict[int, List[Tuple[int, int]]] = {}
        for _ in range(top_count):
            h, row_count = struct.unpack_from("<HB", raw, offset)
            offset += 3
            rows = []
            for _ in range(row_count):
                eid, count = struct.unpack_from("<HB", raw, offset)
                offset += 3
                rows.append((eid, count))
            if rows:
                feature_rows[h] = rows
        event_features: Dict[int, Tuple[int, ...]] = {}
        for _ in range(event_count):
            eid, feature_count = struct.unpack_from("<HB", raw, offset)
            offset += 3
            features = []
            for _ in range(feature_count):
                (h,) = struct.unpack_from("<H", raw, offset)
                offset += 2
                features.append(h)
            if features:
                event_features[eid] = tuple(dict.fromkeys(features))
        return cls(feature_rows, event_features, hash_bits=16)

    def query_state(self, query: str) -> Dict[str, Any]:
        key = str(query)
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        hashes = micro_hashes_for_value(key, bits=self.hash_bits)
        query_set = set(hashes)
        event_scores: Dict[int, float] = {}
        for h in hashes[:96]:
            for eid, count in self.feature_rows.get(h, ()):
                event_scores[eid] = event_scores.get(eid, 0.0) + float(count) * self.hash_weight(h)
        top_events = sorted(event_scores.items(), key=lambda item: item[1], reverse=True)[:8]
        state = {"hashes": hashes, "set": query_set, "weight": self.set_weight(query_set), "top_events": top_events}
        if len(self._query_cache) > 4096:
            self._query_cache.clear()
            self._rank_cache.clear()
            self._predict_cache.clear()
        self._query_cache[key] = state
        return state

    def choice_state(self, choice: str) -> Dict[str, Any]:
        key = str(choice)
        cached = self._choice_cache.get(key)
        if cached is not None:
            return cached
        hashes = micro_hashes_for_value(key, bits=self.hash_bits)
        choice_set = set(hashes)
        row = {
            "hashes": hashes,
            "set": choice_set,
            "weight": self.set_weight(choice_set),
            "length_penalty": 1.0 + 0.06 * math.log1p(len(hashes)),
        }
        if len(self._choice_cache) > 4096:
            self._choice_cache.clear()
            self._rank_cache.clear()
            self._predict_cache.clear()
        self._choice_cache[key] = row
        return row

    def score_state(self, query_state: Dict[str, Any], choice_state: Dict[str, Any]) -> float:
        choice_set = choice_state["set"]
        query_set = query_state["set"]
        if not choice_set:
            return 0.0
        direct = 0.0
        if query_set:
            direct_shared = self.shared_weight(query_set, choice_set)
            direct = direct_shared / math.sqrt(max(1e-9, float(query_state.get("weight", 0.0) * choice_state.get("weight", 0.0))))
        support = 0.0
        alignment_num = 0.0
        alignment_den = 0.0
        for eid, event_score in query_state["top_events"]:
            event_set = self.event_feature_sets.get(eid)
            if not event_set:
                continue
            shared = self.shared_weight(event_set, choice_set)
            if shared <= 0.0:
                continue
            event_weight = self.set_weight(event_set)
            choice_weight = float(choice_state.get("weight", 0.0))
            weight = math.log1p(event_score)
            support = max(support, (shared / math.sqrt(max(1e-9, event_weight * choice_weight))) * weight)
            alignment_num += (shared / max(1e-9, event_weight + choice_weight - shared)) * weight
            alignment_den += weight
        alignment = alignment_num / max(1e-9, alignment_den)
        return (support + 0.6 * alignment + 0.25 * direct) / max(1e-9, float(choice_state["length_penalty"]))

    def gpu_module(self, required: bool = False) -> Any:
        if not self._gpu_checked:
            self._gpu_checked = True
            try:
                import cupy as cp  # type: ignore

                if cp.cuda.runtime.getDeviceCount() > 0:
                    self._gpu_module = cp
            except Exception:
                self._gpu_module = None
        if required and self._gpu_module is None:
            raise RuntimeError("GPU backend requires CuPy with a visible CUDA device")
        return self._gpu_module

    def resolve_device(self, device: str = "auto") -> str:
        selected = str(device or "auto").lower()
        if selected == "gpu":
            if self.hash_bits != 16:
                raise RuntimeError("GPU backend currently requires a 16-bit micro model")
            self.gpu_module(required=True)
            return "gpu"
        if selected == "auto":
            return "cpu"
        return "cpu"

    def score_states_gpu(self, query_state: Dict[str, Any], choice_states: List[Dict[str, Any]]) -> List[float]:
        cp = self.gpu_module(required=True)
        if not choice_states:
            return []
        choice_sets = [row["set"] for row in choice_states]
        choice_lens = cp.asarray([max(1, len(row["set"])) for row in choice_states], dtype=cp.float32)
        penalties = cp.asarray([float(row["length_penalty"]) for row in choice_states], dtype=cp.float32)
        feature_width = 65536

        choice_mask = cp.zeros((len(choice_states), feature_width), dtype=cp.bool_)
        for row_idx, row in enumerate(choice_states):
            hashes = row["hashes"]
            if hashes:
                choice_mask[row_idx, cp.asarray(hashes, dtype=cp.int32)] = True

        query_hashes = query_state["hashes"]
        direct = cp.zeros(len(choice_states), dtype=cp.float32)
        if query_hashes:
            shared_query = choice_mask[:, cp.asarray(query_hashes, dtype=cp.int32)].sum(axis=1, dtype=cp.float32)
            direct = shared_query / cp.sqrt(cp.asarray(float(len(query_hashes)), dtype=cp.float32) * choice_lens)

        top_events = query_state["top_events"]
        if not top_events:
            scores = (0.25 * direct) / penalties
            return [float(x) for x in cp.asnumpy(scores)]

        event_sets = [self.event_feature_sets.get(eid, set()) for eid, _ in top_events]
        event_lens = cp.asarray([max(1, len(row)) for row in event_sets], dtype=cp.float32)
        weights = cp.asarray([math.log1p(score) for _, score in top_events], dtype=cp.float32)
        event_mask = cp.zeros((len(event_sets), feature_width), dtype=cp.bool_)
        for row_idx, hashes in enumerate(event_sets):
            if hashes:
                event_mask[row_idx, cp.asarray(list(hashes), dtype=cp.int32)] = True

        shared = cp.matmul(event_mask.astype(cp.float32), choice_mask.T.astype(cp.float32))
        support = (shared / cp.sqrt(event_lens[:, None] * choice_lens[None, :]) * weights[:, None]).max(axis=0)
        union = event_lens[:, None] + choice_lens[None, :] - shared
        alignment = ((shared / cp.maximum(union, 1.0)) * weights[:, None]).sum(axis=0) / cp.maximum(weights.sum(), 1e-9)
        scores = (support + 0.6 * alignment + 0.25 * direct) / penalties
        return [float(x) for x in cp.asnumpy(scores)]

    def score_choices(self, query_state: Dict[str, Any], choice_states: List[Dict[str, Any]], device: str = "cpu") -> List[float]:
        selected = self.resolve_device(device)
        if selected == "gpu":
            return self.score_states_gpu(query_state, choice_states)
        return [self.score_state(query_state, row) for row in choice_states]

    def rank(self, query: str, choices: List[str], device: str = "cpu") -> Tuple[int, List[float]]:
        key = (str(query), tuple(str(choice) for choice in choices))
        use_cache = str(device or "cpu").lower() in {"cpu", "auto"}
        if use_cache:
            cached = self._rank_cache.get(key)
            if cached is not None:
                return cached
        state = self.query_state(key[0])
        raw_scores = self.score_choices(state, [self.choice_state(choice) for choice in key[1]], device=device)
        scores = self.normalize_scores(raw_scores)
        if not scores:
            return 0, []
        result = (max(range(len(scores)), key=lambda i: scores[i]), scores)
        if use_cache:
            if len(self._rank_cache) > 4096:
                self._rank_cache.clear()
            self._rank_cache[key] = result
        return result

    def predict(self, query: str, choices: List[str], device: str = "cpu") -> int:
        key = (str(query), tuple(str(choice) for choice in choices))
        use_cache = str(device or "cpu").lower() in {"cpu", "auto"}
        if use_cache:
            cached = self._predict_cache.get(key)
            if cached is not None:
                return cached
        if not key[1]:
            return 0
        state = self.query_state(key[0])
        choice_states = [self.choice_state(choice) for choice in key[1]]
        if self.resolve_device(device) == "gpu":
            raw_scores = self.score_states_gpu(state, choice_states)
            return max(range(len(raw_scores)), key=lambda i: raw_scores[i]) if raw_scores else 0
        best_idx = 0
        best_score = -float("inf")
        for idx, choice_state in enumerate(choice_states):
            score = self.score_state(state, choice_state)
            if score > best_score:
                best_idx = idx
                best_score = score
        if use_cache:
            if len(self._predict_cache) > 4096:
                self._predict_cache.clear()
            self._predict_cache[key] = best_idx
        return best_idx

    @staticmethod
    def normalize_scores(raw_scores: List[float]) -> List[float]:
        if not raw_scores:
            return []
        if len(raw_scores) == 1:
            return [1.0]
        max_score = max(raw_scores)
        min_score = min(raw_scores)
        if max_score - min_score < 1e-12:
            return [1.0 / len(raw_scores) for _ in raw_scores]
        mean = sum(raw_scores) / len(raw_scores)
        var = sum((score - mean) ** 2 for score in raw_scores) / len(raw_scores)
        scale = math.sqrt(var) if var > 1e-12 else (max_score - min_score)
        z_scores = [(score - mean) / max(1e-9, scale) for score in raw_scores]
        peak = max(z_scores)
        exps = [math.exp(max(-20.0, min(20.0, score - peak))) for score in z_scores]
        total = sum(exps) + 1e-12
        return [value / total for value in exps]


class BrainMemory:
    """Unified non-gradient event memory.

    The memory stores observed event features, compact integer co-activation
    links, ordered event traces, and HDC event vectors. It does
    not train dense weights.
    """

    def __init__(self, window_size: int = 12, decay: float = 0.72):
        self.window_size = int(window_size)
        self.decay = float(decay)
        self.events: Dict[str, EventRecord] = {}
        self.feature_freq: Counter[str] = Counter()
        self.total_feature_count = 0
        self.modality_freq: Counter[str] = Counter()
        self.label_freq: Counter[str] = Counter()
        self.feature_to_id: Dict[str, int] = {}
        self.id_to_feature: List[str] = []
        self.edges: Dict[Tuple[int, int], float] = {}
        self.edge_last_seen: Dict[Tuple[int, int], int] = {}
        self._neighbors: Dict[int, List[Tuple[int, float]]] = {}
        self._neighbors_dirty = True
        self.feature_to_events: Dict[int, Counter[str]] = defaultdict(Counter)
        self.feature_top_events: Dict[int, Counter[str]] = defaultdict(Counter)
        self.token_unigrams: Counter[str] = Counter()
        self.sequence_starts: Counter[str] = Counter()
        self.token_successors: Dict[str, Counter[str]] = defaultdict(Counter)
        self.order_contexts: Dict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.event_novelty: Dict[str, float] = {}
        self.event_content_sets: Dict[str, set[str]] = {}
        self.event_hv: Dict[str, np.ndarray] = {}
        self.hdc_words = dynamic_hdc_words(0, 0)
        self.hdc_bits = dynamic_hdc_bits(0, 0)
        self.hdc_indexed_bands = hdc_band_count(self.hdc_words, 0)
        self.hdc_index: Dict[Tuple[int, int], set[str]] = defaultdict(set)
        self.cluster_freq: Counter[Tuple[int, ...]] = Counter()
        self.concept_members: Dict[str, List[int]] = {}
        self.short_term_events: List[str] = []
        self.total_exposures = 0
        self._updates_since_prune = 0
        self._rank_feature_cache: Dict[str, List[str]] = {}
        self._rank_content_cache: Dict[str, List[str]] = {}
        self._rank_state_cache: Dict[str, Dict[str, Any]] = {}
        self._rank_choice_cache: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
        self._rank_result_cache: Dict[Tuple[int, str, Tuple[str, ...]], Tuple[int, List[float]]] = {}
        self._feature_weight_cache: Dict[str, float] = {}
        self.rank_mode = "event"

    def feature_id(self, feature: str) -> int:
        found = self.feature_to_id.get(feature)
        if found is not None:
            return found
        idx = len(self.id_to_feature)
        self.feature_to_id[feature] = idx
        self.id_to_feature.append(feature)
        return idx

    def expose_text(self, text: str, source: str = "text", label: Optional[str] = None) -> str:
        features, sequence, preview = self.extract_text(text)
        return self.expose_event("text", features, sequence, source=source, label=label, preview=preview)

    def expose_file(self, path: str | Path, label: Optional[str] = None) -> str:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            return self.expose_text(read_text_file_as_one_event(p), source=str(p), label=label)
        data = p.read_bytes()
        if suffix in IMAGE_SUFFIXES:
            features, sequence, preview = self.extract_image_bytes(data, suffix=suffix, source=str(p))
            modality = "image"
        elif suffix in AUDIO_SUFFIXES:
            features, sequence, preview = self.extract_audio_bytes(data, suffix=suffix, source=str(p))
            modality = "audio"
        else:
            features, sequence, preview = self.extract_binary(data, suffix=suffix, source=str(p))
            modality = "bytes"
        return self.expose_event(modality, features, sequence, source=str(p), label=label, preview=preview)

    def expose_event(
        self,
        modality: str,
        features: Iterable[str],
        sequence: Iterable[str] = (),
        source: str = "",
        label: Optional[str] = None,
        preview: str = "",
    ) -> str:
        uniq = list(dict.fromkeys(str(f) for f in features if f))
        seq = [str(s) for s in sequence if s]
        if label:
            uniq.append(f"label:{label.lower()}")
        identity_features = list(uniq)
        event_id = stable_id(json.dumps([modality, label, identity_features[:128], seq[:128]], ensure_ascii=False))
        novelty = self.novelty_score(uniq)
        now = self.total_exposures + 1
        existing = self.events.get(event_id)

        if existing is not None:
            if source and source not in existing.source.split(" | "):
                existing.source = " | ".join([existing.source, source]) if existing.source else source
            existing.last_accessed_at = now
            existing.novelty = max(existing.novelty, novelty)
            if preview and preview not in existing.preview:
                existing.preview = (existing.preview + " | " + preview)[:240] if existing.preview else preview[:240]
            self.event_novelty[event_id] = existing.novelty
            for feature in identity_features:
                fid = self.feature_id(feature)
                self.feature_freq[feature] += 1
                self.total_feature_count += 1
                self.feature_to_events[fid][event_id] += 1
                top_events = self.feature_top_events[fid]
                top_events[event_id] += 1
                if len(top_events) > self.dynamic_rank_event_cap() * 2:
                    self.feature_top_events[fid] = Counter(dict(top_events.most_common(self.dynamic_rank_event_cap())))
            self.total_exposures += 1
            self.learn_order_trace(seq)
            self.clear_rank_caches()
            self.remember_short_term(event_id)
            return event_id

        ids = [self.feature_id(f) for f in uniq]
        concept_features, concept_ids = self.observe_concepts(ids)
        if concept_features:
            uniq.extend(concept_features)
            ids.extend(concept_ids)
        self.refresh_dynamic_hdc_if_needed(extra_events=1)
        event_vec = self.bundle_event(uniq, seq)
        rec = EventRecord(event_id, modality, source, label, uniq, seq, preview[:240], novelty, event_vec, now, now)
        self.events[event_id] = rec
        self.event_novelty[event_id] = novelty
        self.event_content_sets[event_id] = set(content_features(rec.features))
        self.event_hv[event_id] = rec.hv
        self.index_event_hv(event_id, rec.hv)
        self.modality_freq[modality] += 1
        if label:
            self.label_freq[label] += 1

        for f, fid in zip(uniq, ids):
            self.feature_freq[f] += 1
            self.total_feature_count += 1
            self.feature_to_events[fid][event_id] += 1
            top_events = self.feature_top_events[fid]
            top_events[event_id] += 1
            if len(top_events) > self.dynamic_rank_event_cap() * 2:
                self.feature_top_events[fid] = Counter(dict(top_events.most_common(self.dynamic_rank_event_cap())))

        adaptive_window = max(2, min(self.window_size * 2, int(round(self.window_size * (0.5 + novelty)))))
        edge_gain = self.surprisal_gain(uniq)
        for i, left in enumerate(ids):
            limit = min(len(ids), i + adaptive_window + 1)
            for j in range(i + 1, limit):
                right = ids[j]
                if left == right:
                    continue
                weight = edge_gain * (self.decay ** (j - i - 1))
                self.add_edge(left, right, weight)
                self.add_edge(right, left, weight * 0.35)

        self.total_exposures += 1
        self.learn_order_trace(seq)
        self.clear_rank_caches()
        self.remember_short_term(event_id)
        self._updates_since_prune += len(ids)
        if self._updates_since_prune >= 4096:
            self.prune()
            self._updates_since_prune = 0
        return event_id

    def learn_order_trace(self, seq: List[str]) -> None:
        tokens = [symbol.split(":", 1)[1] for symbol in seq if symbol.startswith("text:")]
        max_order = self.dynamic_generation_order()
        if tokens:
            self.sequence_starts[tokens[0]] += 1
        for pos, token in enumerate(tokens):
            if not token:
                continue
            self.token_unigrams[token] += 1
            for order in range(1, min(max_order, pos) + 1):
                context = tuple(tokens[pos - order:pos])
                if context:
                    self.order_contexts[context][token] += 1
            if pos > 0 and tokens[pos - 1] and tokens[pos - 1] != token:
                self.token_successors[tokens[pos - 1]][token] += 1
        if len(self.token_unigrams) > 50000:
            self.token_unigrams = Counter(dict(self.token_unigrams.most_common(50000)))
        if len(self.sequence_starts) > 20000:
            self.sequence_starts = Counter(dict(self.sequence_starts.most_common(20000)))
        if len(self.token_successors) > 20000:
            for token, rows in list(self.token_successors.items()):
                self.token_successors[token] = Counter(dict(rows.most_common(32)))
        if len(self.order_contexts) > 50000:
            for context, rows in list(self.order_contexts.items()):
                self.order_contexts[context] = Counter(dict(rows.most_common(32)))

    def dynamic_generation_order(self) -> int:
        return max(2, min(5, self.dynamic_sequence_order()))

    def clear_rank_caches(self) -> None:
        self._rank_feature_cache.clear()
        self._rank_content_cache.clear()
        self._rank_state_cache.clear()
        self._rank_choice_cache.clear()
        self._rank_result_cache.clear()
        self._feature_weight_cache.clear()

    def refresh_dynamic_hdc_if_needed(self, extra_features: int = 0, extra_events: int = 0) -> None:
        target_words = dynamic_hdc_words(len(self.feature_to_id) + extra_features, len(self.events) + extra_events)
        target_bands = hdc_band_count(target_words, len(self.events) + extra_events)
        if target_words <= self.hdc_words and target_bands <= self.hdc_indexed_bands:
            return
        old_bands = self.hdc_indexed_bands
        old_words = self.hdc_words
        self.hdc_words = target_words
        self.hdc_bits = target_words * HDC_WORD_BITS
        self.hdc_indexed_bands = target_bands
        for eid, rec in self.events.items():
            full_vec = self.bundle_event(rec.features, rec.sequence)
            old_vec = self.event_hv.get(eid)
            if old_vec is not None and old_vec.size == old_words:
                full_vec[:old_words] = old_vec
            rec.hv = full_vec
            self.event_hv[eid] = full_vec
            self.index_event_hv(eid, full_vec, start_word=old_bands)

    def bundle_event(self, features: List[str], sequence: List[str]) -> np.ndarray:
        return bundle_hv(features, self.hdc_bits)

    def index_event_hv(self, event_id: str, event_vec: np.ndarray, start_word: int = 0) -> None:
        for band in hdc_bands(event_vec, max(1, len(self.events)), start_word=start_word):
            self.hdc_index[band].add(event_id)

    def observe_concepts(self, ids: List[int]) -> Tuple[List[str], List[int]]:
        content_ids = [
            fid for fid in dict.fromkeys(ids)
            if fid < len(self.id_to_feature)
            and not self.id_to_feature[fid].startswith(("mod:", "label:", "concept:"))
        ]
        if len(content_ids) < 3:
            return [], []
        width = max(3, int(math.sqrt(len(content_ids))) + 1)
        cluster = tuple(sorted(content_ids[:width]))
        self.cluster_freq[cluster] += 1
        threshold = self.dynamic_concept_threshold()
        if self.cluster_freq[cluster] < threshold:
            return [], []
        concept = f"concept:{stable_id(','.join(map(str, cluster)), 8)}"
        concept_id = self.feature_id(concept)
        self.concept_members[concept] = list(cluster)
        gain = math.log1p(self.cluster_freq[cluster])
        for member in cluster:
            self.add_edge(member, concept_id, gain)
            self.add_edge(concept_id, member, gain)
        return [concept], [concept_id]

    def dynamic_concept_threshold(self) -> int:
        return max(2, int(math.sqrt(max(1, self.total_exposures + 1))))

    def remember_short_term(self, event_id: str) -> None:
        self.short_term_events.append(event_id)
        limit = self.dynamic_short_term_limit()
        if len(self.short_term_events) > limit:
            del self.short_term_events[: len(self.short_term_events) - limit]

    def dynamic_short_term_limit(self) -> int:
        return max(4, int(math.sqrt(max(1, len(self.events)))) + self.dynamic_sequence_order())

    def add_edge(self, left: int, right: int, weight: float) -> None:
        if left == right:
            return
        key = (left, right)
        last_seen = self.edge_last_seen.get(key, self.total_exposures)
        age = max(0, self.total_exposures - last_seen)
        aged = self.edges.get(key, 0.0) * (self.decay ** min(age, 64))
        self.edges[key] = aged + weight
        self.edge_last_seen[key] = self.total_exposures
        self._neighbors_dirty = True

    def neighbors(self) -> Dict[int, List[Tuple[int, float]]]:
        if not self._neighbors_dirty:
            return self._neighbors
        graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (left, right), weight in self.edges.items():
            graph[left].append((right, weight))
        self._neighbors = dict(graph)
        self._neighbors_dirty = False
        return self._neighbors

    def dynamic_sequence_order(self) -> int:
        if self.total_exposures < 16:
            return 2
        entropy_proxy = math.log2(max(2, len(self.feature_freq)))
        return max(2, min(12, 2 + int(entropy_proxy // 2)))

    def novelty_score(self, features: List[str]) -> float:
        if not features or not self.feature_freq:
            return 1.0
        total = max(1, sum(self.feature_freq.values()))
        vocab = max(1, len(self.feature_freq))
        surprisal = 0.0
        for feature in features:
            prob = (self.feature_freq.get(feature, 0) + 1.0) / (total + vocab)
            surprisal += -math.log2(prob)
        return max(0.1, min(2.0, (surprisal / max(1, len(features))) / 8.0))

    def surprisal_gain(self, features: List[str]) -> float:
        return 0.5 + self.novelty_score(features)

    def extract_text(self, text: str) -> Tuple[List[str], List[str], str]:
        features = text_feature_list(text)
        sequence = text_sequence(text)
        return features, sequence, str(text).strip().replace("\n", " ")[:240]

    def extract_image_bytes(self, data: bytes, suffix: str = "", source: str = "") -> Tuple[List[str], List[str], str]:
        features = ["mod:image", f"image:suffix:{suffix}", f"image:size:{size_bucket(len(data))}"]
        features.extend(byte_histogram_features(data, "image"))
        features.extend(chunk_hash_features(data, "image"))
        features.extend(self.image_header_features(data))
        features.extend(bmp_perceptual_features(data))
        features.extend(self.optional_image_perceptual_features(data))
        sequence = [f for f in features if f.startswith("image:chunk:") or f.startswith("image:phash:")][:32]
        return features, sequence, f"{source} image bytes={len(data)}"

    def image_header_features(self, data: bytes) -> List[str]:
        return image_header_features(data)

    def optional_image_perceptual_features(self, data: bytes) -> List[str]:
        try:
            from PIL import Image
            import io
        except Exception:
            return []
        try:
            with Image.open(io.BytesIO(data)) as img:
                gray8 = img.convert("L").resize((8, 8))
                pixels = list(gray8.getdata())
                avg = sum(pixels) / max(1, len(pixels))
                ahash = 0
                for pix in pixels:
                    ahash = (ahash << 1) | int(pix >= avg)
                gray9 = img.convert("L").resize((9, 8))
                vals = list(gray9.getdata())
                dhash = 0
                for row in range(8):
                    base = row * 9
                    for col in range(8):
                        dhash = (dhash << 1) | int(vals[base + col] > vals[base + col + 1])
        except Exception:
            return []
        return [
            f"image:phash:a:{ahash >> 48:04x}:{(ahash >> 32) & 0xffff:04x}:{(ahash >> 16) & 0xffff:04x}:{ahash & 0xffff:04x}",
            f"image:phash:d:{dhash >> 48:04x}:{(dhash >> 32) & 0xffff:04x}:{(dhash >> 16) & 0xffff:04x}:{dhash & 0xffff:04x}",
        ]

    def extract_audio_bytes(self, data: bytes, suffix: str = "", source: str = "") -> Tuple[List[str], List[str], str]:
        features = ["mod:audio", f"audio:suffix:{suffix}", f"audio:size:{size_bucket(len(data))}"]
        features.extend(byte_histogram_features(data, "audio"))
        features.extend(chunk_hash_features(data, "audio"))
        if suffix == ".wav":
            features.extend(self.wav_features(data))
        sequence = [f for f in features if f.startswith("audio:chunk:") or f.startswith("audio:zcr:")][:32]
        return features, sequence, f"{source} audio bytes={len(data)}"

    def wav_features(self, data: bytes) -> List[str]:
        try:
            import io
            with wave.open(io.BytesIO(data), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                channels = wf.getnchannels()
                duration_ms = int((frames / max(1, rate)) * 1000)
                out = [
                    "audio:format:wav",
                    f"audio:rate:{rate}",
                    f"audio:channels:{channels}",
                    f"audio:duration_ms:{size_bucket(duration_ms)}",
                ]
                out.extend(wav_signal_features(wf))
                return out
        except wave.Error:
            return ["audio:format:wav_unreadable"]

    def extract_binary(self, data: bytes, suffix: str = "", source: str = "") -> Tuple[List[str], List[str], str]:
        features = ["mod:bytes", f"bytes:suffix:{suffix}", f"bytes:size:{size_bucket(len(data))}"]
        features.extend(byte_histogram_features(data, "bytes"))
        features.extend(chunk_hash_features(data, "bytes"))
        features.extend(printable_byte_features(data, "bytes"))
        sequence = [f for f in features if f.startswith("bytes:chunk:")][:32]
        return features, sequence, f"{source} bytes={len(data)}"

    def features_for_query(self, value: str) -> List[str]:
        if looks_like_file_query(value):
            path = Path(value)
            if path.exists() and path.is_file():
                suffix = path.suffix.lower()
                if suffix in TEXT_SUFFIXES:
                    return self.extract_text(read_text_file_as_one_event(path))[0]
                data = path.read_bytes()
                if suffix in IMAGE_SUFFIXES:
                    return self.extract_image_bytes(data, suffix=suffix, source=str(path))[0]
                if suffix in AUDIO_SUFFIXES:
                    return self.extract_audio_bytes(data, suffix=suffix, source=str(path))[0]
                return self.extract_binary(data, suffix=suffix, source=str(path))[0]
        return self.extract_text(value)[0]

    def cached_features_for_query(self, value: str) -> List[str]:
        key = str(value)
        cached = self._rank_feature_cache.get(key)
        if cached is not None:
            return cached
        features = self.features_for_query(key)
        if len(self._rank_feature_cache) > 4096:
            self._rank_feature_cache.clear()
            self._rank_content_cache.clear()
            self._rank_choice_cache.clear()
            self._rank_result_cache.clear()
        self._rank_feature_cache[key] = features
        self._rank_content_cache[key] = content_features(features)
        return features

    def cached_content_for_query(self, value: str) -> List[str]:
        key = str(value)
        cached = self._rank_content_cache.get(key)
        if cached is not None:
            return cached
        features = self.cached_features_for_query(key)
        content = content_features(features)
        self._rank_content_cache[key] = content
        return content

    def compiled_choices(self, choices: List[str]) -> List[Dict[str, Any]]:
        key = tuple(str(choice) for choice in choices)
        cached = self._rank_choice_cache.get(key)
        if cached is not None:
            return cached
        rows = []
        for choice in key:
            features = self.cached_features_for_query(choice)
            content = self.cached_content_for_query(choice)
            content_set = set(content)
            freq = sum(math.log1p(self.feature_freq.get(feature, 0)) for feature in content_set) / max(1, len(content_set))
            length_penalty = 1.0 + 0.08 * math.log1p(len(content)) + 0.03 * freq
            rows.append({
                "choice": choice,
                "features": features,
                "content": content,
                "content_set": content_set,
                "length_penalty": length_penalty,
                "freq_penalty": freq,
            })
        if len(self._rank_choice_cache) > 1024:
            self._rank_choice_cache.clear()
        self._rank_choice_cache[key] = rows
        return rows

    def activation(self, seed_features: Iterable[str], hops: Optional[int] = None, decay: Optional[float] = None) -> Dict[str, float]:
        energy_ids: Dict[int, float] = {}
        hop_count = hops if hops is not None else self.dynamic_hops()
        decay_value = decay if decay is not None else self.dynamic_activation_decay()
        for feature in seed_features:
            fid = self.feature_to_id.get(feature)
            if fid is not None:
                energy_ids[fid] = max(energy_ids.get(fid, 0.0), self.feature_weight(feature))
        for _ in range(max(0, hop_count)):
            next_energy: Dict[int, float] = {}
            graph = self.neighbors()
            for fid, energy in energy_ids.items():
                for neighbor, weight in graph.get(fid, ()):
                    neighbor_feature = self.id_to_feature[neighbor] if neighbor < len(self.id_to_feature) else ""
                    next_energy[neighbor] = next_energy.get(neighbor, 0.0) + energy * weight * decay_value * self.feature_weight(neighbor_feature)
            if next_energy:
                mean_energy = sum(next_energy.values()) / len(next_energy)
                for fid in list(next_energy.keys()):
                    if next_energy[fid] < mean_energy:
                        next_energy[fid] *= 0.5
            for fid, energy in next_energy.items():
                energy_ids[fid] = energy_ids.get(fid, 0.0) + energy
        return {self.id_to_feature[fid]: val for fid, val in energy_ids.items() if fid < len(self.id_to_feature)}

    def dynamic_hops(self) -> int:
        return max(1, min(4, 1 + int(math.log2(max(1, len(self.events))) // 4)))

    def dynamic_activation_decay(self) -> float:
        density = len(self.edges) / max(1, len(self.feature_to_id))
        return max(0.25, min(0.75, 0.65 - 0.02 * math.log1p(density)))

    def dynamic_recall_top_k(self) -> int:
        return max(4, min(64, int(math.sqrt(max(1, len(self.events)))) + 4))

    def dynamic_rank_event_cap(self) -> int:
        return max(4, min(16, int(math.log2(max(2, len(self.events)))) * 2))

    def feature_weight(self, feature: str) -> float:
        if not feature or not self.feature_freq:
            return 1.0
        cached = self._feature_weight_cache.get(feature)
        if cached is not None:
            return cached
        total = max(1, int(getattr(self, "total_feature_count", 0)) or sum(self.feature_freq.values()))
        vocab = max(1, len(self.feature_freq))
        freq = self.feature_freq.get(feature, 0)
        canonical = canonical_feature(feature)
        rarity = 1.0 + math.log1p((total + vocab) / (freq + 1.0)) / math.log2(total + vocab + 2.0)
        if canonical.startswith("id:"):
            value = rarity * 6.0
        elif canonical.startswith(("bi:", "tri:")):
            value = rarity * 2.0
        elif canonical.startswith("tok:"):
            value = rarity * 1.5
        elif canonical.startswith("c3:"):
            value = rarity * 0.25
        elif ":ascii:" in canonical or ":phash:" in canonical or ":zcr" in canonical:
            value = rarity * 2.5
        else:
            value = rarity
        if len(self._feature_weight_cache) > 8192:
            self._feature_weight_cache.clear()
        self._feature_weight_cache[feature] = value
        return value

    def retrieval_feature_weight(self, feature: str, anchor_present: bool = False) -> float:
        canonical = canonical_feature(feature)
        if canonical.startswith("c3:"):
            return 0.02 if anchor_present else 0.15
        if canonical.startswith("id:"):
            return self.feature_weight(feature) * 4.0
        return self.feature_weight(feature)

    def anchor_features(self, features: Iterable[str]) -> List[str]:
        return [feature for feature in dict.fromkeys(features) if canonical_feature(feature).startswith("id:")]

    def expanded_query_features(self, query_features: List[str], limit: Optional[int] = None) -> List[str]:
        base = list(dict.fromkeys(query_features))
        if not base:
            return []
        expansion_limit = limit if limit is not None else max(len(base), int(math.sqrt(max(1, len(self.feature_to_id)))) + self.dynamic_sequence_order())
        energy = self.activation(base, hops=1)
        base_set = set(base)
        ranked = sorted(
            (
                (feature, value * self.feature_weight(feature))
                for feature, value in energy.items()
                if feature not in base_set and not feature.startswith("mod:")
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        expanded = base + [feature for feature, _ in ranked[: max(0, expansion_limit - len(base))]]
        return expanded

    def recency_weight(self, event_id: str) -> float:
        rec = self.events.get(event_id)
        if rec is None:
            return 1.0
        age = max(0, self.total_exposures - rec.last_accessed_at)
        half_life = max(1.0, math.sqrt(max(1, len(self.events))) * self.dynamic_sequence_order())
        return 1.0 / (1.0 + age / half_life)

    def hdc_candidates(self, state: np.ndarray) -> set[str]:
        candidates: set[str] = set()
        for band in hdc_bands(state, max(1, len(self.events))):
            candidates.update(self.hdc_index.get(band, ()))
        return candidates

    def hopfield_recall(self, query_features: List[str], iterations: int = 2, top_k: Optional[int] = None) -> Dict[str, float]:
        state = bundle_hv(query_features, self.hdc_bits)
        if state.size == 0 or not self.event_hv:
            return {}
        k = top_k if top_k is not None else self.dynamic_recall_top_k()
        event_scores: Dict[str, float] = {}
        for _ in range(max(1, iterations)):
            candidates = self.hdc_candidates(state)
            if not candidates:
                break
            ranked = sorted(
                ((eid, hv_similarity(state, self.event_hv[eid], self.hdc_bits)) for eid in candidates),
                key=lambda item: item[1],
                reverse=True,
            )[:k]
            event_scores = {eid: score for eid, score in ranked if score > 0.45}
            if not event_scores:
                break
            vectors = [self.event_hv[eid] for eid in event_scores if eid in self.event_hv]
            if not vectors:
                break
            state = np.bitwise_xor.reduce(np.stack(vectors), axis=0).astype(np.uint64, copy=False)
        return event_scores

    def hopfield_recall_continuous(
        self,
        query_features: List[str],
        iterations: int = 2,
        top_k: Optional[int] = None,
        beta: float = 8.0,
        max_patterns: int = 512,
    ) -> Dict[str, float]:
        query_vec = bundle_hv(query_features, self.hdc_bits)
        if query_vec.size == 0 or not self.event_hv:
            return {}
        candidate_ids = list(self.event_hv.keys())
        if len(candidate_ids) > max_patterns:
            indexed = self.hdc_candidates(query_vec)
            if indexed:
                candidate_ids = list(indexed)
        if not candidate_ids:
            return {}
        if len(candidate_ids) > max_patterns:
            candidate_ids = sorted(
                candidate_ids,
                key=lambda eid: hv_similarity(query_vec, self.event_hv[eid], self.hdc_bits),
                reverse=True,
            )[:max_patterns]
        patterns = np.stack([self.hv_bipolar(self.event_hv[eid]) for eid in candidate_ids]).astype(np.float32, copy=False)
        state = self.hv_bipolar(query_vec).astype(np.float32, copy=False)
        norm = max(1.0, float(state.size))
        for _ in range(max(1, iterations)):
            logits = (patterns @ state) / norm
            logits = logits * float(beta)
            logits = logits - float(np.max(logits))
            weights = np.exp(logits)
            weights = weights / (float(np.sum(weights)) + 1e-12)
            state = weights @ patterns
            state_norm = float(np.linalg.norm(state))
            if state_norm > 1e-9:
                state = state / state_norm * math.sqrt(norm)
        scores = (patterns @ state) / norm
        ranked = sorted(zip(candidate_ids, scores.tolist()), key=lambda item: item[1], reverse=True)
        k = top_k if top_k is not None else self.dynamic_recall_top_k()
        return {eid: float(score) for eid, score in ranked[:k] if score > 0.0}

    def hv_bipolar(self, value: np.ndarray) -> np.ndarray:
        if value.size == 0:
            return np.zeros(0, dtype=np.float32)
        bits = np.unpackbits(value[: self.hdc_words].view(np.uint8)).astype(np.float32, copy=False)
        return bits * 2.0 - 1.0

    def hopfield_recall_feature_continuous(
        self,
        query_features: List[str],
        iterations: int = 2,
        top_k: Optional[int] = None,
        beta: float = 4.0,
        max_patterns: int = 512,
    ) -> Dict[str, float]:
        query_content = content_features(query_features)
        if not query_content or not self.events:
            return {}
        candidate_ids: set[str] = set()
        anchors = self.anchor_features(query_features)
        anchor_ids: set[str] = set()
        for anchor in anchors:
            fid = self.feature_to_id.get(anchor)
            if fid is not None:
                anchor_ids.update(self.feature_to_events.get(fid, {}).keys())
        if anchor_ids:
            candidate_ids = anchor_ids
        for feature in ([] if anchor_ids else [*anchors, *query_content]):
            fid = self.feature_to_id.get(feature)
            if fid is None:
                continue
            candidate_ids.update(self.feature_to_events.get(fid, {}).keys())
        if not candidate_ids:
            candidate_ids = set(self.events.keys())
        if len(candidate_ids) > max_patterns:
            seed = Counter()
            for feature in query_content:
                fid = self.feature_to_id.get(feature)
                if fid is None:
                    continue
                for eid, count in self.feature_to_events.get(fid, {}).items():
                    seed[eid] += count * self.retrieval_feature_weight(feature, anchor_present=bool(anchors))
            candidate_ids = set(eid for eid, _ in seed.most_common(max_patterns))
        state = Counter({feature: self.retrieval_feature_weight(feature, anchor_present=bool(anchors)) for feature in query_content})
        event_sets: Dict[str, set[str]] = {}
        event_norms: Dict[str, float] = {}
        for eid in candidate_ids:
            rec = self.events.get(eid)
            if rec is None:
                continue
            features = self.event_content_sets.get(eid)
            if features is None:
                features = set(content_features(rec.features))
                self.event_content_sets[eid] = features
            event_sets[eid] = features
            event_norms[eid] = math.sqrt(max(1e-9, sum(self.retrieval_feature_weight(f, anchor_present=bool(anchors)) for f in features)))
        scores: Dict[str, float] = {}
        for _ in range(max(1, iterations)):
            state_norm = math.sqrt(max(1e-9, sum(value * value for value in state.values())))
            logits = []
            for eid, features in event_sets.items():
                shared = sum(state.get(feature, 0.0) for feature in features)
                logits.append((eid, shared / max(1e-9, state_norm * event_norms[eid])))
            if not logits:
                break
            peak = max(score for _, score in logits)
            weights = [(eid, math.exp((score - peak) * float(beta))) for eid, score in logits]
            total = sum(weight for _, weight in weights) + 1e-12
            scores = {eid: weight / total for eid, weight in weights}
            next_state: Counter[str] = Counter()
            for eid, weight in scores.items():
                for feature in event_sets[eid]:
                    next_state[feature] += weight * self.retrieval_feature_weight(feature, anchor_present=bool(anchors))
            state = next_state
        k = top_k if top_k is not None else self.dynamic_recall_top_k()
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return {eid: float(score) for eid, score in ranked if score > 0.0}

    def score(self, query: str, choice: str) -> float:
        state = self._rank_query_state(query)
        compiled = self.compiled_choices([choice])[0]
        if not state["query_signal"] or not compiled["features"]:
            return 0.0
        return self._score_compiled_choice_from_state(state, compiled)

    def _rank_query_state(self, query: str) -> Dict[str, Any]:
        cached_state = self._rank_state_cache.get(str(query))
        if cached_state is not None:
            return cached_state
        query_features = self.cached_features_for_query(query)
        if not query_features:
            return {
                "query_signal": [],
                "query_set": set(),
                "energy": {},
                "event_scores": Counter(),
            }
        anchors = self.anchor_features(query_features)
        expanded_query = list(dict.fromkeys([*anchors, *query_features[:64]]))
        query_signal = self.cached_content_for_query(query)
        anchor_present = bool(anchors)
        energy = {feature: self.retrieval_feature_weight(feature, anchor_present=anchor_present) for feature in expanded_query}
        event_scores: Counter[str] = Counter()
        cap = self.dynamic_rank_event_cap()
        for anchor in anchors:
            fid = self.feature_to_id.get(anchor)
            if fid is None:
                continue
            for eid, count in self.feature_to_events.get(fid, {}).items():
                event_scores[eid] += self.retrieval_feature_weight(anchor, anchor_present=True) * max(1, count) * 100.0
        for feature, value in energy.items():
            fid = self.feature_to_id.get(feature)
            if fid is None:
                continue
            rows = self.feature_top_events.get(fid)
            if not rows:
                rows = self.feature_to_events.get(fid, {})
            for eid, count in rows.items():
                event_scores[eid] += value * count * self.recency_weight(eid)
        state = {
            "query_signal": query_signal,
            "query_set": set(query_signal),
            "energy": energy,
            "event_scores": event_scores,
        }
        if len(self._rank_state_cache) > 1024:
            self._rank_state_cache.clear()
        self._rank_state_cache[str(query)] = state
        return state

    def _score_choice_from_state(self, state: Dict[str, Any], choice_features: List[str]) -> float:
        return self._score_compiled_choice_from_state(
            state,
            {
                "features": choice_features,
                "content": content_features(choice_features),
                "content_set": set(content_features(choice_features)),
                "freq_penalty": 0.0,
            },
        )

    def _score_compiled_choice_from_state(self, state: Dict[str, Any], compiled_choice: Dict[str, Any]) -> float:
        return self.mode_score(self.energy_score(state, compiled_choice), self.rank_mode)

    def energy_score(self, state: Dict[str, Any], compiled_choice: Dict[str, Any]) -> Dict[str, float]:
        choice_signal = compiled_choice["content"]
        if not choice_signal:
            return {"score": 0.0, "free_energy": 1e9}
        energy = state["energy"]
        query_set = state["query_set"]
        choice_set = compiled_choice["content_set"]
        overlap = len(query_set & choice_set) / math.sqrt(max(1.0, float(len(query_set) * len(choice_set))))
        weighted_overlap = self.weighted_feature_overlap(query_set, choice_set)
        resonance = sum(energy.get(feature, 0.0) for feature in choice_set) / math.sqrt(max(1.0, float(len(choice_set))))
        event_support = self.event_support_from_scores(state["event_scores"], choice_set)
        alignment = self.event_alignment_from_scores(state["event_scores"], choice_set)
        contrast = self.choice_contrast(state, choice_set)
        freq_penalty = compiled_choice.get("freq_penalty", 0.0)
        evidence = (
            1.0 * overlap
            + 2.2 * weighted_overlap
            + 1.6 * event_support
            + 1.2 * alignment
            + 0.7 * contrast
            + resonance / (1.0 + 0.05 * freq_penalty)
        )
        entropy = self.choice_entropy(choice_set)
        conflict = self.choice_conflict(query_set, choice_set)
        free_energy = conflict + 0.20 * entropy + 0.05 * freq_penalty - evidence
        return {
            "score": evidence - 0.20 * entropy - 0.05 * freq_penalty - conflict,
            "free_energy": free_energy,
            "overlap": overlap,
            "weighted_overlap": weighted_overlap,
            "resonance": resonance,
            "event_support": event_support,
            "alignment": alignment,
            "contrast": contrast,
            "entropy": entropy,
            "conflict": conflict,
            "freq_penalty": freq_penalty,
        }

    def mode_score(self, breakdown: Dict[str, float], mode: Optional[str] = None) -> float:
        selected = mode or self.rank_mode
        weights = ENERGY_MODES.get(selected, ENERGY_MODES["free_energy"])
        return sum(float(breakdown.get(name, 0.0)) * weight for name, weight in weights.items())

    def choice_entropy(self, choice_set: set[str]) -> float:
        if not choice_set:
            return 0.0
        weights = [1.0 / max(1e-9, self.feature_weight(feature)) for feature in choice_set]
        total = sum(weights) + 1e-12
        entropy = 0.0
        for weight in weights:
            p = weight / total
            entropy -= p * math.log(p + 1e-12)
        return entropy / math.log(len(weights) + 1.0)

    def choice_conflict(self, query_set: set[str], choice_set: set[str]) -> float:
        if not query_set or not choice_set:
            return 1.0
        missing = choice_set - query_set
        common = query_set & choice_set
        missing_cost = sum(1.0 / max(1e-9, self.feature_weight(feature)) for feature in missing)
        common_gain = sum(self.feature_weight(feature) for feature in common)
        return missing_cost / (missing_cost + common_gain + 1e-9)

    def rank_energy_breakdown(self, query: str, choices: List[str]) -> List[Dict[str, float]]:
        state = self._rank_query_state(query)
        return [self.energy_score(state, row) for row in self.compiled_choices(choices)]

    def explain_rank(self, query: str, choices: List[str], mode: Optional[str] = None, top_events: int = 5) -> Dict[str, Any]:
        selected_mode = mode or self.rank_mode
        state = self._rank_query_state(query)
        compiled = self.compiled_choices(choices)
        breakdown = [self.energy_score(state, row) for row in compiled]
        raw_scores = [self.mode_score(row, selected_mode) for row in breakdown]
        scores = self.normalize_compiled_choice_scores(raw_scores, compiled)
        prediction = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
        event_rows = []
        for eid, event_score in state["event_scores"].most_common(max(1, top_events)):
            rec = self.events.get(eid)
            if rec is None:
                continue
            event_features = self.event_content_sets.get(eid)
            if event_features is None:
                event_features = set(content_features(rec.features))
                self.event_content_sets[eid] = event_features
            event_rows.append({
                "event_id": eid,
                "score": float(event_score),
                "source": rec.source,
                "modality": rec.modality,
                "preview": rec.preview,
                "shared_query_features": sorted(set(state["query_signal"]) & event_features)[:16],
            })
        choice_rows = []
        for idx, row in enumerate(compiled):
            choice_rows.append({
                "choice": row["choice"],
                "score": scores[idx] if idx < len(scores) else 0.0,
                "raw_score": raw_scores[idx] if idx < len(raw_scores) else 0.0,
                "breakdown": breakdown[idx],
                "shared_query_features": sorted(state["query_set"] & row["content_set"])[:16],
            })
        return {
            "query": query,
            "mode": selected_mode,
            "prediction": prediction,
            "answer": choices[prediction] if choices and prediction < len(choices) else "",
            "choices": choice_rows,
            "events": event_rows,
        }

    def inspect_events(
        self,
        source: Optional[str] = None,
        modality: Optional[str] = None,
        contains: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        contains_text = str(contains).lower() if contains else ""
        rows = []
        for eid, rec in self.events.items():
            if source and source not in rec.source:
                continue
            if modality and modality != rec.modality:
                continue
            haystack = f"{rec.source}\n{rec.preview}\n{' '.join(rec.features[:64])}".lower()
            if contains_text and contains_text not in haystack:
                continue
            rows.append({
                "event_id": eid,
                "source": rec.source,
                "modality": rec.modality,
                "label": rec.label,
                "preview": rec.preview,
                "feature_count": len(rec.features),
                "created_at": rec.created_at,
                "last_accessed_at": rec.last_accessed_at,
            })
            if len(rows) >= max(1, limit):
                break
        return rows

    def rebuild_runtime_indexes(self) -> None:
        self.feature_to_events = defaultdict(Counter)
        self.feature_top_events = defaultdict(Counter)
        self.event_content_sets = {}
        self.event_hv = {}
        self.token_successors = defaultdict(Counter)
        self.order_contexts = defaultdict(Counter)
        self.hdc_words = dynamic_hdc_words(len(self.feature_to_id), len(self.events))
        self.hdc_bits = self.hdc_words * HDC_WORD_BITS
        self.hdc_index = defaultdict(set)
        self.hdc_indexed_bands = hdc_band_count(self.hdc_words, len(self.events))
        for eid, rec in self.events.items():
            content = set(content_features(rec.features))
            self.event_content_sets[eid] = content
            for feature in rec.features:
                fid = self.feature_to_id.get(feature)
                if fid is not None:
                    self.feature_to_events[fid][eid] += 1
                    self.feature_top_events[fid][eid] += 1
            rec.hv = self.bundle_event(rec.features, rec.sequence)
            self.event_hv[eid] = rec.hv
            self.index_event_hv(eid, rec.hv)
            self.learn_order_trace(rec.sequence)
        cap = self.dynamic_rank_event_cap()
        for fid, rows in list(self.feature_top_events.items()):
            self.feature_top_events[fid] = Counter(dict(rows.most_common(cap)))
        self.short_term_events = [eid for eid in self.short_term_events if eid in self.events][-self.dynamic_short_term_limit():]
        self.clear_rank_caches()

    def forget_events(
        self,
        event_ids: Optional[Iterable[str]] = None,
        source: Optional[str] = None,
        modality: Optional[str] = None,
        contains: Optional[str] = None,
    ) -> Dict[str, Any]:
        explicit_ids = set(str(eid) for eid in (event_ids or ()) if str(eid))
        contains_text = str(contains).lower() if contains else ""
        remove_ids = set()
        for eid, rec in self.events.items():
            matched = eid in explicit_ids if explicit_ids else True
            if matched and source:
                matched = source in rec.source
            if matched and modality:
                matched = modality == rec.modality
            if matched and contains_text:
                haystack = f"{rec.source}\n{rec.preview}\n{' '.join(rec.features[:64])}".lower()
                matched = contains_text in haystack
            if matched:
                remove_ids.add(eid)
        if not remove_ids:
            return {"removed": 0, "event_ids": []}

        removed_features: Counter[str] = Counter()
        removed_modalities: Counter[str] = Counter()
        removed_labels: Counter[str] = Counter()
        for eid in remove_ids:
            rec = self.events.pop(eid, None)
            if rec is None:
                continue
            removed_modalities[rec.modality] += 1
            if rec.label:
                removed_labels[rec.label] += 1
            for feature in rec.features:
                removed_features[feature] += 1
            self.event_novelty.pop(eid, None)

        for feature, count in removed_features.items():
            current = self.feature_freq.get(feature, 0) - count
            if current > 0:
                self.feature_freq[feature] = current
            else:
                self.feature_freq.pop(feature, None)
        self.total_feature_count = max(0, self.total_feature_count - sum(removed_features.values()))
        for modality_name, count in removed_modalities.items():
            current = self.modality_freq.get(modality_name, 0) - count
            if current > 0:
                self.modality_freq[modality_name] = current
            else:
                self.modality_freq.pop(modality_name, None)
        for label_name, count in removed_labels.items():
            current = self.label_freq.get(label_name, 0) - count
            if current > 0:
                self.label_freq[label_name] = current
            else:
                self.label_freq.pop(label_name, None)

        active_feature_ids = {self.feature_to_id[feature] for feature in self.feature_freq if feature in self.feature_to_id}
        self.edges = {
            key: weight for key, weight in self.edges.items()
            if key[0] in active_feature_ids and key[1] in active_feature_ids
        }
        self.edge_last_seen = {key: seen for key, seen in self.edge_last_seen.items() if key in self.edges}
        self.cluster_freq.clear()
        self.concept_members = {
            concept: [fid for fid in members if fid in active_feature_ids]
            for concept, members in self.concept_members.items()
            if concept in self.feature_freq
        }
        self._neighbors_dirty = True
        self.rebuild_runtime_indexes()
        return {"removed": len(remove_ids), "event_ids": sorted(remove_ids)}

    def weighted_feature_overlap(self, query_set: set[str], choice_set: set[str]) -> float:
        shared = query_set & choice_set
        if not shared:
            return 0.0
        numerator = sum(self.feature_weight(feature) for feature in shared)
        denominator = math.sqrt(
            max(1e-9, sum(self.feature_weight(feature) for feature in query_set))
            * max(1e-9, sum(self.feature_weight(feature) for feature in choice_set))
        )
        return numerator / denominator

    def event_support_from_energy(self, energy: Dict[str, float], choice_set: set[str]) -> float:
        event_scores: Counter[str] = Counter()
        for feature, value in energy.items():
            fid = self.feature_to_id.get(feature)
            if fid is None:
                continue
            for eid, count in self.feature_to_events.get(fid, {}).items():
                event_scores[eid] += value * count
        support = 0.0
        for eid, event_score in event_scores.most_common(8):
            event_features = set(content_features(self.events[eid].features))
            shared = len(event_features & choice_set) / max(1, len(choice_set))
            support = max(support, shared * math.log1p(event_score))
        return support

    def event_support_from_scores(self, event_scores: Counter[str], choice_set: set[str]) -> float:
        if not event_scores or not choice_set:
            return 0.0
        support = 0.0
        for eid, event_score in event_scores.most_common(8):
            rec = self.events.get(eid)
            if rec is None:
                continue
            event_features = self.event_content_sets.get(eid)
            if event_features is None:
                event_features = set(content_features(rec.features))
                self.event_content_sets[eid] = event_features
            shared = len(event_features & choice_set) / math.sqrt(max(1.0, float(len(event_features) * len(choice_set))))
            support = max(support, shared * math.log1p(event_score))
        return support

    def event_alignment_from_scores(self, event_scores: Counter[str], choice_set: set[str]) -> float:
        if not event_scores or not choice_set:
            return 0.0
        numerator = 0.0
        denominator = 0.0
        for eid, event_score in event_scores.most_common(8):
            rec = self.events.get(eid)
            if rec is None:
                continue
            event_features = self.event_content_sets.get(eid)
            if event_features is None:
                event_features = set(content_features(rec.features))
                self.event_content_sets[eid] = event_features
            shared = len(event_features & choice_set) / max(1, len(event_features | choice_set))
            weight = math.log1p(event_score)
            numerator += shared * weight
            denominator += weight
        return numerator / max(1e-9, denominator)

    def choice_contrast(self, state: Dict[str, Any], choice_set: set[str]) -> float:
        if not choice_set:
            return 0.0
        query_set = state["query_set"]
        if not query_set:
            return 0.0
        shared = query_set & choice_set
        only_choice = choice_set - query_set
        positive = sum(self.feature_weight(feature) for feature in shared)
        negative = sum(1.0 / self.feature_weight(feature) for feature in only_choice)
        return positive / (positive + negative + 1e-9)

    def rank(self, query: str, choices: List[str], mode: Optional[str] = None) -> Tuple[int, List[float]]:
        selected_mode = mode or self.rank_mode
        result_key = (self.total_exposures, selected_mode, str(query), tuple(str(choice) for choice in choices))
        cached = self._rank_result_cache.get(result_key)
        if cached is not None:
            return cached
        state = self._rank_query_state(query)
        compiled = self.compiled_choices(choices)
        raw_scores = [self.mode_score(self.energy_score(state, row), selected_mode) for row in compiled]
        scores = self.normalize_compiled_choice_scores(raw_scores, compiled)
        if not scores:
            return 0, []
        result = (max(range(len(scores)), key=lambda i: scores[i]), scores)
        if len(self._rank_result_cache) > 2048:
            self._rank_result_cache.clear()
        self._rank_result_cache[result_key] = result
        return result

    def normalize_choice_scores(self, raw_scores: List[float], choices: List[str]) -> List[float]:
        return self.normalize_compiled_choice_scores(raw_scores, self.compiled_choices(choices))

    def normalize_compiled_choice_scores(self, raw_scores: List[float], compiled_choices: List[Dict[str, Any]]) -> List[float]:
        if not raw_scores:
            return []
        adjusted = []
        for score, row in zip(raw_scores, compiled_choices):
            adjusted.append(float(score) / max(1e-9, float(row["length_penalty"])))
        if len(adjusted) == 1:
            return [1.0]
        max_score = max(adjusted)
        min_score = min(adjusted)
        if max_score - min_score < 1e-12:
            return [1.0 / len(adjusted) for _ in adjusted]
        mean = sum(adjusted) / len(adjusted)
        var = sum((score - mean) ** 2 for score in adjusted) / len(adjusted)
        scale = math.sqrt(var) if var > 1e-12 else (max_score - min_score)
        z_scores = [(score - mean) / max(1e-9, scale) for score in adjusted]
        peak = max(z_scores)
        exps = [math.exp(max(-20.0, min(20.0, score - peak))) for score in z_scores]
        total = sum(exps) + 1e-12
        return [value / total for value in exps]

    def recall(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_features = self.features_for_query(query)
        expanded_query = self.expanded_query_features(query_features)
        energy = self.activation(expanded_query)
        event_scores: Counter[str] = Counter()
        for feature, value in energy.items():
            fid = self.feature_to_id.get(feature)
            if fid is None:
                continue
            for eid, count in self.feature_to_events.get(fid, {}).items():
                event_scores[eid] += value * count * self.recency_weight(eid)
        for recent_id in self.short_term_events:
            if recent_id in self.events:
                rec = self.events[recent_id]
                recent_overlap = len(set(content_features(rec.features)) & set(content_features(query_features)))
                if recent_overlap:
                    event_scores[recent_id] += recent_overlap * self.recency_weight(recent_id)
        for eid, score in self.hopfield_recall(expanded_query, iterations=2, top_k=max(self.dynamic_recall_top_k(), limit * 2)).items():
            event_scores[eid] += score * 10.0 * self.recency_weight(eid)
        rows = []
        for eid, score in event_scores.most_common(limit):
            rec = self.events[eid]
            rec.last_accessed_at = self.total_exposures
            rows.append({
                "event_id": eid,
                "score": float(score),
                "modality": rec.modality,
                "source": rec.source,
                "label": rec.label,
                "preview": rec.preview,
                "novelty": rec.novelty,
            })
        return rows

    def answer(self, prompt: str, max_new_tokens: int = 24) -> str:
        tokens = tokenize(prompt, max_tokens=64)
        if not tokens:
            return "I need input."
        generated = self.compose_answer(prompt, max_new_tokens=max_new_tokens)
        if generated:
            return generated
        return "I do not have enough memory for that yet."

    def compose_answer(self, prompt: str, max_new_tokens: int = 24) -> str:
        tokens = tokenize(prompt, max_tokens=64)
        energy_text = self.memory_energy_decode_text(prompt, tokens, max_new_tokens=max_new_tokens)
        if energy_text:
            return energy_text
        recalls = self.recall(prompt, limit=max(4, min(12, int(math.sqrt(max(1, len(self.events)))) + 3)))
        candidates = self.answer_candidates(prompt, tokens, recalls, max_new_tokens=max_new_tokens)
        if not candidates:
            return ""
        unique_candidates = list(dict.fromkeys(candidate for candidate in candidates if candidate.strip()))
        if len(unique_candidates) == 1:
            return self.trim_answer(unique_candidates[0], max_new_tokens)
        pred, scores = self.rank(prompt, unique_candidates, mode="event")
        chosen = unique_candidates[pred]
        return self.trim_answer(chosen, max_new_tokens)

    def answer_candidates(
        self,
        prompt: str,
        tokens: List[str],
        recalls: List[Dict[str, Any]],
        max_new_tokens: int = 24,
    ) -> List[str]:
        candidates = []
        analogy = self.answer_by_analogy(tokens, max_new_tokens)
        if analogy:
            candidates.append(analogy)
        graph_text = self.graph_decode_text(prompt, tokens, max_new_tokens=max_new_tokens)
        if graph_text:
            candidates.append(graph_text)
        ranked_fragments = self.recall_fragments(prompt, recalls)
        candidates.extend(ranked_fragments[:6])
        chain = self.event_chain_summary(prompt, recalls, limit=max(8, max_new_tokens))
        if chain:
            candidates.append(chain)
        if ranked_fragments:
            candidates.append(" ".join(ranked_fragments[: min(3, len(ranked_fragments))]))
        return candidates

    def memory_energy_decode_text(
        self,
        prompt: str,
        tokens: List[str],
        max_new_tokens: int = 24,
        temperature: float = 0.0,
        beam_size: int = 1,
    ) -> str:
        if not tokens:
            return ""
        if beam_size > 1:
            return self.energy_beam_decode_text(prompt, tokens, max_new_tokens=max_new_tokens, beam_size=beam_size)
        context = list(tokens[-min(8, len(tokens)):])
        query_features = self.features_for_query(prompt)
        generation_features = self.generation_query_features(query_features)
        event_scores = self.event_scores_for_features(generation_features)
        semantic_energy = self.semantic_token_energy(query_features, event_scores)
        for _ in range(max(1, int(max_new_tokens))):
            scores = self.next_token_energy_scores(context, semantic_energy, event_scores)
            if not scores:
                break
            next_token = self.select_next_token(scores, context, temperature=temperature)
            if not next_token:
                break
            context.append(next_token)
        if len(context) <= len(tokens[-min(8, len(tokens)):]):
            return ""
        return " ".join(context)

    def event_scores_for_features(self, features: Iterable[str]) -> Counter[str]:
        scores: Counter[str] = Counter()
        for feature in features:
            fid = self.feature_to_id.get(feature)
            if fid is None:
                continue
            weight = self.retrieval_feature_weight(feature, anchor_present=canonical_feature(feature).startswith("id:"))
            for eid, count in self.feature_to_events.get(fid, {}).items():
                scores[eid] += weight * count
        return scores

    def generation_query_features(self, query_features: Iterable[str]) -> List[str]:
        strong_prefixes = ("id:", "tok:", "bi:", "tri:")
        out = []
        for feature in query_features:
            canonical = canonical_feature(feature)
            if canonical.startswith(strong_prefixes):
                out.append(feature)
        return list(dict.fromkeys(out))

    def semantic_token_energy(self, query_features: Iterable[str], event_scores: Counter[str], limit: int = 8) -> Counter[str]:
        scores: Counter[str] = Counter()
        for feature in query_features:
            token = self.token_from_feature(feature)
            if token:
                scores[token] += self.retrieval_feature_weight(feature)
        for eid, event_score in event_scores.most_common(max(1, int(limit))):
            rec = self.events.get(eid)
            if not rec:
                continue
            gain = math.log1p(float(event_score))
            for symbol in rec.sequence:
                if not symbol.startswith("text:"):
                    continue
                token = symbol.split(":", 1)[1]
                if token:
                    scores[token] += gain * 0.25
        return scores

    def next_token_energy_scores(
        self,
        context_tokens: List[str],
        semantic_energy: Dict[str, float],
        event_scores: Counter[str],
    ) -> Counter[str]:
        scores: Counter[str] = Counter()
        order_options, matched_order = self.order_backoff_options(context_tokens)
        for token, count in order_options.items():
            scores[token] += math.log1p(count) * (1.0 + matched_order)
        event_options = self.event_next_token_options(context_tokens, event_scores)
        for token, value in event_options.items():
            scores[token] += value * 1.4
        if not scores:
            for token, value in self.global_next_token_options(context_tokens).items():
                scores[token] += value
        for feature, value in semantic_energy.items():
            token = self.token_from_feature(feature) or str(feature)
            if token in scores:
                scores[token] += float(value) * 0.20
        if not scores:
            return scores
        recent = Counter(context_tokens[-8:])
        for token in list(scores.keys()):
            if recent[token]:
                scores[token] *= 1.0 / ((1.0 + recent[token]) ** 2)
            if len(context_tokens) >= 2 and any(context_tokens[i] == context_tokens[-1] and context_tokens[i + 1] == token for i in range(0, len(context_tokens) - 1)):
                scores[token] *= 0.05
        return Counter({token: score for token, score in scores.items() if score > 1e-9})

    def order_backoff_options(self, context_tokens: List[str]) -> Tuple[Counter[str], int]:
        max_order = min(self.dynamic_generation_order(), len(context_tokens))
        for order in range(max_order, 0, -1):
            options = self.order_contexts.get(tuple(context_tokens[-order:]))
            if options:
                return Counter(options), order
        return Counter(), 0

    def global_next_token_options(self, context_tokens: List[str], limit: int = 64) -> Counter[str]:
        scores: Counter[str] = Counter()
        recent = set(context_tokens[-8:])
        if context_tokens:
            for token, count in self.token_successors.get(context_tokens[-1], Counter()).most_common(limit):
                if token not in recent:
                    scores[token] += math.log1p(count) * 0.75
        if scores:
            return scores
        for token, count in self.sequence_starts.most_common(limit):
            if token not in recent:
                scores[token] += math.log1p(count) * 0.55
        if scores:
            return scores
        for token, count in self.token_unigrams.most_common(limit):
            if token not in recent:
                scores[token] += math.log1p(count) * 0.35
        return scores

    def event_next_token_options(self, context_tokens: List[str], event_scores: Counter[str], limit: int = 8) -> Counter[str]:
        out: Counter[str] = Counter()
        if not context_tokens:
            return out
        recent = set(context_tokens[-6:])
        for eid, score in event_scores.most_common(limit):
            rec = self.events.get(eid)
            if not rec or not rec.sequence:
                continue
            seq = [symbol.split(":", 1)[1] for symbol in rec.sequence if symbol.startswith("text:")]
            if not seq:
                continue
            max_width = min(self.dynamic_generation_order(), len(context_tokens), len(seq))
            for width in range(max_width, 0, -1):
                needle = context_tokens[-width:]
                found = False
                for pos in range(0, len(seq) - width):
                    if seq[pos:pos + width] == needle:
                        out[seq[pos + width]] += math.log1p(score) * (1.0 + width)
                        found = True
                if found:
                    break
        if out or len(event_scores) < 2:
            return out
        context_bigrams = set(zip(context_tokens, context_tokens[1:]))
        for eid, score in event_scores.most_common(limit):
            rec = self.events.get(eid)
            if not rec or not rec.sequence:
                continue
            seq = [symbol.split(":", 1)[1] for symbol in rec.sequence if symbol.startswith("text:")]
            prefix = seq[: min(4, len(seq))]
            if any(pair in context_bigrams for pair in zip(prefix, prefix[1:])):
                continue
            for pos, token in enumerate(seq[: min(4, len(seq))]):
                if token and token not in recent:
                    out[token] += math.log1p(float(score)) * (0.35 / (1.0 + pos))
        return out

    def energy_beam_decode_text(self, prompt: str, tokens: List[str], max_new_tokens: int = 24, beam_size: int = 3) -> str:
        query_features = self.features_for_query(prompt)
        generation_features = self.generation_query_features(query_features)
        event_scores = self.event_scores_for_features(generation_features)
        semantic_energy = self.semantic_token_energy(query_features, event_scores)
        beams: List[Tuple[List[str], float]] = [(list(tokens[-min(8, len(tokens)):]), 0.0)]
        width = max(1, min(8, int(beam_size)))
        for _ in range(max(1, int(max_new_tokens))):
            next_beams: List[Tuple[List[str], float]] = []
            for seq, score in beams:
                scores = self.next_token_energy_scores(seq, semantic_energy, event_scores)
                if not scores:
                    next_beams.append((seq, score))
                    continue
                for token, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:width]:
                    next_beams.append((seq + [token], score + math.log1p(value)))
            if not next_beams:
                break
            next_beams.sort(key=lambda item: (item[1] / max(1, len(item[0])), item[1]), reverse=True)
            beams = next_beams[:width]
        best = beams[0][0] if beams else []
        if len(best) <= len(tokens[-min(8, len(tokens)):]):
            return ""
        return " ".join(best)

    def graph_decode_text(
        self,
        prompt: str,
        tokens: List[str],
        max_new_tokens: int = 24,
        temperature: float = 0.0,
        beam_size: int = 1,
    ) -> str:
        if not tokens:
            return ""
        if beam_size > 1:
            return self.beam_decode_text(prompt, tokens, max_new_tokens=max_new_tokens, beam_size=beam_size)
        query_features = self.features_for_query(prompt)
        expanded = self.expanded_query_features(query_features, limit=64)
        energy = self.activation(expanded, hops=2)
        out = list(tokens[-min(6, len(tokens)):])
        used = Counter(out)
        for _ in range(max(1, int(max_new_tokens))):
            scores, matched_order = self.next_token_scores(out, energy)
            if not scores:
                break
            if matched_order == 0:
                break
            for token, count in list(used.items()):
                if token in scores:
                    scores[token] *= 1.0 / ((1.0 + count) ** 2)
            next_token = self.select_next_token(scores, out, temperature=temperature)
            score = scores.get(next_token, 0.0)
            if score <= 0.0 or (len(out) >= 3 and next_token == out[-1] == out[-2] == out[-3]):
                break
            out.append(next_token)
            used[next_token] += 1
        if len(out) <= len(tokens[-min(6, len(tokens)):]):
            return ""
        return " ".join(out)

    def next_token_scores(self, context_tokens: List[str], energy: Dict[str, float]) -> Tuple[Counter[str], int]:
        max_order = min(self.dynamic_generation_order(), len(context_tokens))
        for order in range(max_order, 0, -1):
            context = tuple(context_tokens[-order:])
            options = self.order_contexts.get(context)
            if not options:
                continue
            scores: Counter[str] = Counter()
            for token, count in options.most_common(64):
                scores[token] += float(count) * (1.0 + order)
            for feature, value in energy.items():
                token = self.token_from_feature(feature)
                if token and token in scores:
                    scores[token] += float(value) * 0.35
            return scores, order
        return Counter(), 0

    def select_next_token(self, scores: Counter[str], context_tokens: List[str], temperature: float = 0.0) -> str:
        recent = set(context_tokens[-4:])
        fresh_scores = Counter({token: score for token, score in scores.items() if token not in recent})
        if fresh_scores:
            scores = fresh_scores
        if temperature <= 0.0:
            return max(scores.items(), key=lambda item: (item[1], item[0]))[0]
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:16]
        if not ranked:
            return ""
        peak = ranked[0][1]
        temp = max(1e-6, float(temperature))
        weights = [math.exp((score - peak) / temp) for _, score in ranked]
        total = sum(weights) + 1e-12
        seed_text = "|".join(context_tokens[-8:]) + "|" + ",".join(token for token, _ in ranked)
        draw = int.from_bytes(hashlib.blake2b(seed_text.encode("utf-8", errors="ignore"), digest_size=8).digest(), "little") / float(2 ** 64)
        acc = 0.0
        for (token, _), weight in zip(ranked, weights):
            acc += weight / total
            if draw <= acc:
                return token
        return ranked[-1][0]

    def beam_decode_text(self, prompt: str, tokens: List[str], max_new_tokens: int = 24, beam_size: int = 3) -> str:
        query_features = self.features_for_query(prompt)
        expanded = self.expanded_query_features(query_features, limit=64)
        energy = self.activation(expanded, hops=2)
        beams: List[Tuple[List[str], float]] = [(list(tokens[-min(6, len(tokens)):]), 0.0)]
        width = max(1, min(8, int(beam_size)))
        for _ in range(max(1, int(max_new_tokens))):
            next_beams: List[Tuple[List[str], float]] = []
            for seq, score in beams:
                scores, matched_order = self.next_token_scores(seq, energy)
                if not scores or matched_order == 0:
                    next_beams.append((seq, score))
                    continue
                recent = set(seq[-4:])
                scored = [(token, value) for token, value in scores.items() if token not in recent]
                if not scored:
                    scored = list(scores.items())
                for token, value in sorted(scored, key=lambda item: item[1], reverse=True)[:width]:
                    if len(seq) >= 2 and any(seq[i] == seq[-1] and seq[i + 1] == token for i in range(0, len(seq) - 1)):
                        continue
                    next_beams.append((seq + [token], score + math.log1p(max(0.0, value))))
            if not next_beams:
                break
            next_beams.sort(key=lambda item: (item[1] / max(1, len(item[0])), item[1]), reverse=True)
            beams = next_beams[:width]
        best = beams[0][0] if beams else []
        if len(best) <= len(tokens[-min(6, len(tokens)):]):
            return ""
        return " ".join(best)

    def token_from_feature(self, feature: str) -> str:
        canonical = canonical_feature(feature)
        if canonical.startswith("tok:"):
            token = canonical.split(":", 1)[1]
            if len(token) > 1 and not token.isdigit():
                return token
        return ""

    def recall_fragments(self, prompt: str, recalls: List[Dict[str, Any]]) -> List[str]:
        query_features = set(content_features(self.features_for_query(prompt)))
        scored: List[Tuple[float, str]] = []
        for row in recalls:
            rec = self.events.get(row["event_id"])
            if rec is None:
                continue
            rec_features = self.event_content_sets.get(rec.event_id)
            if rec_features is None:
                rec_features = set(content_features(rec.features))
                self.event_content_sets[rec.event_id] = rec_features
            overlap = self.weighted_feature_overlap(query_features, rec_features)
            for fragment in self.preview_fragments(rec.preview):
                scored.append((float(row.get("score", 0.0)) + overlap * 10.0, fragment))
        scored.sort(key=lambda item: item[0], reverse=True)
        return list(dict.fromkeys(fragment for _, fragment in scored if fragment))

    def preview_fragments(self, text: str) -> List[str]:
        cleaned = " ".join(str(text).replace("\ufeff", " ").split())
        if not cleaned:
            return []
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if part.strip()]
        if parts:
            return parts[:4]
        tokens = cleaned.split()
        if len(tokens) <= 24:
            return [cleaned]
        return [" ".join(tokens[i:i + 24]) for i in range(0, min(len(tokens), 72), 18)]

    def event_chain_summary(self, prompt: str, recalls: List[Dict[str, Any]], limit: int = 24) -> str:
        query_features = self.features_for_query(prompt)
        expanded = self.expanded_query_features(query_features, limit=max(16, min(64, limit * 2)))
        energy = self.activation(expanded, hops=2)
        terms = []
        for feature, _ in sorted(energy.items(), key=lambda item: item[1], reverse=True):
            for term in self.feature_terms(feature):
                if term not in terms:
                    terms.append(term)
            if len(terms) >= limit:
                break
        recalled_terms = []
        for row in recalls[:4]:
            rec = self.events.get(row["event_id"])
            if rec is None:
                continue
            for symbol in rec.sequence:
                if not symbol.startswith("text:"):
                    continue
                term = symbol.split(":", 1)[1]
                if term not in recalled_terms:
                    recalled_terms.append(term)
                if len(recalled_terms) >= limit:
                    break
        merged = list(dict.fromkeys([*tokenize(prompt, max_tokens=64), *terms, *recalled_terms]))
        if not merged:
            return ""
        return " ".join(merged[:limit])

    def feature_terms(self, feature: str) -> List[str]:
        canonical = canonical_feature(feature)
        if canonical.startswith("tok:") or canonical.startswith("stem:"):
            return [canonical.split(":", 1)[1]]
        if canonical.startswith("bi:") or canonical.startswith("tri:"):
            return [part for part in canonical.split(":", 1)[1].split("_") if part]
        return []

    def trim_answer(self, text: str, max_new_tokens: int) -> str:
        words = str(text).split()
        limit = max(1, int(max_new_tokens)) + 16
        return " ".join(words[:limit])

    def answer_by_analogy(self, tokens: List[str], max_new_tokens: int) -> str:
        if not tokens:
            return ""
        recalls = self.recall(" ".join(tokens), limit=max(2, int(math.sqrt(max(1, len(self.events))))))
        best_tail: List[str] = []
        suffix_limit = min(len(tokens), self.dynamic_sequence_order())
        for row in recalls:
            rec = self.events.get(row["event_id"])
            if not rec or not rec.sequence:
                continue
            seq = [symbol.split(":", 1)[-1] for symbol in rec.sequence if symbol.startswith("text:")]
            if not seq:
                continue
            for width in range(suffix_limit, 0, -1):
                needle = tokens[-width:]
                for pos in range(0, len(seq) - width + 1):
                    if seq[pos:pos + width] == needle:
                        tail = seq[pos + width:pos + width + max_new_tokens]
                        if len(tail) > len(best_tail):
                            best_tail = tail
                        break
                if best_tail:
                    break
        if not best_tail:
            return ""
        return " ".join(tokens + best_tail)

    def prune(self, min_edge: float = 0.02, max_edges_per_feature: int = 64) -> int:
        removed = 0
        per_feature: Dict[int, List[Tuple[Tuple[int, int], float]]] = defaultdict(list)
        for key, weight in self.edges.items():
            if weight >= min_edge:
                left, _ = key
                per_feature[left].append((key, weight))
        keep_keys = set()
        for rows in per_feature.values():
            rows.sort(key=lambda item: item[1], reverse=True)
            keep_keys.update(key for key, _ in rows[:max_edges_per_feature])
        old_count = len(self.edges)
        self.edges = {key: weight for key, weight in self.edges.items() if key in keep_keys and weight >= min_edge}
        self.edge_last_seen = {key: seen for key, seen in self.edge_last_seen.items() if key in self.edges}
        removed += old_count - len(self.edges)
        self._neighbors_dirty = True
        return removed

    def stats(self) -> Dict[str, Any]:
        edge_count = len(self.edges)
        return {
            "events": len(self.events),
            "features": len(self.feature_freq),
            "edges": edge_count,
            "concepts": len(self.concept_members),
            "short_term_events": len(self.short_term_events),
            "modalities": dict(self.modality_freq),
            "labels": dict(self.label_freq),
            "total_exposures": self.total_exposures,
            "avg_novelty": sum(self.event_novelty.values()) / max(1, len(self.event_novelty)),
            "hdc_bits": self.hdc_bits,
            "hdc_words": self.hdc_words,
            "hdc_indexed_bands": self.hdc_indexed_bands,
            "activation_hops": self.dynamic_hops(),
            "recall_top_k": self.dynamic_recall_top_k(),
            "rank_event_cap": self.dynamic_rank_event_cap(),
            "memory_bytes_estimate": self.memory_bytes_estimate(),
        }

    def memory_bytes_estimate(self) -> int:
        edge_count = len(self.edges)
        hdc_bytes = sum(vec.nbytes for vec in self.event_hv.values() if isinstance(vec, np.ndarray))
        return int(len(self.events) * 384 + len(self.feature_freq) * 80 + edge_count * 16 + hdc_bytes)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = self.__getstate__()
        with lzma.open(p, "wb", preset=0) as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_rank_feature_cache"] = {}
        state["_rank_content_cache"] = {}
        state["_rank_state_cache"] = {}
        state["_rank_choice_cache"] = {}
        state["_rank_result_cache"] = {}
        state["_feature_weight_cache"] = {}
        state["hdc_index"] = defaultdict(set)
        state["event_hv"] = {}
        state["event_content_sets"] = {}
        state["feature_top_events"] = defaultdict(Counter)
        return state

    def save_uncompressed(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _compact_micro_rank_bytes(
        self,
        max_events: int,
        max_features: int,
        event_feature_cap: Optional[int],
        postings_per_feature: Optional[int],
        hash_bits: int = 16,
    ) -> bytes:
        max_events = max(1, int(max_events))
        max_features = max(1, int(max_features))
        feature_cap = None if event_feature_cap is None else max(1, int(event_feature_cap))
        posting_cap = None if postings_per_feature is None else max(1, int(postings_per_feature))
        selected_hash_bits = 32 if int(hash_bits) == 32 else 16
        event_ids = {
            eid: idx
            for idx, (eid, _) in enumerate(
                sorted(self.event_novelty.items(), key=lambda item: item[1], reverse=True)[:max_events]
            )
        }
        merged_rows: Dict[int, Counter[int]] = defaultdict(Counter)
        for fid, rows in self.feature_top_events.items():
            feature = self.id_to_feature[fid] if fid < len(self.id_to_feature) else ""
            h = micro_feature_hash(feature, bits=selected_hash_bits)
            if h is None:
                continue
            for eid, count in rows.items():
                compact_eid = event_ids.get(eid)
                if compact_eid is not None:
                    merged_rows[h][compact_eid] += int(count)
        ranked_features = sorted(
            merged_rows.items(),
            key=lambda item: (sum(item[1].values()), len(item[1])),
            reverse=True,
        )[:max_features]
        selected_feature_hashes = {h for h, _ in ranked_features}
        top_rows: Dict[int, List[Tuple[int, int]]] = {}
        for h, counts in ranked_features:
            rows = counts.most_common(posting_cap)
            packed = [(eid, count) for eid, count in rows]
            if packed:
                top_rows[h] = packed
        event_rows: Dict[int, List[int]] = {}
        for eid, compact_eid in event_ids.items():
            feats = self.event_content_sets.get(eid)
            if feats is None and eid in self.events:
                feats = set(content_features(self.events[eid].features))
            all_hashes = [h for h in (micro_feature_hash(feature, bits=selected_hash_bits) for feature in (feats or ())) if h is not None]
            preferred = sorted({h for h in all_hashes if h in selected_feature_hashes})
            if feature_cap is None:
                extra = sorted({h for h in all_hashes if h not in selected_feature_hashes})
                preferred.extend(extra)
                hashes = preferred
            elif len(preferred) < feature_cap:
                extra = sorted({h for h in all_hashes if h not in selected_feature_hashes})
                preferred.extend(extra[: max(0, feature_cap - len(preferred))])
                hashes = preferred[:feature_cap]
            else:
                hashes = preferred[:feature_cap]
            if hashes:
                event_rows[compact_eid] = hashes
        out = bytearray()
        out.extend(b"PMR2")
        write_varuint(out, selected_hash_bits)
        write_varuint(out, len(top_rows))
        for h, rows in sorted(top_rows.items()):
            write_varuint(out, h)
            write_varuint(out, len(rows))
            for eid, count in rows:
                write_varuint(out, eid)
                write_varuint(out, count)
        write_varuint(out, len(event_rows))
        for eid, hashes in sorted(event_rows.items()):
            write_varuint(out, eid)
            write_varuint(out, len(hashes))
            for h in hashes:
                write_varuint(out, h)
        return zlib.compress(bytes(out), level=9)

    def compact_micro_rank_bytes(self, max_bytes: Optional[int] = None, hash_bits: int = 16) -> bytes:
        event_count = max(1, len(self.events))
        feature_count = max(1, len(self.feature_freq))
        if max_bytes is None:
            return self._compact_micro_rank_bytes(
                max_events=event_count,
                max_features=feature_count,
                event_feature_cap=None,
                postings_per_feature=None,
                hash_bits=hash_bits,
            )
        attempts = []
        max_events = min(4096, event_count)
        max_features = min(8192, feature_count)
        event_feature_cap = 32
        postings_per_feature = 4
        for _ in range(12):
            attempts.append((max_events, max_features, event_feature_cap, postings_per_feature))
            if max_events <= 32 and max_features <= 128 and event_feature_cap <= 8 and postings_per_feature <= 1:
                break
            max_events = max(32, max_events // 2)
            max_features = max(128, max_features // 2)
            event_feature_cap = max(8, event_feature_cap // 2)
            postings_per_feature = max(1, postings_per_feature // 2)
        best = b""
        for config in attempts:
            payload = self._compact_micro_rank_bytes(*config, hash_bits=hash_bits)
            best = payload
            if max_bytes is None or len(payload) <= max_bytes:
                return payload
        return best

    def save_rank_micro_only(self, path: str | Path, max_bytes: Optional[int] = None, hash_bits: int = 16) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.compact_micro_rank_bytes(max_bytes=max_bytes, hash_bits=hash_bits))

    @classmethod
    def load(cls, path: str | Path) -> "BrainMemory":
        p = Path(path)
        with p.open("rb") as raw:
            magic = raw.read(6)
        if magic.startswith(b"\xfd7zXZ"):
            opener = lzma.open
        elif magic.startswith(b"\x1f\x8b"):
            opener = gzip.open
        else:
            opener = open
        with opener(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            mem = cls(window_size=obj.get("window_size", 12), decay=obj.get("decay", 0.72))
            mem.__dict__.update(obj)
            obj = mem
        elif not isinstance(obj, cls):
            raise TypeError("brain memory file has incompatible type")
        obj.ensure_runtime_fields()
        return obj

    def ensure_runtime_fields(self) -> None:
        if not hasattr(self, "hdc_words"):
            self.hdc_words = dynamic_hdc_words(len(getattr(self, "feature_to_id", {})), len(getattr(self, "events", {})))
        if not hasattr(self, "hdc_bits"):
            self.hdc_bits = self.hdc_words * HDC_WORD_BITS
        if not hasattr(self, "hdc_indexed_bands"):
            self.hdc_indexed_bands = hdc_band_count(self.hdc_words, len(getattr(self, "events", {})))
        if not hasattr(self, "hdc_index"):
            self.hdc_index = defaultdict(set)
        if not hasattr(self, "event_hv"):
            self.event_hv = {}
        if not hasattr(self, "feature_top_events"):
            self.feature_top_events = defaultdict(Counter)
        if not hasattr(self, "total_feature_count"):
            self.total_feature_count = sum(getattr(self, "feature_freq", Counter()).values())
        if not hasattr(self, "event_content_sets"):
            self.event_content_sets = {}
        if not hasattr(self, "_rank_feature_cache"):
            self._rank_feature_cache = {}
        if not hasattr(self, "_rank_content_cache"):
            self._rank_content_cache = {}
        if not hasattr(self, "_rank_state_cache"):
            self._rank_state_cache = {}
        if not hasattr(self, "_rank_choice_cache"):
            self._rank_choice_cache = {}
        if not hasattr(self, "_rank_result_cache"):
            self._rank_result_cache = {}
        if not hasattr(self, "_feature_weight_cache"):
            self._feature_weight_cache = {}
        if not hasattr(self, "token_successors"):
            self.token_successors = defaultdict(Counter)
        if not hasattr(self, "order_contexts"):
            self.order_contexts = defaultdict(Counter)
        if not hasattr(self, "token_unigrams"):
            self.token_unigrams = Counter()
        if not hasattr(self, "sequence_starts"):
            self.sequence_starts = Counter()
        if not self.order_contexts:
            for rec in getattr(self, "events", {}).values():
                self.learn_order_trace(rec.sequence)
        if not self.token_unigrams or not self.sequence_starts:
            self.token_unigrams = Counter()
            self.sequence_starts = Counter()
            for rec in getattr(self, "events", {}).values():
                tokens = [symbol.split(":", 1)[1] for symbol in rec.sequence if symbol.startswith("text:")]
                if tokens:
                    self.sequence_starts[tokens[0]] += 1
                    self.token_unigrams.update(tokens)
        if not hasattr(self, "rank_mode"):
            self.rank_mode = "free_energy"
        if not hasattr(self, "cluster_freq"):
            self.cluster_freq = Counter()
        if not hasattr(self, "concept_members"):
            self.concept_members = {}
        if not hasattr(self, "short_term_events"):
            self.short_term_events = list(getattr(self, "events", {}).keys())[-self.dynamic_short_term_limit():]
        if not hasattr(self, "_neighbors_dirty"):
            self._neighbors_dirty = True
        for eid, rec in self.events.items():
            if not isinstance(getattr(rec, "hv", None), np.ndarray) or rec.hv.size != self.hdc_words:
                rec.hv = self.bundle_event(rec.features, rec.sequence)
            if not hasattr(rec, "created_at"):
                rec.created_at = self.total_exposures
            if not hasattr(rec, "last_accessed_at"):
                rec.last_accessed_at = rec.created_at
            self.event_hv[eid] = rec.hv
            self.event_content_sets[eid] = set(content_features(rec.features))
        self.hdc_index = defaultdict(set)
        self.hdc_indexed_bands = hdc_band_count(self.hdc_words, len(self.events))
        for eid, vec in self.event_hv.items():
            self.index_event_hv(eid, vec)
        if not self.feature_top_events:
            cap = self.dynamic_rank_event_cap()
            self.feature_top_events = defaultdict(Counter)
            for fid, rows in self.feature_to_events.items():
                self.feature_top_events[fid] = Counter(dict(Counter(rows).most_common(cap)))


def aspect_bucket(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    ratio = width / height
    if ratio < 0.8:
        return "portrait"
    if ratio > 1.25:
        return "landscape"
    return "square"


def wav_signal_features(wf: wave.Wave_read) -> List[str]:
    width = wf.getsampwidth()
    channels = max(1, wf.getnchannels())
    frames = wf.getnframes()
    if width != 2 or frames <= 0:
        return []
    pos = wf.tell()
    wf.rewind()
    raw = wf.readframes(min(frames, 4096))
    wf.setpos(pos)
    count = len(raw) // 2
    if count < 2:
        return []
    samples = struct.unpack("<" + "h" * count, raw[: count * 2])
    mono = [sum(samples[i:i + channels]) / channels for i in range(0, len(samples), channels)]
    if len(mono) < 2:
        return []
    crossings = sum(1 for a, b in zip(mono, mono[1:]) if (a < 0 <= b) or (a >= 0 > b))
    zcr = crossings / max(1, len(mono) - 1)
    rms = math.sqrt(sum(s * s for s in mono) / len(mono)) / 32768.0
    rough = sum(abs(b - a) for a, b in zip(mono, mono[1:])) / max(1, len(mono) - 1) / 32768.0
    return [
        f"audio:zcr:{int(zcr * 32)}",
        f"audio:zcr64:{int(zcr * 64)}",
        f"audio:zcr128:{int(zcr * 128)}",
        f"audio:rms:{int(rms * 32)}",
        f"audio:roughness:{int(rough * 32)}",
        f"audio:roughness64:{int(rough * 64)}",
    ]


def read_text_file_as_one_event(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".csv":
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            for row in csv.reader(f):
                rows.append(" ".join(str(cell).strip() for cell in row if str(cell).strip()))
        return "\n".join(row for row in rows if row)
    if suffix == ".jsonl":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        return json.dumps(json.loads(path.read_text(encoding="utf-8", errors="ignore")), ensure_ascii=False)
    return path.read_text(encoding="utf-8", errors="ignore")


def text_subword_features(token: str) -> List[str]:
    out = []
    if len(token) >= 4:
        out.append(f"text:stem:{rough_stem(token)}")
        padded = f"_{token}_"
        for i in range(len(padded) - 2):
            out.append(f"text:c3:{padded[i:i + 3]}")
    return out[:12]


def rough_stem(token: str) -> str:
    for suffix in ("ing", "ed", "ly", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def content_features(features: List[str]) -> List[str]:
    filtered = [canonical_feature(feature) for feature in features if not str(feature).startswith("mod:")]
    return filtered if filtered else list(features)


def canonical_feature(feature: str) -> str:
    text = str(feature)
    for prefix, target in (
        ("text:tok:", "tok:"),
        ("text:id:", "id:"),
        ("text:stem:", "stem:"),
        ("text:bi:", "bi:"),
        ("text:tri:", "tri:"),
        ("text:c3:", "c3:"),
    ):
        if text.startswith(prefix):
            return target + text.split(prefix, 1)[1]
    return text
