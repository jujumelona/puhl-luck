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


@dataclass
class EventRecord:
    event_id: str
    modality: str
    source: str
    label: Optional[str]
    features: List[str]
    sequence: List[str]
    preview: str
    novelty: float
    hv: np.ndarray = field(repr=False)
    created_at: int = 0
    last_accessed_at: int = 0
