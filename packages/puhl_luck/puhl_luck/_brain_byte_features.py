from __future__ import annotations

from ._brain_defs import *
from ._brain_hashing import *

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
