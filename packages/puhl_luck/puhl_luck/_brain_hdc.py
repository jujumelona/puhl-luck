from __future__ import annotations

from ._brain_defs import *
from ._brain_hashing import *

# Try to import Rust implementations
try:
    import sys
    import os
    # Add package directory to path if needed
    pkg_dir = os.path.dirname(__file__)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    
    import puhl_luck_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

def dynamic_hdc_words(feature_count: int, event_count: int) -> int:
    scale = max(4, feature_count + 2 * event_count + 2)
    return max(2, math.ceil(math.log2(scale)))


def dynamic_hdc_bits(feature_count: int, event_count: int) -> int:
    return dynamic_hdc_words(feature_count, event_count) * HDC_WORD_BITS


def feature_hv(feature: str, words: int) -> np.ndarray:
    # Use Rust if available (9.7x faster)
    if RUST_AVAILABLE:
        return puhl_luck_core.feature_hv_rust(feature, words)
    
    # Fallback to Python
    seed = hashlib.blake2b(feature.encode("utf-8", errors="ignore"), digest_size=16).digest()
    chunks = []
    block_i = 0
    while len(chunks) < words:
        block = hashlib.blake2b(seed + block_i.to_bytes(4, "little"), digest_size=32).digest()
        chunks.extend(struct.unpack("<4Q", block))
        block_i += 1
    return np.array(chunks[:words], dtype=np.uint64)


def rotate_hv(value: np.ndarray, amount: int) -> np.ndarray:
    # Use Rust if available
    if RUST_AVAILABLE:
        return puhl_luck_core.rotate_hv_rust(value, amount)
    
    # Fallback to Python
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
    
    # Use Rust if available (bundling is Rust-only optimization)
    if RUST_AVAILABLE:
        return puhl_luck_core.bundle_hv_rust(feature_list, words)
    
    # Fallback to Python
    value = np.zeros(words, dtype=np.uint64)
    for i, feature in enumerate(feature_list):
        value ^= rotate_hv(feature_hv(feature, words), i)
    return value


def hv_similarity(a: np.ndarray, b: np.ndarray, bits: Optional[int] = None) -> float:
    # Use Rust if available (26.6x faster)
    if RUST_AVAILABLE:
        # Ensure correct dtype for Rust
        if a.dtype != np.uint64:
            a = a.astype(np.uint64)
        if b.dtype != np.uint64:
            b = b.astype(np.uint64)
        return puhl_luck_core.hv_similarity_rust(a, b)
    
    # Fallback to Python
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
