from __future__ import annotations

from ._brain_defs import *

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
