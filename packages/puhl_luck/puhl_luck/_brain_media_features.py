from __future__ import annotations

from ._brain_defs import *
from ._brain_hashing import *
from ._brain_text_features import *

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
