from __future__ import annotations

from ._brain_common import *


class MemoryExtractorMixin:
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

