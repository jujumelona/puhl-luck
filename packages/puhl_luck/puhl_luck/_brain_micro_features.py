from __future__ import annotations

from ._brain_defs import *
from ._brain_byte_features import *
from ._brain_hashing import *
from ._brain_media_features import *
from ._brain_text_features import *

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
