from __future__ import annotations

from ._brain_defs import *
from ._brain_hashing import *

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
