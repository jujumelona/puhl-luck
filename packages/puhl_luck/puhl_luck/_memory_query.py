from __future__ import annotations

from ._brain_common import *


class MemoryQueryMixin:
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

