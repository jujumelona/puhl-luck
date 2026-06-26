from __future__ import annotations

from ._memory_exposure import MemoryExposureMixin
from ._memory_extractors import MemoryExtractorMixin
from ._memory_graph import MemoryGraphMixin


class MemoryLearningMixin(
    MemoryExposureMixin,
    MemoryGraphMixin,
    MemoryExtractorMixin,
):
    pass
