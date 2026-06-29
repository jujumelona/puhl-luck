from __future__ import annotations

from ._memory_activation import MemoryActivationMixin
from ._memory_management import MemoryManagementMixin
from ._memory_query import MemoryQueryMixin
from ._memory_recall import MemoryRecallMixin
from ._memory_scoring import MemoryScoringMixin


class MemoryRankingMixin(
    MemoryQueryMixin,
    MemoryActivationMixin,
    MemoryScoringMixin,
    MemoryManagementMixin,
    MemoryRecallMixin,
):
    pass
