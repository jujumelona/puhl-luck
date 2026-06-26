from __future__ import annotations

from ._memory_micro_export import MemoryMicroExportMixin
from ._memory_save import MemorySaveMixin
from ._memory_stats import MemoryStatsMixin


class MemoryPersistenceMixin(
    MemoryStatsMixin,
    MemorySaveMixin,
    MemoryMicroExportMixin,
):
    pass
