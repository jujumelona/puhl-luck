from __future__ import annotations

from ._memory_answering import MemoryAnsweringMixin
from ._memory_energy_decode import MemoryEnergyDecodeMixin
from ._memory_fragments import MemoryFragmentsMixin
from ._memory_order_decode import MemoryOrderDecodeMixin


class MemoryGenerationMixin(
    MemoryAnsweringMixin,
    MemoryEnergyDecodeMixin,
    MemoryOrderDecodeMixin,
    MemoryFragmentsMixin,
):
    pass
