from __future__ import annotations

from ._memory_activation_core import MemoryActivationCoreMixin
from ._memory_hopfield import MemoryHopfieldMixin


class MemoryActivationMixin(
    MemoryActivationCoreMixin,
    MemoryHopfieldMixin,
):
    pass
