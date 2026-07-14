"""Bit-per-weight accounting so quant methods are compared at equal footprint."""
from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class Footprint:
    total_params: int
    file_bytes: int

    @property
    def bpw(self) -> float:
        return self.file_bytes * 8 / self.total_params

    @property
    def gib(self) -> float:
        return self.file_bytes / BYTES_PER_GIB


def target_bytes(total_params: int, bpw: float) -> int:
    return round(total_params * bpw / 8)


def match_tolerance(a: Footprint, b: Footprint, rel_tol: float = 0.03) -> bool:
    """True when two builds are within rel_tol of each other by file size."""
    larger = max(a.file_bytes, b.file_bytes)
    return abs(a.file_bytes - b.file_bytes) / larger <= rel_tol
