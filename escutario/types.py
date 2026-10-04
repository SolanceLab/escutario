# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Shared data shapes for Escutário's analysers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PitchTrack:
    """A fundamental-frequency track at a fixed hop.

    f0_hz holds 0.0 for unvoiced frames. confidence is 0..1 per frame.
    times[i] is the centre of frame i in seconds.
    """

    times: np.ndarray
    f0_hz: np.ndarray
    confidence: np.ndarray
    hop_s: float

    def __post_init__(self) -> None:
        n = len(self.times)
        if len(self.f0_hz) != n or len(self.confidence) != n:
            raise ValueError("PitchTrack arrays must share one length")

    @property
    def voiced(self) -> np.ndarray:
        return self.f0_hz > 0
