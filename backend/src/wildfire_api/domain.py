from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .repository import WildfireMetadata


@dataclass
class SpreadPrediction:
    metadata: WildfireMetadata
    sample_index: int
    total_samples: int
    threshold: float
    probabilities: np.ndarray
    mask: np.ndarray
    ground_truth: np.ndarray
    observation_dates: Tuple[str, ...]
    target_date: Optional[str]
