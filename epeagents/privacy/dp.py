"""Differential privacy utilities for federated aggregation."""
from __future__ import annotations

import math
import random
from typing import Dict


def l2_clip(updates: Dict[str, float], clip_value: float) -> Dict[str, float]:
    norm = math.sqrt(sum(value * value for value in updates.values()))
    if norm <= clip_value or norm == 0:
        return updates.copy()
    scale = clip_value / norm
    return {k: v * scale for k, v in updates.items()}


def add_gaussian_noise(updates: Dict[str, float], sigma: float) -> Dict[str, float]:
    if sigma == 0:
        return updates.copy()
    return {k: v + random.gauss(0, sigma) for k, v in updates.items()}
