"""Federated averaging server with privacy hooks."""
from __future__ import annotations

from typing import Dict, Iterable

from epeagents.privacy import dp


class FederatedServer:
    def __init__(self, clip_value: float = 1.5, noise_sigma: float = 0.01) -> None:
        self.clip_value = clip_value
        self.noise_sigma = noise_sigma

    def aggregate(self, client_updates: Iterable[Dict[str, float]]) -> Dict[str, float]:
        updates = list(client_updates)
        if not updates:
            return {}
        clipped = [dp.l2_clip(update, self.clip_value) for update in updates]
        aggregated: Dict[str, float] = {}
        for update in clipped:
            for topic, value in update.items():
                aggregated.setdefault(topic, 0.0)
                aggregated[topic] += value
        for topic in aggregated:
            aggregated[topic] /= len(clipped)
        return dp.add_gaussian_noise(aggregated, self.noise_sigma)
