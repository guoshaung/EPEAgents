"""Educational personalization environment for privacy-aware multi-agent experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, List


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class StudentState:
    """Represents the simulated knowledge state for a single learner."""

    skills: Dict[str, float]
    fatigue: float = field(default_factory=lambda: random.uniform(0.0, 0.2))
    curiosity: float = field(default_factory=lambda: random.uniform(0.2, 0.8))

    def copy(self) -> "StudentState":
        return StudentState(skills=self.skills.copy(), fatigue=self.fatigue, curiosity=self.curiosity)


class PersonalizedLearningEnv:
    """Toy environment for simulating multi-agent personalization strategies."""

    def __init__(
        self,
        topics: List[str],
        base_learning_rate: float = 0.15,
        fatigue_penalty: float = 0.03,
        stochasticity: float = 0.02,
    ) -> None:
        self.topics = topics
        self.base_learning_rate = base_learning_rate
        self.fatigue_penalty = fatigue_penalty
        self.stochasticity = stochasticity
        self._states: Dict[str, StudentState] = {}

    def reset_agent(self, agent_id: str) -> StudentState:
        state = StudentState(skills={topic: random.uniform(0.2, 0.6) for topic in self.topics})
        self._states[agent_id] = state
        return state.copy()

    def get_state(self, agent_id: str) -> StudentState:
        if agent_id not in self._states:
            return self.reset_agent(agent_id)
        return self._states[agent_id].copy()

    def describe(self) -> str:
        return (
            "Topics: "
            + ", ".join(self.topics)
            + "; higher fatigue slows learning; curiosity multiplies learning gains."
        )

    def step(self, agent_id: str, topic: str) -> StudentState:
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        state = self._states.setdefault(agent_id, self.reset_agent(agent_id))
        difficulty = 1.0 - state.skills[topic]
        curiosity_bonus = 1.0 + 0.5 * (state.curiosity - 0.5)
        delta = self.base_learning_rate * difficulty * curiosity_bonus
        delta -= self.fatigue_penalty * state.fatigue
        delta += random.gauss(0, self.stochasticity)
        state.skills[topic] = _bounded(state.skills[topic] + delta)
        state.fatigue = _bounded(state.fatigue + 0.05 + random.uniform(-0.01, 0.01))
        state.curiosity = _bounded(state.curiosity + random.uniform(-0.03, 0.03))
        return state.copy()

    def reward(self, previous: StudentState, current: StudentState, topic: str) -> float:
        knowledge_gain = current.skills[topic] - previous.skills[topic]
        balance = 1.0 - abs(sum(current.skills.values()) / len(self.topics) - 0.5)
        fatigue_cost = current.fatigue - previous.fatigue
        return knowledge_gain + 0.1 * balance - 0.05 * fatigue_cost
