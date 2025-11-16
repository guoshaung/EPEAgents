"""Lightweight agents for federated multi-agent personalization demos."""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from epeagents.environment.learning_env import StudentState


class BaseAgent:
    """Rule-based agent that maintains per-topic policy weights."""

    def __init__(self, agent_id: str, topics: List[str], lr: float = 0.1) -> None:
        self.agent_id = agent_id
        self.topics = topics
        self.learning_rate = lr
        self.policy = {topic: random.uniform(0.5, 1.5) for topic in topics}

    def get_policy(self) -> Dict[str, float]:
        return self.policy.copy()

    def load_policy(self, new_policy: Dict[str, float], mixing: float = 0.5) -> None:
        for topic, value in new_policy.items():
            if topic in self.policy:
                self.policy[topic] = (1 - mixing) * self.policy[topic] + mixing * value

    def _score_topic(self, state: StudentState, topic: str, llm_hint: Optional[Dict[str, float]]) -> float:
        deficit = 1.0 - state.skills[topic]
        fatigue_tax = 1.0 + state.fatigue
        hint_bonus = 1.0
        if llm_hint and topic in llm_hint:
            hint_bonus += llm_hint[topic]
        return self.policy[topic] * deficit * hint_bonus / fatigue_tax

    def act(self, state: StudentState, llm_hint: Optional[Dict[str, float]] = None) -> str:
        scores = {topic: self._score_topic(state, topic, llm_hint) for topic in self.topics}
        return max(scores, key=scores.get)

    def update(self, topic: str, reward: float) -> None:
        self.policy[topic] += self.learning_rate * reward
        self.policy[topic] = max(0.1, min(2.0, self.policy[topic]))


class MetaLearnerAgent(BaseAgent):
    """Extends :class:`BaseAgent` with meta-learning style preferences."""

    def __init__(self, agent_id: str, topics: List[str], lr: float = 0.1, meta_lr: float = 0.05) -> None:
        super().__init__(agent_id, topics, lr)
        self.meta_preferences = {topic: random.uniform(0.8, 1.2) for topic in topics}
        self.meta_lr = meta_lr

    def adapt(self, trajectory_stats: Dict[str, float]) -> None:
        for topic, delta in trajectory_stats.items():
            if topic in self.meta_preferences:
                self.meta_preferences[topic] += self.meta_lr * delta
                self.meta_preferences[topic] = max(0.5, min(1.5, self.meta_preferences[topic]))

    def _score_topic(self, state: StudentState, topic: str, llm_hint: Optional[Dict[str, float]]) -> float:
        base = super()._score_topic(state, topic, llm_hint)
        return base * self.meta_preferences[topic]

    def update(self, topic: str, reward: float) -> None:
        super().update(topic, reward)
        self.adapt({topic: reward})
