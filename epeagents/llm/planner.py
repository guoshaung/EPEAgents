"""Mock LLM planner used to provide task guidance to agents."""
from __future__ import annotations

from typing import Dict

from epeagents.environment.learning_env import StudentState


class LLMPlanner:
    """Heuristic planner mimicking an LLM prompt response."""

    def __init__(self, creativity: float = 0.4) -> None:
        self.creativity = creativity

    def plan(self, state: StudentState, description: str) -> Dict[str, float]:
        hint: Dict[str, float] = {}
        avg_skill = sum(state.skills.values()) / len(state.skills)
        for topic, value in state.skills.items():
            gap = (avg_skill - value) * (1.0 + self.creativity)
            curiosity_effect = (state.curiosity - 0.5) * 0.2
            fatigue_guard = -0.1 if state.fatigue > 0.7 else 0.0
            hint[topic] = gap + curiosity_effect + fatigue_guard
        return hint

    def recommend_curriculum(self, state: StudentState) -> str:
        weakest_topic = min(state.skills, key=state.skills.get)
        strongest_topic = max(state.skills, key=state.skills.get)
        return (
            f"Focus on {weakest_topic} next while interleaving with {strongest_topic} to prevent forgetting. "
            f"Current curiosity {state.curiosity:.2f} and fatigue {state.fatigue:.2f}."
        )
