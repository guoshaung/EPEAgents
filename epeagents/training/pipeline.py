"""High-level orchestration for multi-agent personalized learning demos."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from epeagents.agents.base_agent import BaseAgent, MetaLearnerAgent
from epeagents.environment.learning_env import PersonalizedLearningEnv
from epeagents.federated.server import FederatedServer
from epeagents.llm.planner import LLMPlanner


@dataclass
class RoundSummary:
    round_index: int
    avg_reward: float
    curriculum_notes: List[str]


class TrainingPipeline:
    def __init__(
        self,
        topics: Sequence[str],
        agent_factory: Iterable[Tuple[str, str]],
        rounds: int = 5,
        steps_per_round: int = 10,
    ) -> None:
        self.env = PersonalizedLearningEnv(list(topics))
        self.agents = self._build_agents(agent_factory, topics)
        self.rounds = rounds
        self.steps_per_round = steps_per_round
        self.server = FederatedServer()
        self.planner = LLMPlanner()

    def _build_agents(self, agent_factory: Iterable[Tuple[str, str]], topics: Sequence[str]):
        agents = []
        for agent_id, agent_type in agent_factory:
            if agent_type == "meta":
                agents.append(MetaLearnerAgent(agent_id, list(topics)))
            else:
                agents.append(BaseAgent(agent_id, list(topics)))
        return agents

    def run(self) -> List[RoundSummary]:
        summaries: List[RoundSummary] = []
        for round_index in range(self.rounds):
            rewards: List[float] = []
            curriculum_notes: List[str] = []
            client_policies = []
            for agent in self.agents:
                state = self.env.get_state(agent.agent_id)
                curriculum_notes.append(self.planner.recommend_curriculum(state))
                llm_hint = self.planner.plan(state, self.env.describe())
                for _ in range(self.steps_per_round):
                    previous = state
                    action = agent.act(state, llm_hint)
                    new_state = self.env.step(agent.agent_id, action)
                    reward = self.env.reward(previous, new_state, action)
                    agent.update(action, reward)
                    rewards.append(reward)
                    state = new_state
                client_policies.append(agent.get_policy())
            aggregated = self.server.aggregate(client_policies)
            for agent in self.agents:
                agent.load_policy(aggregated)
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            summaries.append(RoundSummary(round_index, avg_reward, curriculum_notes))
        return summaries
