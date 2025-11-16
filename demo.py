"""Entry point showcasing federated multi-agent personalization with privacy hooks."""
from __future__ import annotations

from epeagents import TrainingPipeline


def main() -> None:
    topics = ["algebra", "calculus", "geometry", "statistics"]
    agent_factory = [("agent_a", "base"), ("agent_b", "meta"), ("agent_c", "base")]
    pipeline = TrainingPipeline(topics, agent_factory, rounds=3, steps_per_round=8)
    summaries = pipeline.run()
    for summary in summaries:
        print(f"Round {summary.round_index}: avg_reward={summary.avg_reward:.3f}")
        for note in summary.curriculum_notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
