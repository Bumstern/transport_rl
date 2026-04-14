import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

from sb3_contrib import MaskablePPO

from src.optimizer.main import SimulatorEnv
from src.optimizer.settings import GENERATOR_SETTINGS
from src.simulator.utils.data_generator.generator import InputDataGenerator


@dataclass(frozen=True)
class EvalConfig:
    model_path: Path
    episodes: int
    deterministic: bool
    seed: int | None
    output_path: Path | None


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    requests_num: int
    reward: float
    missed_requests_num: int
    unfinished_ratio: float


def build_generator() -> InputDataGenerator:
    return InputDataGenerator(
        load_point_names=GENERATOR_SETTINGS.load_point_names,
        unload_point_names=GENERATOR_SETTINGS.unload_point_names,
        requests_num_min=GENERATOR_SETTINGS.min_requests_num,
        requests_num_max=GENERATOR_SETTINGS.max_requests_num,
        trucks_num=GENERATOR_SETTINGS.max_truck_num,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, "%d.%m.%Y"),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, "%d.%m.%Y"),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        min_distance=GENERATOR_SETTINGS.min_distance,
        max_distance=GENERATOR_SETTINGS.max_distance,
    )


def build_env() -> SimulatorEnv:
    return SimulatorEnv(build_generator())


def evaluate(config: EvalConfig) -> tuple[list[EpisodeMetrics], dict]:
    env = build_env()
    model = MaskablePPO.load(str(config.model_path))

    episode_metrics: list[EpisodeMetrics] = []
    for episode_id in range(config.episodes):
        episode_seed = None if config.seed is None else config.seed + episode_id
        observation, _ = env.reset(seed=episode_seed)
        terminated = False
        cumulative_reward = 0.0
        last_info = None

        while not terminated:
            action_mask = env.action_masks()
            action, _ = model.predict(
                observation,
                action_masks=action_mask,
                deterministic=config.deterministic,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            last_info = info

            if truncated:
                raise RuntimeError("Evaluation episode was truncated unexpectedly.")

        assert last_info is not None, "Evaluation finished without episode info."
        episode_metrics.append(
            EpisodeMetrics(
                episode=episode_id,
                requests_num=env._current_env.requests_num,
                reward=float(cumulative_reward),
                missed_requests_num=int(last_info["missed_requests_num"]),
                unfinished_ratio=float(last_info["unfinished_ratio"][0]),
            )
        )

    summary = {
        "model_path": str(config.model_path),
        "episodes": config.episodes,
        "deterministic": config.deterministic,
        "seed": config.seed,
        "avg_reward": mean(metric.reward for metric in episode_metrics),
        "avg_missed_requests_num": mean(metric.missed_requests_num for metric in episode_metrics),
        "avg_unfinished_ratio": mean(metric.unfinished_ratio for metric in episode_metrics),
        "min_missed_requests_num": min(metric.missed_requests_num for metric in episode_metrics),
        "max_missed_requests_num": max(metric.missed_requests_num for metric in episode_metrics),
        "fully_served_rate": (
            sum(metric.missed_requests_num == 0 for metric in episode_metrics) / len(episode_metrics)
        ),
    }
    return episode_metrics, summary


def save_report(
    output_path: Path,
    config: EvalConfig,
    episode_metrics: list[EpisodeMetrics],
    summary: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "model_path": str(config.model_path),
            "episodes": config.episodes,
            "deterministic": config.deterministic,
            "seed": config.seed,
        },
        "summary": summary,
        "episodes": [asdict(metric) for metric in episode_metrics],
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))


def print_report(episode_metrics: list[EpisodeMetrics], summary: dict) -> None:
    print("Evaluation summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nPer-episode metrics:")
    for metric in episode_metrics:
        print(asdict(metric))


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate MaskablePPO on SimulatorEnv.")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the trained model zip file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for evaluation episodes.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of deterministic predictions.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to save the evaluation report as JSON.",
    )
    args = parser.parse_args()

    return EvalConfig(
        model_path=args.model_path,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        seed=args.seed,
        output_path=args.output_path,
    )


def main() -> None:
    config = parse_args()
    episode_metrics, summary = evaluate(config)
    print_report(episode_metrics, summary)
    if config.output_path is not None:
        save_report(config.output_path, config, episode_metrics, summary)
        print(f"\nSaved report to: {config.output_path}")


if __name__ == "__main__":
    main()
