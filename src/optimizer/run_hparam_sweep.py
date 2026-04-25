import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src.optimizer.eval import EvalConfig, evaluate, save_report
from src.optimizer.train import OBSERVATION_PRESETS, TrainConfig, get_observation_feature_config, train


@dataclass(frozen=True)
class SweepVariant:
    name: str
    n_steps: int
    clip_range: float
    learning_rate: float
    learning_rate_schedule: str
    net_arch: list[int]


DEFAULT_VARIANTS = (
    SweepVariant(
        name="baseline_tuned",
        n_steps=512,
        clip_range=0.2,
        learning_rate=1e-4,
        learning_rate_schedule="quarter_decay",
        net_arch=[128, 128, 128],
    ),
    SweepVariant(
        name="bigger_net",
        n_steps=512,
        clip_range=0.2,
        learning_rate=1e-4,
        learning_rate_schedule="quarter_decay",
        net_arch=[256, 256, 256],
    ),
    SweepVariant(
        name="larger_rollout",
        n_steps=1024,
        clip_range=0.2,
        learning_rate=1e-4,
        learning_rate_schedule="quarter_decay",
        net_arch=[128, 128, 128],
    ),
    SweepVariant(
        name="faster_lr",
        n_steps=512,
        clip_range=0.2,
        learning_rate=3e-4,
        learning_rate_schedule="quarter_decay",
        net_arch=[128, 128, 128],
    ),
    SweepVariant(
        name="more_conservative",
        n_steps=512,
        clip_range=0.1,
        learning_rate=1e-4,
        learning_rate_schedule="quarter_decay",
        net_arch=[128, 128, 128],
    ),
)


def build_train_config(
    variant: SweepVariant,
    total_timesteps: int,
    eval_freq: int,
    seed: int | None,
    observation_preset: str,
    base_dir: Path,
    verbose: int,
    early_stop_patience_episodes: int,
) -> TrainConfig:
    variant_dir = base_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    return TrainConfig(
        total_timesteps=total_timesteps,
        n_steps=variant.n_steps,
        clip_range=variant.clip_range,
        learning_rate=variant.learning_rate,
        learning_rate_schedule=variant.learning_rate_schedule,
        learning_rate_step_points=(0.75, 0.5, 0.25),
        learning_rate_step_values=(3e-3, 7e-4, 4e-4, 1e-4),
        net_arch=variant.net_arch,
        eval_freq=eval_freq,
        seed=seed,
        model_dir=variant_dir / "models",
        best_model_dir=variant_dir / "best",
        tensorboard_dir=variant_dir / "tb",
        verbose=verbose,
        observation_feature_config=get_observation_feature_config(observation_preset),
        early_stop_patience_episodes=early_stop_patience_episodes,
    )


def build_eval_config(
    variant: SweepVariant,
    observation_preset: str,
    episodes: int,
    seed: int | None,
    base_dir: Path,
) -> EvalConfig:
    variant_dir = base_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    return EvalConfig(
        policy="model",
        model_path=variant_dir / "best" / "best_model.zip",
        episodes=episodes,
        deterministic=True,
        seed=seed,
        output_path=variant_dir / "eval.json",
        observation_feature_config=get_observation_feature_config(observation_preset),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed PPO hyperparameter sweep.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/hparam_sweep"),
        help="Base directory for sweep artifacts.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=20_000,
        help="Training timesteps per variant. Short by default for screening.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=2048,
        help="Evaluation frequency during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Common seed for all variants.",
    )
    parser.add_argument(
        "--observation-preset",
        choices=OBSERVATION_PRESETS,
        default="all",
        help="Observation preset to keep fixed during the hyperparameter sweep.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Evaluate each best checkpoint after training.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=30,
        help="Number of episodes for post-training evaluation.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Stable-Baselines3 verbosity level.",
    )
    parser.add_argument(
        "--early-stop-patience-episodes",
        "--early-stop-patience-epochs",
        type=int,
        dest="early_stop_patience_episodes",
        default=0,
        help="Stop training if unfinished_ratio does not improve for this many completed episodes. 0 disables early stopping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for variant in DEFAULT_VARIANTS:
        print(f"\n=== Training variant: {variant.name} ===")
        train_config = build_train_config(
            variant=variant,
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            seed=args.seed,
            observation_preset=args.observation_preset,
            base_dir=args.base_dir,
            verbose=args.verbose,
            early_stop_patience_episodes=args.early_stop_patience_episodes,
        )
        last_model_path = train(train_config)
        best_model_path = train_config.best_model_dir / "best_model.zip"

        row = {
            "variant": variant.name,
            "observation_preset": args.observation_preset,
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
            "n_steps": variant.n_steps,
            "clip_range": variant.clip_range,
            "learning_rate": variant.learning_rate,
            "learning_rate_schedule": variant.learning_rate_schedule,
            "net_arch": variant.net_arch,
            "last_model_path": str(last_model_path),
            "best_model_path": str(best_model_path),
        }

        if args.run_eval and best_model_path.exists():
            print(f"=== Evaluating variant: {variant.name} ===")
            eval_config = build_eval_config(
                variant=variant,
                observation_preset=args.observation_preset,
                episodes=args.eval_episodes,
                seed=args.seed,
                base_dir=args.base_dir,
            )
            episode_metrics, eval_summary = evaluate(eval_config)
            save_report(eval_config.output_path, eval_config, episode_metrics, eval_summary)
            row.update(
                {
                    "avg_missed_requests_num": eval_summary["avg_missed_requests_num"],
                    "avg_served_requests_num": eval_summary["avg_served_requests_num"],
                    "avg_skip_rate": eval_summary["avg_skip_rate"],
                    "avg_reward": eval_summary["avg_reward"],
                }
            )

        summary_rows.append(row)

    summary_path = args.base_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"\nSaved sweep summary to: {summary_path}")


if __name__ == "__main__":
    main()
