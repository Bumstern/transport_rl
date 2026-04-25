import argparse
import json
from pathlib import Path

from src.optimizer.eval import EvalConfig, evaluate, save_report
from src.optimizer.train import (
    OBSERVATION_PRESETS,
    TrainConfig,
    get_observation_feature_config,
    train,
)


def build_train_config(
    preset: str,
    total_timesteps: int,
    seed: int | None,
    base_dir: Path,
    n_steps: int,
    clip_range: float,
    learning_rate: float,
    eval_freq: int,
    verbose: int,
    early_stop_patience_episodes: int,
) -> TrainConfig:
    preset_dir = base_dir / preset
    preset_dir.mkdir(parents=True, exist_ok=True)

    return TrainConfig(
        total_timesteps=total_timesteps,
        n_steps=n_steps,
        clip_range=clip_range,
        learning_rate=learning_rate,
        learning_rate_schedule="quarter_decay",
        learning_rate_step_points=(0.75, 0.5, 0.25),
        learning_rate_step_values=(3e-3, 7e-4, 4e-4, 1e-4),
        net_arch=[128, 128, 128],
        eval_freq=eval_freq,
        seed=seed,
        model_dir=preset_dir / "models",
        best_model_dir=preset_dir / "best",
        tensorboard_dir=preset_dir / "tb",
        verbose=verbose,
        observation_feature_config=get_observation_feature_config(preset),
        early_stop_patience_episodes=early_stop_patience_episodes,
    )


def build_eval_config(
    preset: str,
    model_path: Path,
    episodes: int,
    seed: int | None,
    base_dir: Path,
) -> EvalConfig:
    preset_dir = base_dir / preset
    preset_dir.mkdir(parents=True, exist_ok=True)
    return EvalConfig(
        policy="model",
        model_path=model_path,
        episodes=episodes,
        deterministic=True,
        seed=seed,
        output_path=preset_dir / "eval.json",
        observation_feature_config=get_observation_feature_config(preset),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run observation ablation experiments.")
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=OBSERVATION_PRESETS,
        default=list(OBSERVATION_PRESETS),
        help="Observation presets to train.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/ablation"),
        help="Base directory for ablation artifacts.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=20_000,
        help="Training timesteps per preset. Keep this short for screening runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Common seed for all runs.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
        help="Rollout steps per PPO update.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.25,
        help="PPO clip range.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=2048,
        help="Evaluation frequency during training.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Stable-Baselines3 verbosity level.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Evaluate the best checkpoint after each training run.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for post-training evaluation.",
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
    for preset in args.presets:
        print(f"\n=== Training preset: {preset} ===")
        train_config = build_train_config(
            preset=preset,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            base_dir=args.base_dir,
            n_steps=args.n_steps,
            clip_range=args.clip_range,
            learning_rate=args.learning_rate,
            eval_freq=args.eval_freq,
            verbose=args.verbose,
            early_stop_patience_episodes=args.early_stop_patience_episodes,
        )
        last_model_path = train(train_config)
        best_model_path = train_config.best_model_dir / "best_model.zip"

        row = {
            "preset": preset,
            "last_model_path": str(last_model_path),
            "best_model_path": str(best_model_path),
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
        }

        if args.run_eval and best_model_path.exists():
            print(f"=== Evaluating preset: {preset} ===")
            eval_config = build_eval_config(
                preset=preset,
                model_path=best_model_path,
                episodes=args.eval_episodes,
                seed=args.seed,
                base_dir=args.base_dir,
            )
            episode_metrics, eval_summary = evaluate(eval_config)
            save_report(eval_config.output_path, eval_config, episode_metrics, eval_summary)
            row.update({
                "avg_missed_requests_num": eval_summary["avg_missed_requests_num"],
                "avg_served_requests_num": eval_summary["avg_served_requests_num"],
                "avg_skip_rate": eval_summary["avg_skip_rate"],
            })

        summary_rows.append(row)

    summary_path = args.base_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"\nSaved ablation summary to: {summary_path}")


if __name__ == "__main__":
    main()
