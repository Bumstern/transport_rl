import argparse
import json
from pathlib import Path

from src.optimizer.eval import EvalConfig, evaluate, save_report
from src.optimizer.train import OBSERVATION_PRESETS, get_observation_feature_config


def build_eval_config(
    preset: str,
    base_dir: Path,
    episodes: int,
    seed: int | None,
    output_name: str,
) -> EvalConfig:
    preset_dir = base_dir / preset
    model_path = preset_dir / "best" / "best_model.zip"
    output_path = preset_dir / output_name
    return EvalConfig(
        policy="model",
        model_path=model_path,
        episodes=episodes,
        deterministic=True,
        seed=seed,
        output_path=output_path,
        observation_feature_config=get_observation_feature_config(preset),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate trained ablation models and build a summary.")
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=OBSERVATION_PRESETS,
        default=list(OBSERVATION_PRESETS),
        help="Observation presets to re-evaluate.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/ablation"),
        help="Base ablation directory with trained models.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per preset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for evaluation episodes.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="eval_recheck.json",
        help="Per-preset evaluation report filename.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="re_eval_summary.json",
        help="Summary filename in the base ablation directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for preset in args.presets:
        eval_config = build_eval_config(
            preset=preset,
            base_dir=args.base_dir,
            episodes=args.episodes,
            seed=args.seed,
            output_name=args.output_name,
        )
        if not eval_config.model_path.exists():
            summary_rows.append(
                {
                    "preset": preset,
                    "status": "missing_model",
                    "model_path": str(eval_config.model_path),
                }
            )
            continue

        print(f"\n=== Re-evaluating preset: {preset} ===")
        episode_metrics, eval_summary = evaluate(eval_config)
        save_report(eval_config.output_path, eval_config, episode_metrics, eval_summary)
        summary_rows.append(
            {
                "preset": preset,
                "status": "ok",
                "model_path": str(eval_config.model_path),
                "report_path": str(eval_config.output_path),
                "episodes": args.episodes,
                "seed": args.seed,
                "avg_missed_requests_num": eval_summary["avg_missed_requests_num"],
                "avg_served_requests_num": eval_summary["avg_served_requests_num"],
                "avg_skip_rate": eval_summary["avg_skip_rate"],
                "avg_reward": eval_summary["avg_reward"],
                "avg_unfinished_ratio": eval_summary["avg_unfinished_ratio"],
                "min_missed_requests_num": eval_summary["min_missed_requests_num"],
                "max_missed_requests_num": eval_summary["max_missed_requests_num"],
                "fully_served_rate": eval_summary["fully_served_rate"],
            }
        )

    summary_path = args.base_dir / args.summary_name
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"\nSaved re-evaluation summary to: {summary_path}")


if __name__ == "__main__":
    main()
