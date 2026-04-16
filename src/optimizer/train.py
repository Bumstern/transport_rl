import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback

from src.optimizer.main import SimulatorEnv
from src.optimizer.settings import GENERATOR_SETTINGS
from src.simulator.utils.data_generator.generator import InputDataGenerator


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int
    n_steps: int
    clip_range: float
    learning_rate: float
    net_arch: list[int]
    eval_freq: int
    seed: int | None
    model_dir: Path
    best_model_dir: Path
    tensorboard_dir: Path
    verbose: int


class InfoLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "missed_requests_num" in info:
                self.logger.record("custom/missed_requests_num", info["missed_requests_num"])
            if "unfinished_ratio" in info:
                self.logger.record("custom/unfinished_ratio", float(info["unfinished_ratio"][0]))
        return True


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


def ensure_output_dirs(config: TrainConfig) -> None:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.best_model_dir.mkdir(parents=True, exist_ok=True)
    config.tensorboard_dir.mkdir(parents=True, exist_ok=True)


def build_model(config: TrainConfig, env: SimulatorEnv) -> MaskablePPO:
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        n_steps=config.n_steps,
        clip_range=config.clip_range,
        learning_rate=config.learning_rate,
        verbose=config.verbose,
        tensorboard_log=str(config.tensorboard_dir),
        seed=config.seed,
        policy_kwargs={
            "net_arch": config.net_arch,
        },
    )


def train(config: TrainConfig) -> Path:
    ensure_output_dirs(config)

    train_env = build_env()
    eval_env = build_env()
    model = build_model(config, train_env)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(config.best_model_dir),
        log_path=str(config.tensorboard_dir),
        eval_freq=config.eval_freq,
        deterministic=True,
        render=False,
    )
    info_logger_callback = InfoLoggerCallback()

    model.learn(
        total_timesteps=config.total_timesteps,
        progress_bar=True,
        callback=[eval_callback, info_logger_callback],
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = config.model_dir / f"{timestamp}.zip"
    model.save(str(model_path))
    return model_path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on SimulatorEnv.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1024 * GENERATOR_SETTINGS.max_requests_num,
        help="Total number of training timesteps.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
        help="Number of rollout steps per update.",
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
        help="Evaluation frequency in environment steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed for training.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("output/models"),
        help="Directory for saving timestamped checkpoints.",
    )
    parser.add_argument(
        "--best-model-dir",
        type=Path,
        default=Path("output/models/best"),
        help="Directory for saving the best evaluation checkpoint.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=Path("output/logs/tensorboard"),
        help="Directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Stable-Baselines3 verbosity level.",
    )
    args = parser.parse_args()

    return TrainConfig(
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        clip_range=args.clip_range,
        learning_rate=args.learning_rate,
        net_arch=[128, 128, 128],
        eval_freq=args.eval_freq,
        seed=args.seed,
        model_dir=args.model_dir,
        best_model_dir=args.best_model_dir,
        tensorboard_dir=args.tensorboard_dir,
        verbose=args.verbose,
    )


def main() -> None:
    config = parse_args()
    model_path = train(config)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
