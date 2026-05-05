import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback

from src.optimizer.main import SimulatorEnv
from src.optimizer.settings import (
    GENERATOR_SETTINGS,
    DEFAULT_OBSERVATION_FEATURES,
    ObservationFeatureConfig,
)
from src.optimizer.train_pool import (
    TrainInstanceSampler,
    TrainPoolEnvWrapper,
    build_train_pool_seeds,
)
from src.simulator.utils.data_generator.generator import InputDataGenerator

OBSERVATION_PRESETS = (
    "all",
    "no_current_selection",
    "no_unfinished_ratio",
    "no_current_selection_no_unfinished_ratio",
    "no_current_selection_no_unfinished_ratio_no_time_windows",
    "no_executed_requests",
    "pairwise_only",
)


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int
    n_steps: int
    clip_range: float
    learning_rate: float
    learning_rate_schedule: str
    learning_rate_step_points: tuple[float, ...]
    learning_rate_step_values: tuple[float, ...]
    net_arch: list[int]
    eval_freq: int
    n_eval_episodes: int
    seed: int | None
    resume_from: Path | None
    train_pool_size: int
    train_pool_seed: int | None
    fixed_pool_ratio: float
    model_dir: Path
    best_model_dir: Path
    tensorboard_dir: Path
    verbose: int
    observation_feature_config: ObservationFeatureConfig
    early_stop_patience_episodes: int


class EpisodeLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            if "missed_requests_num" in info:
                self.logger.record("episode/missed_requests_num", info["missed_requests_num"])
            if "unfinished_ratio" in info:
                self.logger.record("episode/unfinished_ratio", float(info["unfinished_ratio"][0]))
        current_lr = self.model.policy.optimizer.param_groups[0]["lr"]
        self.logger.record("train/learning_rate", current_lr)
        return True


class MetricsEvalCallback(MaskableEvalCallback):
    def _on_step(self) -> bool:
        should_run_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        should_continue = super()._on_step()
        if should_run_eval:
            self._log_eval_metrics()
        return should_continue

    def _log_eval_metrics(self) -> None:
        observation = self.eval_env.reset()
        completed_episodes = 0
        missed_requests_nums: list[int] = []
        unfinished_ratios: list[float] = []

        while completed_episodes < self.n_eval_episodes:
            action_masks = get_action_masks(self.eval_env)
            actions, _ = self.model.predict(
                observation,
                action_masks=action_masks,
                deterministic=self.deterministic,
            )
            observation, _, dones, infos = self.eval_env.step(actions)

            for done, info in zip(dones, infos):
                if not done:
                    continue
                if "missed_requests_num" in info:
                    missed_requests_nums.append(int(info["missed_requests_num"]))
                if "unfinished_ratio" in info:
                    unfinished_ratios.append(float(info["unfinished_ratio"][0]))
                completed_episodes += 1
                if completed_episodes >= self.n_eval_episodes:
                    break

        if missed_requests_nums:
            self.logger.record(
                "eval/avg_missed_requests_num",
                sum(missed_requests_nums) / len(missed_requests_nums),
            )
        if unfinished_ratios:
            self.logger.record(
                "eval/avg_unfinished_ratio",
                sum(unfinished_ratios) / len(unfinished_ratios),
            )


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self._patience_episodes = patience_episodes
        self._best_unfinished_ratio: float | None = None
        self._episodes_since_improvement = 0

    def _on_step(self) -> bool:
        if self._patience_episodes <= 0:
            return True

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done or "unfinished_ratio" not in info:
                continue

            current_unfinished_ratio = float(info["unfinished_ratio"][0])
            if (
                self._best_unfinished_ratio is None
                or current_unfinished_ratio < self._best_unfinished_ratio
            ):
                self._best_unfinished_ratio = current_unfinished_ratio
                self._episodes_since_improvement = 0
            else:
                self._episodes_since_improvement += 1

        if self._best_unfinished_ratio is not None:
            self.logger.record("train/best_unfinished_ratio", self._best_unfinished_ratio)
        self.logger.record("train/episodes_since_improvement", self._episodes_since_improvement)

        if self._episodes_since_improvement >= self._patience_episodes:
            if self.verbose > 0:
                print(
                    "Early stopping triggered:"
                    f" no improvement for {self._episodes_since_improvement} completed episodes."
                )
            return False
        return True


def quarter_decay_schedule(progress_remaining: float) -> float:
    if progress_remaining >= 0.75:
        return 3e-3
    if progress_remaining >= 0.5:
        return 7e-4
    if progress_remaining >= 0.25:
        return 4e-4
    return 1e-4


def _parse_float_tuple(raw_value: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in raw_value.split(",") if value.strip())


def build_piecewise_schedule(step_points: tuple[float, ...], step_values: tuple[float, ...]) -> Callable[[float], float]:
    if len(step_values) != len(step_points) + 1:
        raise ValueError("learning_rate_step_values must contain exactly one more value than learning_rate_step_points")
    if any(point < 0.0 or point > 1.0 for point in step_points):
        raise ValueError("learning_rate_step_points must be in [0, 1]")
    if tuple(sorted(step_points, reverse=True)) != step_points:
        raise ValueError("learning_rate_step_points must be sorted in descending order")

    def schedule(progress_remaining: float) -> float:
        for point, value in zip(step_points, step_values):
            if progress_remaining >= point:
                return value
        return step_values[-1]

    return schedule


def build_learning_rate(config: TrainConfig):
    if config.learning_rate_schedule == "quarter_decay":
        return quarter_decay_schedule
    if config.learning_rate_schedule == "piecewise":
        return build_piecewise_schedule(config.learning_rate_step_points, config.learning_rate_step_values)
    if config.learning_rate_schedule == "constant":
        return config.learning_rate
    raise ValueError(f"Unknown learning rate schedule: {config.learning_rate_schedule}")


def build_generator(seed: int | None = None) -> InputDataGenerator:
    return InputDataGenerator(
        load_point_names=GENERATOR_SETTINGS.load_point_names,
        unload_point_names=GENERATOR_SETTINGS.unload_point_names,
        requests_num_min=GENERATOR_SETTINGS.min_requests_num,
        requests_num_max=GENERATOR_SETTINGS.max_requests_num,
        trucks_num=GENERATOR_SETTINGS.max_truck_num,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, "%d.%m.%Y"),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, "%d.%m.%Y"),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        load_to_load_distance_range=GENERATOR_SETTINGS.load_to_load_distance_range.model_dump(),
        unload_to_unload_distance_range=GENERATOR_SETTINGS.unload_to_unload_distance_range.model_dump(),
        load_to_unload_distance_range=GENERATOR_SETTINGS.load_to_unload_distance_range.model_dump(),
        seed=seed,
    )


def get_observation_feature_config(
    preset: str,
    pairwise_lookahead_requests: int = 1,
) -> ObservationFeatureConfig:
    if pairwise_lookahead_requests <= 0:
        raise ValueError("pairwise_lookahead_requests must be positive")

    presets = {
        "all": DEFAULT_OBSERVATION_FEATURES,
        "no_current_selection": DEFAULT_OBSERVATION_FEATURES.model_copy(update={"use_current_selection": False}),
        "no_unfinished_ratio": DEFAULT_OBSERVATION_FEATURES.model_copy(update={"use_unfinished_ratio": False}),
        "no_current_selection_no_unfinished_ratio": DEFAULT_OBSERVATION_FEATURES.model_copy(
            update={
                "use_current_selection": False,
                "use_unfinished_ratio": False,
            }
        ),
        "no_current_selection_no_unfinished_ratio_no_time_windows": DEFAULT_OBSERVATION_FEATURES.model_copy(
            update={
                "use_current_selection": False,
                "use_unfinished_ratio": False,
                "use_time_windows": False,
            }
        ),
        "no_executed_requests": DEFAULT_OBSERVATION_FEATURES.model_copy(update={"use_executed_requests": False}),
        "pairwise_only": DEFAULT_OBSERVATION_FEATURES.model_copy(
            update={
                "use_executed_requests": False,
                "use_unfinished_ratio": False,
                "use_current_selection": False,
                "use_next_request_tw": True,
                "use_time_windows": True,
            }
        ),
    }
    return presets[preset].model_copy(
        update={"pairwise_lookahead_requests": pairwise_lookahead_requests}
    )


def build_fixed_instances(count: int, seed: int | None) -> list[tuple[dict, list[dict]]]:
    if count <= 0:
        raise ValueError("count must be positive")
    return build_generator(seed=seed).generate_many(count)


def build_env(
    observation_feature_config: ObservationFeatureConfig,
    *,
    seed: int | None = None,
    fixed_instances: list[tuple[dict, list[dict]]] | None = None,
) -> SimulatorEnv:
    return SimulatorEnv(
        build_generator(seed=seed),
        observation_feature_config,
        fixed_instances=fixed_instances,
    )


def ensure_output_dirs(config: TrainConfig) -> None:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.best_model_dir.mkdir(parents=True, exist_ok=True)
    config.tensorboard_dir.mkdir(parents=True, exist_ok=True)


def _hash_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def serialize_train_config(config: TrainConfig) -> dict:
    return {
        "total_timesteps": config.total_timesteps,
        "n_steps": config.n_steps,
        "clip_range": config.clip_range,
        "learning_rate": config.learning_rate,
        "learning_rate_schedule": config.learning_rate_schedule,
        "learning_rate_step_points": list(config.learning_rate_step_points),
        "learning_rate_step_values": list(config.learning_rate_step_values),
        "net_arch": list(config.net_arch),
        "eval_freq": config.eval_freq,
        "n_eval_episodes": config.n_eval_episodes,
        "seed": config.seed,
        "resume_from": str(config.resume_from) if config.resume_from is not None else None,
        "train_pool_size": config.train_pool_size,
        "train_pool_seed": config.train_pool_seed,
        "fixed_pool_ratio": config.fixed_pool_ratio,
        "model_dir": str(config.model_dir),
        "best_model_dir": str(config.best_model_dir),
        "tensorboard_dir": str(config.tensorboard_dir),
        "verbose": config.verbose,
        "observation_feature_config": config.observation_feature_config.model_dump(),
        "early_stop_patience_episodes": config.early_stop_patience_episodes,
    }


def build_training_metadata(config: TrainConfig) -> dict:
    routes_path = Path("input/routes.json")
    generator_settings_path = Path("input/generator_settings.json")
    return {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config": serialize_train_config(config),
        "generator_settings": json.loads(generator_settings_path.read_text()),
        "routes": {
            "path": str(routes_path),
            "sha256": _hash_file(routes_path),
        },
    }


def save_training_metadata(model_path: Path, config: TrainConfig) -> Path:
    metadata_path = model_path.with_suffix(".config.json")
    metadata_path.write_text(
        json.dumps(build_training_metadata(config), ensure_ascii=False, indent=2)
    )
    return metadata_path


def build_model(config: TrainConfig, env: SimulatorEnv) -> MaskablePPO:
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        n_steps=config.n_steps,
        clip_range=config.clip_range,
        learning_rate=build_learning_rate(config),
        verbose=config.verbose,
        tensorboard_log=str(config.tensorboard_dir),
        seed=config.seed,
        policy_kwargs={
            "net_arch": config.net_arch,
        },
    )


def load_or_build_model(config: TrainConfig, env: SimulatorEnv) -> MaskablePPO:
    if config.resume_from is None:
        return build_model(config, env)

    model = MaskablePPO.load(str(config.resume_from))
    model.set_env(env)
    return model


def train(config: TrainConfig) -> Path:
    ensure_output_dirs(config)

    train_pool_seeds = build_train_pool_seeds(config.train_pool_size, config.train_pool_seed)
    train_sampler = TrainInstanceSampler(
        pool_seeds=train_pool_seeds,
        fixed_pool_ratio=config.fixed_pool_ratio,
        selection_seed=config.seed,
        fresh_seed=None if config.seed is None else config.seed + 10_000,
    )
    train_env = TrainPoolEnvWrapper(
        build_env(config.observation_feature_config, seed=config.seed),
        train_sampler,
    )
    eval_seed = None if config.seed is None else config.seed + 1
    eval_instances = build_fixed_instances(config.n_eval_episodes, seed=eval_seed)
    eval_env = build_env(
        config.observation_feature_config,
        seed=eval_seed,
        fixed_instances=eval_instances,
    )
    model = load_or_build_model(config, train_env)

    eval_callback = MetricsEvalCallback(
        eval_env,
        best_model_save_path=str(config.best_model_dir),
        log_path=str(config.tensorboard_dir),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    episode_logger_callback = EpisodeLoggerCallback()
    early_stopping_callback = EarlyStoppingCallback(
        patience_episodes=config.early_stop_patience_episodes,
        verbose=config.verbose,
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        progress_bar=True,
        callback=[eval_callback, episode_logger_callback, early_stopping_callback],
        reset_num_timesteps=config.resume_from is None,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = config.model_dir / f"{timestamp}.zip"
    model.save(str(model_path))
    save_training_metadata(model_path, config)
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
        default=1024,
        help="Number of rollout steps per update.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.1,
        help="PPO clip range.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=7e-4,
        help="Optimizer learning rate for constant schedule.",
    )
    parser.add_argument(
        "--learning-rate-schedule",
        choices=["constant", "quarter_decay", "piecewise"],
        default="quarter_decay",
        help="Learning rate schedule. quarter_decay uses 3e-3 -> 7e-4 -> 4e-4 -> 1e-4 across training quarters.",
    )
    parser.add_argument(
        "--learning-rate-step-points",
        type=str,
        default="0.75,0.5,0.25",
        help="Descending progress points for piecewise learning-rate schedule, e.g. '0.75,0.5,0.25'.",
    )
    parser.add_argument(
        "--learning-rate-step-values",
        type=str,
        default="3e-3,7e-4,4e-4,1e-4",
        help="Learning-rate values for piecewise schedule. Must contain one more value than step points.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=4096,
        help="Evaluation frequency in environment steps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of fixed evaluation instances created once at training start.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed for training.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional path to an existing model checkpoint to continue training from.",
    )
    parser.add_argument(
        "--train-pool-size",
        type=int,
        default=128,
        help="Number of fixed train-pool seeds sampled once at training start.",
    )
    parser.add_argument(
        "--train-pool-seed",
        type=int,
        default=12345,
        help="Seed used to create the fixed train-pool seed list.",
    )
    parser.add_argument(
        "--fixed-pool-ratio",
        type=float,
        default=0.3,
        help="Probability of taking the next training episode from the fixed train-pool instead of fresh generation.",
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
    parser.add_argument(
        "--observation-preset",
        choices=OBSERVATION_PRESETS,
        default="all",
        help="Observation ablation preset.",
    )
    parser.add_argument(
        "--pairwise-lookahead-requests",
        type=int,
        default=1,
        help="How many future requests to encode in pairwise truck-request features.",
    )
    parser.add_argument(
        "--early-stop-patience-episodes",
        "--early-stop-patience-epochs",
        type=int,
        dest="early_stop_patience_episodes",
        default=0,
        help="Stop training if unfinished_ratio does not improve for this many completed episodes. 0 disables early stopping.",
    )
    args = parser.parse_args()

    return TrainConfig(
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        clip_range=args.clip_range,
        learning_rate=args.learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        learning_rate_step_points=_parse_float_tuple(args.learning_rate_step_points),
        learning_rate_step_values=_parse_float_tuple(args.learning_rate_step_values),
        net_arch=[128, 128, 128],
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        seed=args.seed,
        resume_from=args.resume_from,
        train_pool_size=args.train_pool_size,
        train_pool_seed=args.train_pool_seed,
        fixed_pool_ratio=args.fixed_pool_ratio,
        model_dir=args.model_dir,
        best_model_dir=args.best_model_dir,
        tensorboard_dir=args.tensorboard_dir,
        verbose=args.verbose,
        observation_feature_config=get_observation_feature_config(
            args.observation_preset,
            pairwise_lookahead_requests=args.pairwise_lookahead_requests,
        ),
        early_stop_patience_episodes=args.early_stop_patience_episodes,
    )


def main() -> None:
    config = parse_args()
    model_path = train(config)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
