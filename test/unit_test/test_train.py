import json
from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np

from src.optimizer.train import EarlyStoppingCallback
from src.optimizer.train import EpisodeLoggerCallback
from src.optimizer.train import TrainConfig
from src.optimizer.train import build_fixed_instances
from src.optimizer.train import build_training_metadata
from src.optimizer.train import build_piecewise_schedule
from src.optimizer.train import build_env
from src.optimizer.train import get_observation_feature_config
from src.optimizer.train import load_or_build_model
from src.optimizer.train import quarter_decay_schedule
from src.optimizer.train import save_training_metadata
from src.optimizer.train import serialize_train_config
from src.optimizer.settings import DEFAULT_OBSERVATION_FEATURES


class DummyLogger:
    def __init__(self) -> None:
        self.records = {}

    def record(self, key: str, value: float) -> None:
        self.records[key] = value


class DummyModel:
    def __init__(self) -> None:
        self.logger = DummyLogger()
        self.policy = type(
            "DummyPolicy",
            (),
            {"optimizer": type("DummyOptimizer", (), {"param_groups": [{"lr": 7e-4}]})()},
        )()


def _build_callback(patience_episodes: int) -> EarlyStoppingCallback:
    callback = EarlyStoppingCallback(patience_episodes=patience_episodes)
    callback.model = DummyModel()
    return callback


def _build_train_config(**overrides) -> TrainConfig:
    base_config = TrainConfig(
        total_timesteps=1024,
        n_steps=256,
        clip_range=0.1,
        learning_rate=7e-4,
        learning_rate_schedule="constant",
        learning_rate_step_points=(0.75, 0.5, 0.25),
        learning_rate_step_values=(3e-3, 7e-4, 4e-4, 1e-4),
        net_arch=[128, 128, 128],
        eval_freq=512,
        n_eval_episodes=10,
        seed=42,
        resume_from=None,
        train_pool_size=128,
        train_pool_seed=12345,
        fixed_pool_ratio=0.3,
        model_dir=Path("output/models"),
        best_model_dir=Path("output/models/best"),
        tensorboard_dir=Path("output/logs/tensorboard"),
        verbose=1,
        observation_feature_config=DEFAULT_OBSERVATION_FEATURES,
        early_stop_patience_episodes=0,
    )
    return replace(base_config, **overrides)


def test_episode_logger_callback_logs_only_completed_episode_metrics() -> None:
    callback = EpisodeLoggerCallback()
    callback.model = DummyModel()
    callback.locals = {
        "infos": [
            {"missed_requests_num": 3, "unfinished_ratio": np.array([0.6], dtype=np.float32)},
            {"missed_requests_num": 4, "unfinished_ratio": np.array([0.8], dtype=np.float32)},
        ],
        "dones": [False, True],
    }

    assert callback._on_step() is True
    assert "custom/missed_requests_num" not in callback.logger.records
    assert "custom/unfinished_ratio" not in callback.logger.records
    assert callback.logger.records["episode/missed_requests_num"] == 4
    assert callback.logger.records["episode/unfinished_ratio"] == pytest.approx(0.8)
    assert callback.logger.records["train/learning_rate"] == pytest.approx(7e-4)


def test_quarter_decay_schedule_switches_learning_rate_by_quarters() -> None:
    assert quarter_decay_schedule(1.0) == pytest.approx(3e-3)
    assert quarter_decay_schedule(0.75) == pytest.approx(3e-3)

    assert quarter_decay_schedule(0.74) == pytest.approx(7e-4)
    assert quarter_decay_schedule(0.50) == pytest.approx(7e-4)

    assert quarter_decay_schedule(0.49) == pytest.approx(4e-4)
    assert quarter_decay_schedule(0.25) == pytest.approx(4e-4)

    assert quarter_decay_schedule(0.24) == pytest.approx(1e-4)
    assert quarter_decay_schedule(0.0) == pytest.approx(1e-4)


def test_piecewise_schedule_uses_configured_thresholds() -> None:
    schedule = build_piecewise_schedule(
        step_points=(0.75, 0.5, 0.25),
        step_values=(3e-3, 7e-4, 4e-4, 1e-4),
    )

    assert schedule(1.0) == pytest.approx(3e-3)
    assert schedule(0.75) == pytest.approx(3e-3)

    assert schedule(0.74) == pytest.approx(7e-4)
    assert schedule(0.50) == pytest.approx(7e-4)

    assert schedule(0.49) == pytest.approx(4e-4)
    assert schedule(0.25) == pytest.approx(4e-4)

    assert schedule(0.24) == pytest.approx(1e-4)
    assert schedule(0.0) == pytest.approx(1e-4)


def test_piecewise_schedule_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="one more"):
        build_piecewise_schedule(step_points=(0.75, 0.5), step_values=(3e-3, 7e-4))

    with pytest.raises(ValueError, match="descending order"):
        build_piecewise_schedule(
            step_points=(0.5, 0.75, 0.25),
            step_values=(3e-3, 7e-4, 4e-4, 1e-4),
        )


def test_early_stopping_callback_stops_after_patience_completed_episodes() -> None:
    callback = _build_callback(patience_episodes=2)

    callback.locals = {
        "infos": [{"unfinished_ratio": np.array([0.8], dtype=np.float32)}],
        "dones": [True],
    }
    assert callback._on_step() is True
    assert callback._best_unfinished_ratio == pytest.approx(0.8)
    assert callback._episodes_since_improvement == 0

    callback.locals = {
        "infos": [{"unfinished_ratio": np.array([0.85], dtype=np.float32)}],
        "dones": [True],
    }
    assert callback._on_step() is True
    assert callback._episodes_since_improvement == 1

    callback.locals = {
        "infos": [{"unfinished_ratio": np.array([0.9], dtype=np.float32)}],
        "dones": [True],
    }
    assert callback._on_step() is False
    assert callback._episodes_since_improvement == 2
    assert callback.logger.records["train/best_unfinished_ratio"] == pytest.approx(0.8)
    assert callback.logger.records["train/episodes_since_improvement"] == 2


def test_early_stopping_callback_resets_patience_after_improvement() -> None:
    callback = _build_callback(patience_episodes=2)

    callback.locals = {
        "infos": [{"unfinished_ratio": np.array([0.8], dtype=np.float32)}],
        "dones": [True],
    }
    assert callback._on_step() is True

    callback.locals = {
        "infos": [{"unfinished_ratio": np.array([0.85], dtype=np.float32)}],
        "dones": [True],
    }
    assert callback._on_step() is True
    assert callback._episodes_since_improvement == 1

    callback.locals = {
        "infos": [{"unfinished_ratio": np.array([0.7], dtype=np.float32)}],
        "dones": [True],
    }
    assert callback._on_step() is True
    assert callback._best_unfinished_ratio == pytest.approx(0.7)
    assert callback._episodes_since_improvement == 0


def test_early_stopping_callback_ignores_unfinished_ratio_until_episode_done() -> None:
    callback = _build_callback(patience_episodes=1)

    callback.locals = {
        "infos": [{"unfinished_ratio": np.array([0.8], dtype=np.float32)}],
        "dones": [False],
    }
    assert callback._on_step() is True
    assert callback._best_unfinished_ratio is None
    assert callback._episodes_since_improvement == 0


def test_build_fixed_instances_is_deterministic_for_same_seed() -> None:
    first_instances = build_fixed_instances(3, seed=123)
    second_instances = build_fixed_instances(3, seed=123)

    assert first_instances == second_instances


def test_build_env_cycles_over_fixed_instances() -> None:
    fixed_instances = build_fixed_instances(2, seed=321)
    env = build_env(
        DEFAULT_OBSERVATION_FEATURES,
        seed=321,
        fixed_instances=fixed_instances,
    )

    env.reset()
    first_signature = (
        env._current_env.requests_num,
        [request.point_to_load.name for request in env._current_env.requests],
    )
    env.reset()
    second_signature = (
        env._current_env.requests_num,
        [request.point_to_load.name for request in env._current_env.requests],
    )
    env.reset()
    cycled_signature = (
        env._current_env.requests_num,
        [request.point_to_load.name for request in env._current_env.requests],
    )

    assert first_signature != second_signature
    assert cycled_signature == first_signature


def test_get_observation_feature_config_applies_pairwise_lookahead() -> None:
    config = get_observation_feature_config(
        "no_current_selection",
        pairwise_lookahead_requests=3,
    )

    assert config.use_current_selection is False
    assert config.use_pairwise_features is True
    assert config.pairwise_lookahead_requests == 3


def test_get_observation_feature_config_supports_no_time_windows_preset() -> None:
    config = get_observation_feature_config(
        "no_current_selection_no_unfinished_ratio_no_time_windows",
        pairwise_lookahead_requests=2,
    )

    assert config.use_current_selection is False
    assert config.use_unfinished_ratio is False
    assert config.use_time_windows is False
    assert config.use_next_request_tw is True
    assert config.use_pairwise_features is True
    assert config.pairwise_lookahead_requests == 2


def test_serialize_train_config_includes_resume_path() -> None:
    config = _build_train_config(resume_from=Path("output/models/checkpoint.zip"))

    serialized = serialize_train_config(config)

    assert serialized["resume_from"] == "output/models/checkpoint.zip"
    assert serialized["observation_feature_config"]["pairwise_lookahead_requests"] == 1


def test_build_training_metadata_contains_routes_hash() -> None:
    config = _build_train_config()

    metadata = build_training_metadata(config)

    assert metadata["config"]["seed"] == 42
    assert metadata["routes"]["path"] == "input/routes.json"
    assert len(metadata["routes"]["sha256"]) == 64
    assert metadata["generator_settings"]["max_truck_num"] > 0


def test_save_training_metadata_writes_json_next_to_model(tmp_path) -> None:
    config = _build_train_config(
        model_dir=tmp_path,
        best_model_dir=tmp_path / "best",
        tensorboard_dir=tmp_path / "tb",
    )
    model_path = tmp_path / "model.zip"

    metadata_path = save_training_metadata(model_path, config)

    assert metadata_path == tmp_path / "model.config.json"
    written = json.loads(metadata_path.read_text())
    assert written["config"]["model_dir"] == str(tmp_path)


def test_load_or_build_model_loads_checkpoint_and_sets_env(monkeypatch) -> None:
    loaded_paths = []
    env_markers = []

    class LoadedModel:
        def set_env(self, env) -> None:
            env_markers.append(env)

    loaded_model = LoadedModel()

    monkeypatch.setattr(
        "src.optimizer.train.MaskablePPO.load",
        lambda path: loaded_paths.append(path) or loaded_model,
    )

    config = _build_train_config(resume_from=Path("output/models/checkpoint.zip"))
    env = object()

    model = load_or_build_model(config, env)

    assert model is loaded_model
    assert loaded_paths == ["output/models/checkpoint.zip"]
    assert env_markers == [env]
