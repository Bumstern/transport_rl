import pytest
import numpy as np

from src.optimizer.train import EarlyStoppingCallback
from src.optimizer.train import EpisodeLoggerCallback
from src.optimizer.train import build_piecewise_schedule
from src.optimizer.train import quarter_decay_schedule


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
