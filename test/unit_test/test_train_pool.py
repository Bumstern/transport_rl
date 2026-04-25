import gymnasium
import numpy as np
import pytest

from src.optimizer.train_pool import (
    TrainInstanceSampler,
    TrainPoolEnvWrapper,
    build_train_pool_seeds,
)


class DummyEnv(gymnasium.Env):
    def __init__(self) -> None:
        self.action_space = gymnasium.spaces.Discrete(2)
        self.observation_space = gymnasium.spaces.Discrete(2)
        self.reset_calls: list[tuple[int | None, dict | None]] = []

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append((seed, options))
        return 0, {"seed": seed}

    def step(self, action):
        return 0, 0.0, True, False, {}

    def action_masks(self):
        return np.array([True, False], dtype=bool)


def test_build_train_pool_seeds_is_deterministic() -> None:
    assert build_train_pool_seeds(4, seed=123) == build_train_pool_seeds(4, seed=123)


def test_build_train_pool_seeds_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="pool_size"):
        build_train_pool_seeds(-1, seed=123)


def test_sampler_uses_only_pool_when_ratio_is_one() -> None:
    pool_seeds = [11, 22, 33]
    sampler = TrainInstanceSampler(
        pool_seeds=pool_seeds,
        fixed_pool_ratio=1.0,
        selection_seed=7,
        fresh_seed=8,
    )

    sampled_seeds = [sampler.sample_seed() for _ in range(20)]

    assert all(seed in pool_seeds for seed in sampled_seeds)


def test_sampler_uses_only_fresh_when_ratio_is_zero() -> None:
    pool_seeds = [11, 22, 33]
    sampler = TrainInstanceSampler(
        pool_seeds=pool_seeds,
        fixed_pool_ratio=0.0,
        selection_seed=7,
        fresh_seed=8,
    )

    sampled_seeds = [sampler.sample_seed() for _ in range(20)]

    assert all(seed not in pool_seeds for seed in sampled_seeds)


def test_sampler_is_deterministic_for_same_rng_seeds() -> None:
    sampler_a = TrainInstanceSampler(
        pool_seeds=[10, 20, 30],
        fixed_pool_ratio=0.4,
        selection_seed=123,
        fresh_seed=456,
    )
    sampler_b = TrainInstanceSampler(
        pool_seeds=[10, 20, 30],
        fixed_pool_ratio=0.4,
        selection_seed=123,
        fresh_seed=456,
    )

    sequence_a = [sampler_a.sample_seed() for _ in range(30)]
    sequence_b = [sampler_b.sample_seed() for _ in range(30)]

    assert sequence_a == sequence_b


def test_sampler_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="fixed_pool_ratio"):
        TrainInstanceSampler(pool_seeds=[1], fixed_pool_ratio=1.5)


def test_sampler_requires_pool_when_ratio_is_positive() -> None:
    with pytest.raises(ValueError, match="pool_seeds"):
        TrainInstanceSampler(pool_seeds=[], fixed_pool_ratio=0.1)


def test_wrapper_passes_sampled_seed_to_base_env_reset() -> None:
    env = DummyEnv()
    sampler = TrainInstanceSampler(
        pool_seeds=[101],
        fixed_pool_ratio=1.0,
        selection_seed=1,
        fresh_seed=2,
    )
    wrapped_env = TrainPoolEnvWrapper(env, sampler)

    observation, info = wrapped_env.reset()

    assert observation == 0
    assert info["seed"] == 101
    assert wrapped_env.last_seed == 101
    assert env.reset_calls == [(101, None)]


def test_wrapper_explicit_seed_overrides_sampler() -> None:
    env = DummyEnv()
    sampler = TrainInstanceSampler(
        pool_seeds=[101],
        fixed_pool_ratio=1.0,
        selection_seed=1,
        fresh_seed=2,
    )
    wrapped_env = TrainPoolEnvWrapper(env, sampler)

    _, info = wrapped_env.reset(seed=999, options={"episode": 1})

    assert info["seed"] == 999
    assert wrapped_env.last_seed == 999
    assert env.reset_calls == [(999, {"episode": 1})]


def test_wrapper_proxies_action_masks() -> None:
    wrapped_env = TrainPoolEnvWrapper(
        DummyEnv(),
        TrainInstanceSampler(pool_seeds=[1], fixed_pool_ratio=1.0, selection_seed=1, fresh_seed=2),
    )

    action_masks = wrapped_env.action_masks()

    assert np.array_equal(action_masks, np.array([True, False], dtype=bool))
