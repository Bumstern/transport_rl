import gymnasium
import numpy as np


class TrainInstanceSampler:
    def __init__(
        self,
        pool_seeds: list[int],
        fixed_pool_ratio: float,
        selection_seed: int | None = None,
        fresh_seed: int | None = None,
    ) -> None:
        if not 0.0 <= fixed_pool_ratio <= 1.0:
            raise ValueError("fixed_pool_ratio must be in [0, 1]")
        if fixed_pool_ratio > 0.0 and not pool_seeds:
            raise ValueError("pool_seeds must be non-empty when fixed_pool_ratio > 0")

        self._pool_seeds = list(pool_seeds)
        self._fixed_pool_ratio = fixed_pool_ratio
        self._selection_rng = np.random.default_rng(selection_seed)
        self._fresh_rng = np.random.default_rng(fresh_seed)

    def sample_seed(self) -> int:
        use_fixed_pool = (
            len(self._pool_seeds) > 0
            and self._selection_rng.random() < self._fixed_pool_ratio
        )
        if use_fixed_pool:
            return int(self._selection_rng.choice(np.array(self._pool_seeds, dtype=np.int64)))
        return int(self._fresh_rng.integers(0, np.iinfo(np.int32).max))


class TrainPoolEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, sampler: TrainInstanceSampler):
        super().__init__(env)
        self._sampler = sampler
        self._last_seed: int | None = None

    @property
    def last_seed(self) -> int | None:
        return self._last_seed

    def reset(self, *, seed=None, options=None):
        selected_seed = self._sampler.sample_seed() if seed is None else seed
        self._last_seed = selected_seed
        return self.env.reset(seed=selected_seed, options=options)

    def action_masks(self):
        return self.env.action_masks()


def build_train_pool_seeds(pool_size: int, seed: int | None) -> list[int]:
    if pool_size < 0:
        raise ValueError("pool_size must be non-negative")
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, np.iinfo(np.int32).max)) for _ in range(pool_size)]
