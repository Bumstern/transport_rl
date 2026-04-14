import random
from datetime import datetime

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from src.optimizer.main import SimulatorEnv
from src.optimizer.settings import GENERATOR_SETTINGS
from src.simulator.environment import Environment
from src.simulator.model.simulator import Simulator
from src.simulator.utils.data_generator.generator import InputDataGenerator


@pytest.fixture(scope='module')
def rl_env() -> SimulatorEnv:
    generator = InputDataGenerator(
        load_point_names=GENERATOR_SETTINGS.load_point_names,
        unload_point_names=GENERATOR_SETTINGS.unload_point_names,
        requests_num_min=GENERATOR_SETTINGS.min_requests_num,
        requests_num_max=GENERATOR_SETTINGS.max_requests_num,
        trucks_num=GENERATOR_SETTINGS.max_truck_num,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, '%d.%m.%Y'),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, '%d.%m.%Y'),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        min_distance=GENERATOR_SETTINGS.min_distance,
        max_distance=GENERATOR_SETTINGS.max_distance
    )
    env = SimulatorEnv(generator)
    return env


def test_simulator_run(simulator: Simulator, environment: Environment, requests_constraints: list[list[int]]):
    # Тестируем обычный запуск в ограничениях
    list_with_missed, truck_positions, truck_available_times = simulator.run(
        [random.choice(req_constr) for req_constr in requests_constraints]
    )
    assert isinstance(list_with_missed, list)
    assert isinstance(truck_positions, list)
    assert isinstance(truck_available_times, list)
    assert len(truck_positions) == len(environment.trucks)
    assert len(truck_available_times) == len(environment.trucks)

    # Тестируем, что запуск вне ограничения выкинет ошибку
    all_trucks_ids = np.arange(start=0, stop=len(environment.trucks), step=1, dtype=int)
    with pytest.raises(AssertionError):
        simulator.run([random.choice(list(set(all_trucks_ids).symmetric_difference(set(req_constr))))
                       for req_constr in requests_constraints])


def _assert_observation_is_valid(rl_env: SimulatorEnv, observation: dict) -> None:
    assert rl_env.observation_space.contains(observation)
    assert observation["travel_time_to_load"].shape == (GENERATOR_SETTINGS.max_truck_num,)
    assert observation["time_slack_to_window_start"].shape == (GENERATOR_SETTINGS.max_truck_num,)
    assert np.all(observation["travel_time_to_load"] >= 0.0)
    assert np.all(observation["travel_time_to_load"] <= 1.0)
    assert np.all(observation["time_slack_to_window_start"] >= -1.0)
    assert np.all(observation["time_slack_to_window_start"] <= 1.0)


def test_rl_env_masked_episode_smoke(rl_env: SimulatorEnv):
    observation, info = rl_env.reset(seed=42)
    _assert_observation_is_valid(rl_env, observation)
    assert info["missed_requests_num"] == 0
    assert info["current_selection"] == []

    step_count = 0
    terminated = False

    while not terminated:
        current_request_id = len(rl_env._current_selection)
        action_mask = rl_env.action_masks()

        assert action_mask.dtype == bool
        assert action_mask.shape == (GENERATOR_SETTINGS.max_truck_num + 1,)
        assert np.any(action_mask)

        allowed_actions = np.flatnonzero(action_mask)
        allowed_truck_ids = set(allowed_actions - 1)
        expected_allowed_truck_ids = set(rl_env._current_requests_constrains[current_request_id])
        assert allowed_truck_ids == expected_allowed_truck_ids

        action = int(random.choice(allowed_actions))
        observation, reward, terminated, truncated, info = rl_env.step(action)

        step_count += 1
        _assert_observation_is_valid(rl_env, observation)
        assert truncated is False
        assert reward in (-1, 1)
        assert len(info["current_selection"]) == step_count
        assert info["missed_requests_num"] >= 0
        assert info["missed_requests_num"] <= len(info["current_selection"])

        if not terminated:
            next_mask = rl_env.action_masks()
            assert next_mask.shape == (GENERATOR_SETTINGS.max_truck_num + 1,)

    assert step_count == rl_env._current_env.requests_num
    assert len(rl_env._current_selection) == rl_env._current_env.requests_num


# # Запускает случайные действия, поэтому чтобы он не упал нужно запускать с _apply_restrictions_to_selection
# def test_rl_env_check(rl_env: SimulatorEnv):
#     check_env(rl_env)
