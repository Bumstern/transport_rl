import random
from datetime import datetime

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from src.optimizer.main import SimulatorEnv
from src.optimizer.settings import GENERATOR_SETTINGS, DEFAULT_OBSERVATION_FEATURES
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
        load_to_load_distance_range=GENERATOR_SETTINGS.load_to_load_distance_range.model_dump(),
        unload_to_unload_distance_range=GENERATOR_SETTINGS.unload_to_unload_distance_range.model_dump(),
        load_to_unload_distance_range=GENERATOR_SETTINGS.load_to_unload_distance_range.model_dump(),
        seed=42,
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
    expected_pairwise_shape = (
        rl_env._observation_feature_config.pairwise_lookahead_requests,
        GENERATOR_SETTINGS.max_truck_num,
    )
    assert observation["travel_time_to_load"].shape == expected_pairwise_shape
    assert observation["travel_time_with_cargo_to_unload"].shape == expected_pairwise_shape
    assert observation["earliness_to_window_start"].shape == expected_pairwise_shape
    assert observation["lateness_to_window_start"].shape == expected_pairwise_shape
    assert np.all(observation["travel_time_to_load"] >= 0.0)
    assert np.all(observation["travel_time_to_load"] <= 1.0)
    assert np.all(observation["travel_time_with_cargo_to_unload"] >= 0.0)
    assert np.all(observation["travel_time_with_cargo_to_unload"] <= 1.0)
    assert np.all(observation["earliness_to_window_start"] >= 0.0)
    assert np.all(observation["earliness_to_window_start"] <= 1.0)
    assert np.all(observation["lateness_to_window_start"] >= 0.0)
    assert np.all(observation["lateness_to_window_start"] <= 1.0)


def _assert_reward_is_valid(reward: float) -> None:
    # Базовая часть награды теперь зависит от типа действия:
    # skip -> -2, miss -> -1, success -> +1, плюс shaping по slack в диапазоне [-1, 1],
    # и опциональный terminal bonus.
    assert -3.0 <= reward <= 100.0


def _copy_observation(observation: dict) -> dict:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in observation.items()
    }


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
        _assert_reward_is_valid(reward)
        assert len(info["current_selection"]) == step_count
        assert info["missed_requests_num"] >= 0
        assert info["missed_requests_num"] <= len(info["current_selection"])

        if not terminated:
            next_mask = rl_env.action_masks()
            assert next_mask.shape == (GENERATOR_SETTINGS.max_truck_num + 1,)

    assert step_count == rl_env._current_env.requests_num
    assert len(rl_env._current_selection) == rl_env._current_env.requests_num


def test_reward_uses_previous_observation_slack(rl_env: SimulatorEnv, monkeypatch: pytest.MonkeyPatch):
    observation, _ = rl_env.reset(seed=42)
    action_mask = rl_env.action_masks()
    action = int(next(action for action in np.flatnonzero(action_mask) if action != 0))
    truck_id = action - 1

    previous_observation = _copy_observation(observation)
    previous_observation["earliness_to_window_start"][0][truck_id] = np.float32(0.0625)
    previous_observation["lateness_to_window_start"][0][truck_id] = np.float32(0.0)
    rl_env._current_observation = previous_observation

    truck_positions = [truck.position.current_point.model_copy(deep=True) for truck in rl_env._current_env.trucks]
    truck_available_times = [0] * len(rl_env._current_env.trucks)

    monkeypatch.setattr(
        rl_env._simulator,
        "run",
        lambda selection, env: ([], truck_positions, truck_available_times)
    )

    original_create_observation = rl_env._obs_builder.create_observation

    def fake_create_observation(*args, **kwargs):
        next_observation = original_create_observation(*args, **kwargs)
        next_observation["earliness_to_window_start"][0][truck_id] = np.float32(0.0)
        next_observation["lateness_to_window_start"][0][truck_id] = np.float32(1.0)
        return next_observation

    monkeypatch.setattr(rl_env._obs_builder, "create_observation", fake_create_observation)

    _, reward, _, truncated, _ = rl_env.step(action)

    assert truncated is False
    assert reward == pytest.approx(1.0 + rl_env._slack_penalty(0.0625))


def test_reward_penalizes_skip_with_minus_two(rl_env: SimulatorEnv):
    rl_env.reset(seed=42)
    action_mask = rl_env.action_masks()

    assert action_mask[0]

    _, reward, _, truncated, info = rl_env.step(0)

    assert truncated is False
    assert reward == pytest.approx(-2.0)
    assert info["current_selection"] == [-1]


def test_reward_is_positive_for_successful_assignment(rl_env: SimulatorEnv, monkeypatch: pytest.MonkeyPatch):
    observation, _ = rl_env.reset(seed=42)
    action_mask = rl_env.action_masks()
    action = int(next(action for action in np.flatnonzero(action_mask) if action != 0))
    truck_id = action - 1

    previous_observation = _copy_observation(observation)
    previous_observation["earliness_to_window_start"][0][truck_id] = np.float32(0.0)
    previous_observation["lateness_to_window_start"][0][truck_id] = np.float32(0.0)
    rl_env._current_observation = previous_observation

    truck_positions = [truck.position.current_point.model_copy(deep=True) for truck in rl_env._current_env.trucks]
    truck_available_times = [0] * len(rl_env._current_env.trucks)
    monkeypatch.setattr(
        rl_env._simulator,
        "run",
        lambda selection, env: ([], truck_positions, truck_available_times)
    )

    _, reward, _, truncated, info = rl_env.step(action)

    assert truncated is False
    assert reward == pytest.approx(2.0)
    assert info["missed_requests_num"] == 0


def test_rl_env_step_runs_simulator_once(rl_env: SimulatorEnv):
    rl_env.reset(seed=42)
    original_run = rl_env._simulator.run
    calls_num = 0

    def counting_run(*args, **kwargs):
        nonlocal calls_num
        calls_num += 1
        return original_run(*args, **kwargs)

    rl_env._simulator.run = counting_run
    try:
        action_mask = rl_env.action_masks()
        action = int(np.flatnonzero(action_mask)[0])
        observation, reward, terminated, truncated, info = rl_env.step(action)
    finally:
        rl_env._simulator.run = original_run

    assert calls_num == 1
    _assert_observation_is_valid(rl_env, observation)
    _assert_reward_is_valid(reward)
    assert truncated is False
    assert len(info["current_selection"]) == 1


def test_rl_env_supports_observation_ablation_preset():
    generator = InputDataGenerator(
        load_point_names=GENERATOR_SETTINGS.load_point_names,
        unload_point_names=GENERATOR_SETTINGS.unload_point_names,
        requests_num_min=GENERATOR_SETTINGS.min_requests_num,
        requests_num_max=GENERATOR_SETTINGS.max_requests_num,
        trucks_num=GENERATOR_SETTINGS.max_truck_num,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, '%d.%m.%Y'),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, '%d.%m.%Y'),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        load_to_load_distance_range=GENERATOR_SETTINGS.load_to_load_distance_range.model_dump(),
        unload_to_unload_distance_range=GENERATOR_SETTINGS.unload_to_unload_distance_range.model_dump(),
        load_to_unload_distance_range=GENERATOR_SETTINGS.load_to_unload_distance_range.model_dump(),
        seed=42,
    )
    feature_config = DEFAULT_OBSERVATION_FEATURES.model_copy(
        update={
            "use_current_selection": False,
            "use_unfinished_ratio": False,
            "use_executed_requests": False,
        }
    )
    env = SimulatorEnv(generator, feature_config)

    observation, info = env.reset(seed=42)

    assert "current_selection" not in observation
    assert "unfinished_ratio" not in observation
    assert "executed_requests" not in observation
    assert "current_selection" not in env.observation_space.spaces
    assert "unfinished_ratio" not in env.observation_space.spaces
    assert "executed_requests" not in env.observation_space.spaces
    assert env.observation_space.contains(observation)
    assert "unfinished_ratio" in info
    assert info["unfinished_ratio"] == pytest.approx(np.array([0.0], dtype=np.float32))


def test_terminal_reward_bonus_is_added_on_last_step_only(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = InputDataGenerator(
        load_point_names=GENERATOR_SETTINGS.load_point_names,
        unload_point_names=GENERATOR_SETTINGS.unload_point_names,
        requests_num_min=GENERATOR_SETTINGS.min_requests_num,
        requests_num_max=GENERATOR_SETTINGS.max_requests_num,
        trucks_num=GENERATOR_SETTINGS.max_truck_num,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, '%d.%m.%Y'),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, '%d.%m.%Y'),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        load_to_load_distance_range=GENERATOR_SETTINGS.load_to_load_distance_range.model_dump(),
        unload_to_unload_distance_range=GENERATOR_SETTINGS.unload_to_unload_distance_range.model_dump(),
        load_to_unload_distance_range=GENERATOR_SETTINGS.load_to_unload_distance_range.model_dump(),
        seed=42,
    )
    env = SimulatorEnv(generator, terminal_reward_multiplier=10.0)
    observation, _ = env.reset(seed=42)
    env._current_selection = [-1] * (env._current_env.requests_num - 1)
    action_mask = env.action_masks()
    action = int(next(action for action in np.flatnonzero(action_mask) if action != 0))
    truck_id = action - 1

    previous_observation = _copy_observation(observation)
    previous_observation["earliness_to_window_start"][0][truck_id] = np.float32(0.0)
    previous_observation["lateness_to_window_start"][0][truck_id] = np.float32(0.0)
    env._current_observation = previous_observation

    truck_positions = [truck.position.current_point.model_copy(deep=True) for truck in env._current_env.trucks]
    truck_available_times = [0] * len(env._current_env.trucks)
    monkeypatch.setattr(
        env._simulator,
        "run",
        lambda selection, cur_env: ([], truck_positions, truck_available_times),
    )

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    expected_served_ratio = 1.0 - (info["missed_requests_num"] / env._current_env.requests_num)
    assert reward == pytest.approx(2.0 + 10.0 * expected_served_ratio)


def test_request_simulation_waits_until_window_start(simulator: Simulator, environment: Environment) -> None:
    request = None
    truck = None
    travel_time = None
    for candidate_request in environment.requests:
        for candidate_truck in environment.trucks:
            candidate_truck = candidate_truck.model_copy(deep=True)
            candidate_travel_time = environment.route_manager.calculate_travel_time_to_point(
                truck=candidate_truck,
                with_cargo=False,
                request=candidate_request,
                departure_point=candidate_truck.position.current_point,
                destination_point=candidate_request.point_to_load,
            )
            if candidate_travel_time < candidate_request.point_to_load.date_start_window:
                request = candidate_request
                truck = candidate_truck
                travel_time = candidate_travel_time
                break
        if request is not None:
            break

    assert request is not None
    assert truck is not None
    assert travel_time is not None

    unload_time = environment.route_manager.calculate_travel_time_to_point(
        truck=truck,
        with_cargo=True,
        request=request,
        departure_point=request.point_to_load,
        destination_point=request.point_to_unload,
    ) + simulator._cargo_process(truck, request, is_loading_process=False)
    load_time = simulator._cargo_process(truck, request, is_loading_process=True)
    current_time = max(request.point_to_load.date_start_window - travel_time - 5, 0)

    completed, total_time = simulator._request_simulation(
        truck=truck,
        request=request,
        current_time=current_time,
    )

    waiting_time = request.point_to_load.date_start_window - (current_time + travel_time)
    assert completed is True
    assert waiting_time >= 0
    assert total_time == travel_time + waiting_time + load_time + unload_time


def test_request_simulation_rejects_arrival_after_window_end(simulator: Simulator, environment: Environment) -> None:
    request = environment.requests[0]
    truck = environment.trucks[0].model_copy(deep=True)

    travel_time = environment.route_manager.calculate_travel_time_to_point(
        truck=truck,
        with_cargo=False,
        request=request,
        departure_point=truck.position.current_point,
        destination_point=request.point_to_load,
    )
    current_time = request.point_to_load.date_end_window - travel_time + 1

    completed, total_time = simulator._request_simulation(
        truck=truck,
        request=request,
        current_time=current_time,
    )

    assert completed is False
    assert total_time == 0


def test_reset_with_same_seed_recreates_same_instance(rl_env: SimulatorEnv):
    first_observation, _ = rl_env.reset(seed=123)
    first_requests_num = rl_env._current_env.requests_num
    first_request_names = [request.info.name for request in rl_env._current_env.requests]
    first_time_windows = first_observation["time_windows"].copy()

    second_observation, _ = rl_env.reset(seed=123)

    assert rl_env._current_env.requests_num == first_requests_num
    assert [request.info.name for request in rl_env._current_env.requests] == first_request_names
    assert np.array_equal(second_observation["time_windows"], first_time_windows)


def test_fixed_instances_cycle_deterministically():
    generator = InputDataGenerator(
        load_point_names=GENERATOR_SETTINGS.load_point_names,
        unload_point_names=GENERATOR_SETTINGS.unload_point_names,
        requests_num_min=GENERATOR_SETTINGS.min_requests_num,
        requests_num_max=GENERATOR_SETTINGS.max_requests_num,
        trucks_num=GENERATOR_SETTINGS.max_truck_num,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, '%d.%m.%Y'),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, '%d.%m.%Y'),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        load_to_load_distance_range=GENERATOR_SETTINGS.load_to_load_distance_range.model_dump(),
        unload_to_unload_distance_range=GENERATOR_SETTINGS.unload_to_unload_distance_range.model_dump(),
        load_to_unload_distance_range=GENERATOR_SETTINGS.load_to_unload_distance_range.model_dump(),
        seed=7,
    )
    fixed_instances = generator.generate_many(2)
    env = SimulatorEnv(generator, fixed_instances=fixed_instances)

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


def test_rl_env_supports_pairwise_lookahead():
    generator = InputDataGenerator(
        load_point_names=GENERATOR_SETTINGS.load_point_names,
        unload_point_names=GENERATOR_SETTINGS.unload_point_names,
        requests_num_min=GENERATOR_SETTINGS.min_requests_num,
        requests_num_max=GENERATOR_SETTINGS.max_requests_num,
        trucks_num=GENERATOR_SETTINGS.max_truck_num,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, '%d.%m.%Y'),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, '%d.%m.%Y'),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        load_to_load_distance_range=GENERATOR_SETTINGS.load_to_load_distance_range.model_dump(),
        unload_to_unload_distance_range=GENERATOR_SETTINGS.unload_to_unload_distance_range.model_dump(),
        load_to_unload_distance_range=GENERATOR_SETTINGS.load_to_unload_distance_range.model_dump(),
        seed=42,
    )
    feature_config = DEFAULT_OBSERVATION_FEATURES.model_copy(
        update={"pairwise_lookahead_requests": 3}
    )
    env = SimulatorEnv(generator, feature_config)

    observation, _ = env.reset(seed=42)

    assert observation["travel_time_to_load"].shape == (3, GENERATOR_SETTINGS.max_truck_num)
    assert observation["travel_time_with_cargo_to_unload"].shape == (3, GENERATOR_SETTINGS.max_truck_num)
    assert observation["earliness_to_window_start"].shape == (3, GENERATOR_SETTINGS.max_truck_num)
    assert observation["lateness_to_window_start"].shape == (3, GENERATOR_SETTINGS.max_truck_num)
    assert env.observation_space.contains(observation)


# # Запускает случайные действия, поэтому чтобы он не упал нужно запускать с _apply_restrictions_to_selection
# def test_rl_env_check(rl_env: SimulatorEnv):
#     check_env(rl_env)
