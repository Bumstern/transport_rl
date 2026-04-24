import numpy as np

from src.optimizer.settings import GENERATOR_SETTINGS, DEFAULT_OBSERVATION_FEATURES
from src.optimizer.utils.observation_builder import ObservationBuilder
from src.simulator.units.point import Point


def test_create_observation_contains_pairwise_features(obs_builder, environment, requests_constraints):
    observation = obs_builder.create_observation([], [])

    assert "travel_time_to_load" in observation
    assert "travel_time_with_cargo_to_unload" in observation
    assert "earliness_to_window_start" in observation
    assert "lateness_to_window_start" in observation
    assert observation["travel_time_to_load"].shape == (1, GENERATOR_SETTINGS.max_truck_num)
    assert observation["travel_time_with_cargo_to_unload"].shape == (1, GENERATOR_SETTINGS.max_truck_num)
    assert observation["earliness_to_window_start"].shape == (1, GENERATOR_SETTINGS.max_truck_num)
    assert observation["lateness_to_window_start"].shape == (1, GENERATOR_SETTINGS.max_truck_num)
    assert observation["travel_time_to_load"].dtype == np.float32
    assert observation["travel_time_with_cargo_to_unload"].dtype == np.float32
    assert observation["earliness_to_window_start"].dtype == np.float32
    assert observation["lateness_to_window_start"].dtype == np.float32


def test_forbidden_trucks_have_extreme_pairwise_values(obs_builder, environment, requests_constraints):
    current_request_id = next(
        request_id
        for request_id, allowed_trucks in enumerate(requests_constraints)
        if any(
            truck_id not in allowed_trucks
            for truck_id in range(len(environment.trucks))
        )
    )
    current_selection = [-1] * current_request_id

    forbidden_truck_ids = [
        truck_id for truck_id in range(len(environment.trucks))
        if truck_id not in requests_constraints[current_request_id]
    ]
    assert len(forbidden_truck_ids) > 0

    observation = obs_builder.create_observation([], current_selection)

    for truck_id in forbidden_truck_ids:
        assert observation["travel_time_to_load"][0][truck_id] == np.float32(1.0)
        assert observation["travel_time_with_cargo_to_unload"][0][truck_id] == np.float32(1.0)
        assert observation["earliness_to_window_start"][0][truck_id] == np.float32(0.0)
        assert observation["lateness_to_window_start"][0][truck_id] == np.float32(1.0)


def test_truck_available_time_shifts_from_earliness_to_lateness(obs_builder, requests_constraints):
    current_selection = []
    current_request_id = len(current_selection)
    allowed_truck_id = next(
        truck_id for truck_id in requests_constraints[current_request_id]
        if truck_id != -1
    )
    next_request = obs_builder._env.requests[current_request_id]

    truck_positions = [truck.position.current_point.model_copy(deep=True) for truck in obs_builder._env.trucks]
    truck_positions[allowed_truck_id] = Point(name=next_request.point_to_load.name)

    observation_without_delay = obs_builder.create_observation(
        missed_requests_ids=[],
        current_selection=current_selection,
        truck_positions=truck_positions,
        truck_available_times=[0] * len(obs_builder._env.trucks)
    )
    observation_with_delay = obs_builder.create_observation(
        missed_requests_ids=[],
        current_selection=current_selection,
        truck_positions=truck_positions,
        truck_available_times=[obs_builder._env.end_date] * len(obs_builder._env.trucks)
    )

    expected_earliness_without_delay = (
        next_request.point_to_load.date_start_window / obs_builder._env.end_date
    )
    assert observation_without_delay["travel_time_to_load"][0][allowed_truck_id] == np.float32(0.0)
    assert observation_without_delay["earliness_to_window_start"][0][allowed_truck_id] == np.float32(
        expected_earliness_without_delay
    )
    assert observation_without_delay["lateness_to_window_start"][0][allowed_truck_id] == np.float32(0.0)
    assert observation_with_delay["travel_time_to_load"][0][allowed_truck_id] == np.float32(0.0)
    expected_signed_slack_with_delay = max(expected_earliness_without_delay - 1.0, -1.0)
    assert observation_with_delay["earliness_to_window_start"][0][allowed_truck_id] == np.float32(
        max(expected_signed_slack_with_delay, 0.0)
    )
    assert observation_with_delay["lateness_to_window_start"][0][allowed_truck_id] == np.float32(
        max(-expected_signed_slack_with_delay, 0.0)
    )
    assert np.all(
        observation_with_delay["earliness_to_window_start"]
        <= observation_without_delay["earliness_to_window_start"]
    )


def test_truck_position_affects_travel_time_to_load(obs_builder, requests_constraints):
    current_selection = []
    current_request_id = len(current_selection)
    allowed_truck_id = next(
        truck_id for truck_id in requests_constraints[current_request_id]
        if truck_id != -1
    )

    next_request = obs_builder._env.requests[current_request_id]
    truck_positions_far = [truck.position.current_point.model_copy(deep=True) for truck in obs_builder._env.trucks]
    truck_positions_near = [truck.position.current_point.model_copy(deep=True) for truck in obs_builder._env.trucks]
    truck_positions_near[allowed_truck_id] = Point(name=next_request.point_to_load.name)

    observation_far = obs_builder.create_observation(
        missed_requests_ids=[],
        current_selection=current_selection,
        truck_positions=truck_positions_far,
        truck_available_times=[0] * len(obs_builder._env.trucks)
    )
    observation_near = obs_builder.create_observation(
        missed_requests_ids=[],
        current_selection=current_selection,
        truck_positions=truck_positions_near,
        truck_available_times=[0] * len(obs_builder._env.trucks)
    )

    assert observation_near["travel_time_to_load"][0][allowed_truck_id] == np.float32(0.0)
    expected_earliness = (
        next_request.point_to_load.date_start_window / obs_builder._env.end_date
    )
    assert observation_near["earliness_to_window_start"][0][allowed_truck_id] == np.float32(expected_earliness)
    assert observation_near["lateness_to_window_start"][0][allowed_truck_id] == np.float32(0.0)
    assert (
        observation_near["travel_time_to_load"][0][allowed_truck_id]
        <= observation_far["travel_time_to_load"][0][allowed_truck_id]
    )


def test_request_route_affects_travel_time_with_cargo_to_unload(obs_builder, requests_constraints):
    current_selection = []
    current_request_id = len(current_selection)
    allowed_truck_id = next(
        truck_id for truck_id in requests_constraints[current_request_id]
        if truck_id != -1
    )

    next_request = obs_builder._env.requests[current_request_id]
    observation = obs_builder.create_observation([], current_selection)

    expected_travel_time = obs_builder._env.route_manager.calculate_travel_time_to_point(
        truck=obs_builder._env.trucks[allowed_truck_id],
        with_cargo=True,
        request=next_request,
        departure_point=next_request.point_to_load,
        destination_point=next_request.point_to_unload
    )
    expected_normalized_travel_time = min(expected_travel_time / obs_builder._env.end_date, 1.0)

    assert observation["travel_time_with_cargo_to_unload"][0][allowed_truck_id] == np.float32(
        expected_normalized_travel_time
    )


def test_pairwise_lookahead_pads_missing_future_requests(environment, requests_constraints):
    feature_config = DEFAULT_OBSERVATION_FEATURES.model_copy(
        update={"pairwise_lookahead_requests": 3}
    )
    lookahead_builder = ObservationBuilder(environment, requests_constraints, feature_config)
    current_selection = [-1] * (environment.requests_num - 1)

    observation = lookahead_builder.create_observation([], current_selection)

    assert observation["travel_time_to_load"].shape == (3, GENERATOR_SETTINGS.max_truck_num)
    assert np.all(observation["travel_time_to_load"][1:] == np.float32(1.0))
    assert np.all(observation["travel_time_with_cargo_to_unload"][1:] == np.float32(1.0))
    assert np.all(observation["earliness_to_window_start"][1:] == np.float32(0.0))
    assert np.all(observation["lateness_to_window_start"][1:] == np.float32(1.0))


def test_pairwise_lookahead_uses_real_future_request_values(environment, requests_constraints):
    feature_config = DEFAULT_OBSERVATION_FEATURES.model_copy(
        update={"pairwise_lookahead_requests": 3}
    )
    lookahead_builder = ObservationBuilder(environment, requests_constraints, feature_config)
    current_selection = []

    observation = lookahead_builder.create_observation([], current_selection)

    for lookahead_offset in range(3):
        request_id = lookahead_offset
        request = environment.requests[request_id]
        allowed_truck_id = next(
            truck_id for truck_id in requests_constraints[request_id]
            if truck_id != -1
        )

        expected_travel_time_to_load = environment.route_manager.calculate_travel_time_to_point(
            truck=environment.trucks[allowed_truck_id],
            with_cargo=False,
            request=request,
            departure_point=environment.trucks[allowed_truck_id].position.current_point,
            destination_point=request.point_to_load,
        )
        expected_travel_time_with_cargo_to_unload = environment.route_manager.calculate_travel_time_to_point(
            truck=environment.trucks[allowed_truck_id],
            with_cargo=True,
            request=request,
            departure_point=request.point_to_load,
            destination_point=request.point_to_unload,
        )
        expected_signed_slack = request.point_to_load.date_start_window - expected_travel_time_to_load

        assert observation["travel_time_to_load"][lookahead_offset][allowed_truck_id] == np.float32(
            min(expected_travel_time_to_load / environment.end_date, 1.0)
        )
        assert observation["travel_time_with_cargo_to_unload"][lookahead_offset][allowed_truck_id] == np.float32(
            min(expected_travel_time_with_cargo_to_unload / environment.end_date, 1.0)
        )
        assert observation["earliness_to_window_start"][lookahead_offset][allowed_truck_id] == np.float32(
            min(max(expected_signed_slack / environment.end_date, 0.0), 1.0)
        )
        assert observation["lateness_to_window_start"][lookahead_offset][allowed_truck_id] == np.float32(
            min(max(-expected_signed_slack / environment.end_date, 0.0), 1.0)
        )

        assert observation["travel_time_to_load"][lookahead_offset][allowed_truck_id] != np.float32(1.0)
