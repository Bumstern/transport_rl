import numpy as np

from src.optimizer.settings import GENERATOR_SETTINGS
from src.simulator.units.point import Point


def test_create_observation_contains_pairwise_features(obs_builder, environment, requests_constraints):
    observation = obs_builder.create_observation([], [])

    assert "travel_time_to_load" in observation
    assert "time_slack_to_window_start" in observation
    assert observation["travel_time_to_load"].shape == (GENERATOR_SETTINGS.max_truck_num,)
    assert observation["time_slack_to_window_start"].shape == (GENERATOR_SETTINGS.max_truck_num,)
    assert observation["travel_time_to_load"].dtype == np.float32
    assert observation["time_slack_to_window_start"].dtype == np.float32


def test_forbidden_trucks_have_extreme_pairwise_values(obs_builder, environment, requests_constraints):
    current_selection = []
    current_request_id = len(current_selection)

    forbidden_truck_ids = [
        truck_id for truck_id in range(len(environment.trucks))
        if truck_id not in requests_constraints[current_request_id]
    ]
    assert len(forbidden_truck_ids) > 0

    observation = obs_builder.create_observation([], current_selection)

    for truck_id in forbidden_truck_ids:
        assert observation["travel_time_to_load"][truck_id] == np.float32(1.0)
        assert observation["time_slack_to_window_start"][truck_id] == np.float32(-1.0)


def test_truck_available_time_decreases_time_slack(obs_builder, requests_constraints):
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

    expected_slack_without_delay = (
        next_request.point_to_load.date_start_window / obs_builder._env.end_date
    )
    assert observation_without_delay["travel_time_to_load"][allowed_truck_id] == np.float32(0.0)
    assert observation_without_delay["time_slack_to_window_start"][allowed_truck_id] == np.float32(
        expected_slack_without_delay
    )
    assert observation_with_delay["travel_time_to_load"][allowed_truck_id] == np.float32(0.0)
    assert observation_with_delay["time_slack_to_window_start"][allowed_truck_id] == np.float32(
        max(expected_slack_without_delay - 1.0, -1.0)
    )
    assert np.all(
        observation_with_delay["time_slack_to_window_start"]
        <= observation_without_delay["time_slack_to_window_start"]
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

    assert observation_near["travel_time_to_load"][allowed_truck_id] == np.float32(0.0)
    expected_slack = (
        next_request.point_to_load.date_start_window / obs_builder._env.end_date
    )
    assert observation_near["time_slack_to_window_start"][allowed_truck_id] == np.float32(expected_slack)
    assert (
        observation_near["travel_time_to_load"][allowed_truck_id]
        <= observation_far["travel_time_to_load"][allowed_truck_id]
    )
