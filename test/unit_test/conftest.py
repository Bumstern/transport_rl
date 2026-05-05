from datetime import datetime

import pytest
from sb3_contrib import MaskablePPO

from src.optimizer.settings import GENERATOR_SETTINGS
from src.optimizer.utils.observation_builder import ObservationBuilder
from src.simulator.builder import get_env, get_requests_constraints
from src.simulator.environment import Environment
from src.simulator.model.simulator import Simulator
from src.simulator.utils.data_generator.generator import InputDataGenerator


@pytest.fixture(scope='module')
def input_generator() -> InputDataGenerator:
    return InputDataGenerator(
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


@pytest.fixture(scope='module')
def environment(input_generator: InputDataGenerator):
    input_data, routes_data = input_generator.generate_all(None)
    env: Environment = get_env(input_data, routes_data)
    return env


@pytest.fixture(scope='module')
def simulator(environment: Environment):
    sim = Simulator(environment)
    return sim


@pytest.fixture(scope='module')
def requests_constraints(environment: Environment) -> list[list[int]]:
    req_constrains = get_requests_constraints(environment, True)
    return req_constrains


@pytest.fixture(scope='module')
def obs_builder(environment: Environment, requests_constraints: list[list[int]]) -> ObservationBuilder:
    obb = ObservationBuilder(environment, requests_constraints)
    return obb


@pytest.fixture(scope='session')
def rl_model() -> MaskablePPO:
    rl_model_file_path = 'output/best/best_model.zip'
    model = MaskablePPO.load(rl_model_file_path)
    return model
