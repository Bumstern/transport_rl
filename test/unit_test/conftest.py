import json

import pytest
from sb3_contrib import MaskablePPO

from src.optimizer.utils.observation_builder import ObservationBuilder
from src.simulator.builder import get_env, get_requests_constraints
from src.simulator.environment import Environment
from src.simulator.model.simulator import Simulator


@pytest.fixture(scope='module')
def environment():
    input_file_path: str = 'output/input.json'
    route_file_path: str = 'output/routes.json'

    with open(input_file_path, 'r') as f:
        input_data = json.load(f)

    with open(route_file_path, 'r') as f:
        routes_data = json.load(f)

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
