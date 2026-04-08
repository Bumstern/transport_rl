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
    list_with_missed = simulator.run([random.choice(req_constr) for req_constr in requests_constraints])
    assert isinstance(list_with_missed, list)

    # Тестируем, что запуск вне ограничения выкинет ошибку
    all_trucks_ids = np.arange(start=0, stop=len(environment.trucks), step=1, dtype=int)
    with pytest.raises(AssertionError):
        list_with_missed = simulator.run([random.choice(list(set(all_trucks_ids).symmetric_difference(set(req_constr))))
                                          for req_constr in requests_constraints])


def test_rl_env_check(rl_env: SimulatorEnv):
    check_env(rl_env)



