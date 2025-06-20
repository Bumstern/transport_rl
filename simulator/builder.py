import datetime
import json

from simulator.environment import Environment
from simulator.units.requirement import apply_requirements
from simulator.utils.time import Time


def __open_file(file_path: str):
    with open(file_path, 'r') as f:
        file_data = json.load(f)
    return file_data


def get_env(input_data: dict, routes_data: dict) -> Environment:
    # Переводим все даты в периоды
    time = Time(input_data['time']['simulator_start_date'], input_data['time']['simulator_end_date'])
    input_data = time.transition_to_periods(input_data)
    # routes_data = time.transition_to_periods(routes_data)

    env_data = {
        'end_date': time.end_period,
        'route_manager': routes_data,
        'trucks': input_data['trucks'],
        'requests': input_data['requests']
    }
    env = Environment(**env_data)
    return env


def get_requests_constrains(env: Environment, with_missed: bool) -> list[list[int]]:
    requests_constrains = apply_requirements(env.requests, env.trucks, with_missed=with_missed)
    return requests_constrains
