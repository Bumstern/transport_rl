import json

from simulator.environment import Environment
from simulator.model.simulator import Simulator
from simulator.units.route import Route
from simulator.utils.time import Time


class Builder:

    def __init__(self, input_path: str, routes_path: str):
        input_dict_data = self.__open_file(input_path)
        routes_dict_data = self.__open_file(routes_path)
        self._env = self.__init_env(input_dict_data, routes_dict_data)
        self._simulator = self.__init_simulator()

    def __open_file(self, file_path: str):
        with open(file_path, 'r') as f:
            file_data = json.load(f)
        return file_data

    def __init_env(self, input_data: dict, routes_data: dict):
        # Переводим все даты в периоды
        time = Time(input_data['time']['simulator_start_date'])
        input_data = time.transition_to_periods(input_data)
        # routes_data = time.transition_to_periods(routes_data)

        env_data = {
            'route_manager': routes_data,
            'trucks': input_data['trucks'],
            'requests': input_data['requests']
        }
        env = Environment(**env_data)
        return env

    def __init_simulator(self) -> Simulator:
        simulator = Simulator(self._env)
        return simulator

    def run_selection(self, selection: list[int]):
        count_of_missed_requests = self._simulator.run(selection)
        print('Count of missed requests:', count_of_missed_requests)
