import copy
import datetime
import json
import os
import random


class InputDataGenerator:

    def __init__(
            self,
            load_point_names: list[str],
            unload_point_names: list[str],
            requests_num: int,
            trucks_num: int,
            simulator_start_date: datetime.datetime,
            simulator_end_date: datetime.datetime,
            capacities_variants: list[int],
            min_distance: int,
            max_distance: int
    ):
        self._load_point_names= load_point_names
        self._unload_point_names = unload_point_names
        self._requests_num = requests_num
        self._simulator_start_date = simulator_start_date
        self._simulator_end_date = simulator_end_date
        self._trucks_num = trucks_num
        self._capacities_variants = capacities_variants
        self._max_simulator_duration_in_hours = (simulator_end_date - simulator_start_date).days * 24
        self._min_distance = min_distance
        self._max_distance = max_distance

    def _get_date_window(
            self,
            time_gap_from_start_in_hours: int,
            duration_in_hours: int
    ) -> (datetime.datetime, datetime.datetime):
        window_start = self._simulator_start_date + datetime.timedelta(hours=time_gap_from_start_in_hours)
        window_end = window_start + datetime.timedelta(hours=duration_in_hours)
        return window_start, window_end

    def generate_requests(self) -> list[dict]:
        request_structure = {
            'info': {
                'name': ''
            },
            'point_to_load': {
                'name': '',
                'date_start_window': 0,
                'date_end_window': 0
            },
            'point_to_unload': {
                'name': ''
            },
            'fix_route': '',
            'volume': 0
        }

        requests = []
        for request_id in range(self._requests_num):
            new_request = copy.deepcopy(request_structure)
            new_request['info']['name'] = f'Request_{request_id}'
            new_request['point_to_load']['name'] = random.choice(self._load_point_names)
            window_start, window_end = self._get_date_window(
                time_gap_from_start_in_hours=random.randint(0, self._max_simulator_duration_in_hours),
                duration_in_hours=random.randint(5, 12)
            )
            new_request['point_to_load']['date_start_window'] = str(window_start)
            new_request['point_to_load']['date_end_window'] = str(window_end)
            new_request['point_to_unload']['name'] = random.choice(self._unload_point_names)
            new_request['fix_route'] = None
            new_request['volume'] = random.choice(self._capacities_variants)
            requests.append(new_request)
        return requests

    def generate_trucks(self) -> list[dict]:
        truck_structure = {
            'info': {
                'name': ''
            },
            'position': {
                'current_point': {
                    'name': ''
                }
            },
            'cargo_params': {
                'capacity': 0,
                'loading_speed': 0,
                'unloading_speed': 0
            },
            'moving_params': {
                'speed_with_cargo': 0,
                'speed_without_cargo': 0
            }
        }

        trucks = []
        for truck_id in range(self._trucks_num):
            new_truck = copy.deepcopy(truck_structure)
            new_truck['info']['name'] =  f'Truck_{truck_id}'
            new_truck['position']['current_point']['name'] = random.choice(self._unload_point_names)
            new_truck['cargo_params']['capacity'] = random.choice(self._capacities_variants)
            new_truck['cargo_params']['loading_speed'] = 10
            new_truck['cargo_params']['unloading_speed'] = 10
            new_truck['moving_params']['speed_with_cargo'] = 10
            new_truck['moving_params']['speed_without_cargo'] = 12
            trucks.append(new_truck)
        return trucks

    def generate_routes(self):
        route_structure = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [[1,1], [2, 2]]
            },
            'properties': {
                'distance': 0,
                'points': [
                    {'name': ''},
                    {'name': ''}
                ]
            }
        }

        routes = []
        for load_point_name in self._load_point_names:
            for unload_point_name in self._unload_point_names:
                new_route = copy.deepcopy(route_structure)
                new_route['properties']['points'][0]['name'] = load_point_name
                new_route['properties']['points'][1]['name'] = unload_point_name
                new_route['properties']['distance'] = random.randint(self._min_distance, self._max_distance)
                routes.append(new_route)
        return routes

    def generate_all(self, dir_path: str | None) -> (dict, list[dict]):
        generated_file = {}
        generated_file['time'] = {
            'simulator_start_date': str(self._simulator_start_date),
            'simulator_end_date': str(self._simulator_end_date)
        }
        generated_file['trucks'] = self.generate_trucks()
        generated_file['requests'] = self.generate_requests()

        if dir_path is not None:
            with open(os.path.join(dir_path, 'input.json'), 'w') as f:
                json.dump(generated_file, f, default=str)

        routes_data = self.generate_routes()

        if dir_path is not None:
            with open(os.path.join(dir_path, 'routes.json'), 'w') as f:
                json.dump(routes_data, f, default=str)
        
        return generated_file, routes_data
