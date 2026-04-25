import copy
import datetime
import json
import math
import os

import numpy as np


class InputDataGenerator:

    def __init__(
            self,
            load_point_names: list[str],
            unload_point_names: list[str],
            requests_num_min: int,
            requests_num_max: int,
            trucks_num: int,
            simulator_start_date: datetime.datetime,
            simulator_end_date: datetime.datetime,
            capacities_variants: list[int],
            min_distance: int,
            max_distance: int,
            seed: int | None = None,
            routes_file_path: str = 'input/routes.json',
    ):
        self._load_point_names= load_point_names
        self._unload_point_names = unload_point_names
        self._requests_num_min = requests_num_min
        assert(requests_num_min < requests_num_max)
        self._requests_num_max = requests_num_max
        self._simulator_start_date = simulator_start_date
        self._simulator_end_date = simulator_end_date
        self._trucks_num = trucks_num
        self._capacities_variants = capacities_variants
        self._max_simulator_duration_in_hours = (simulator_end_date - simulator_start_date).days * 24
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._rng = np.random.default_rng(seed)
        self._routes_file_path = routes_file_path
        self._routes_data = self._load_or_create_routes()

    def reseed(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    def _choice(self, values: list):
        return values[int(self._rng.integers(0, len(values)))]

    def _load_or_create_routes(self) -> list[dict]:
        if os.path.exists(self._routes_file_path):
            with open(self._routes_file_path, 'r') as f:
                return json.load(f)

        routes_data = self._generate_logical_routes()
        routes_dir = os.path.dirname(self._routes_file_path)
        if routes_dir:
            os.makedirs(routes_dir, exist_ok=True)
        with open(self._routes_file_path, 'w') as f:
            json.dump(routes_data, f, indent=2)
        return routes_data

    def _build_cluster_positions(
            self,
            point_names: list[str],
            center_x: float,
            center_y: float,
            radius_min: float,
            radius_max: float,
    ) -> dict[str, tuple[float, float]]:
        if len(point_names) == 0:
            return {}

        angle_offset = float(self._rng.uniform(0.0, 2.0 * math.pi))
        positions = {}
        for point_id, point_name in enumerate(point_names):
            angle = angle_offset + (2.0 * math.pi * point_id / len(point_names))
            radius = float(self._rng.uniform(radius_min, radius_max))
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions[point_name] = (x, y)
        return positions

    def _build_point_positions(self) -> dict[str, tuple[float, float]]:
        load_positions = self._build_cluster_positions(
            point_names=self._load_point_names,
            center_x=0.0,
            center_y=0.0,
            radius_min=2200.0,
            radius_max=2600.0,
        )

        unload_center_x = float(self._rng.uniform(
            max(self._min_distance + 4500.0, 8000.0),
            max(self._max_distance - 2500.0, self._min_distance + 5000.0),
        ))
        unload_positions = self._build_cluster_positions(
            point_names=self._unload_point_names,
            center_x=unload_center_x,
            center_y=float(self._rng.uniform(-1200.0, 1200.0)),
            radius_min=1200.0,
            radius_max=2600.0,
        )
        return load_positions | unload_positions

    @staticmethod
    def _build_route(point_from_name: str, point_to_name: str, distance: int) -> dict:
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [[1, 1], [2, 2]]
            },
            'properties': {
                'distance': distance,
                'points': [
                    {'name': point_from_name},
                    {'name': point_to_name}
                ]
            }
        }

    def _calculate_distance(
            self,
            point_from: tuple[float, float],
            point_to: tuple[float, float]
    ) -> int:
        euclidean_distance = math.dist(point_from, point_to)
        noisy_distance = euclidean_distance * float(self._rng.uniform(0.95, 1.05))
        return int(round(min(max(noisy_distance, self._min_distance), self._max_distance)))

    def _generate_logical_routes(self) -> list[dict]:
        point_positions = self._build_point_positions()
        point_names = self._load_point_names + self._unload_point_names

        routes = []
        for point_from_id, point_from_name in enumerate(point_names):
            for point_to_name in point_names[point_from_id + 1:]:
                distance = self._calculate_distance(
                    point_positions[point_from_name],
                    point_positions[point_to_name]
                )
                routes.append(self._build_route(point_from_name, point_to_name, distance))
        return routes

    def _get_date_window(
            self,
            time_gap_from_start_in_hours: int,
            duration_in_hours: int
    ) -> (datetime.datetime, datetime.datetime):
        window_start = self._simulator_start_date + datetime.timedelta(hours=time_gap_from_start_in_hours)
        window_end = window_start + datetime.timedelta(hours=duration_in_hours)
        window_end = min(window_end, self._simulator_end_date)
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

        requests_num = int(self._rng.integers(self._requests_num_min, self._requests_num_max + 1))
        requests = []
        for request_id in range(requests_num):
            new_request = copy.deepcopy(request_structure)
            new_request['info']['name'] = f'Request_{request_id}'
            new_request['point_to_load']['name'] = self._choice(self._load_point_names)
            window_start, window_end = self._get_date_window(
                time_gap_from_start_in_hours=int(self._rng.integers(0, self._max_simulator_duration_in_hours + 1)),
                duration_in_hours=24
            )
            new_request['point_to_load']['date_start_window'] = str(window_start)
            new_request['point_to_load']['date_end_window'] = str(window_end)
            new_request['point_to_unload']['name'] = self._choice(self._unload_point_names)
            new_request['fix_route'] = None
            new_request['volume'] = self._choice(self._capacities_variants)
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
            new_truck['position']['current_point']['name'] = self._choice(self._unload_point_names)
            new_truck['cargo_params']['capacity'] = self._choice(self._capacities_variants)
            new_truck['cargo_params']['loading_speed'] = 4000
            new_truck['cargo_params']['unloading_speed'] = 2000
            new_truck['moving_params']['speed_with_cargo'] = 10
            new_truck['moving_params']['speed_without_cargo'] = 12
            trucks.append(new_truck)
        return trucks

    def generate_routes(self):
        return copy.deepcopy(self._routes_data)

    def generate_many(self, count: int) -> list[tuple[dict, list[dict]]]:
        return [self.generate_all(None) for _ in range(count)]

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
