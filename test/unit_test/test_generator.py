import json
from datetime import datetime

from src.optimizer.settings import GENERATOR_SETTINGS
from src.simulator.utils.data_generator.generator import InputDataGenerator


def _build_generator(routes_file_path: str, seed: int | None) -> InputDataGenerator:
    return InputDataGenerator(
        load_point_names=GENERATOR_SETTINGS.load_point_names,
        unload_point_names=GENERATOR_SETTINGS.unload_point_names,
        requests_num_min=GENERATOR_SETTINGS.min_requests_num,
        requests_num_max=GENERATOR_SETTINGS.max_requests_num,
        trucks_num=GENERATOR_SETTINGS.max_truck_num,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, '%d.%m.%Y'),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, '%d.%m.%Y'),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        min_distance=GENERATOR_SETTINGS.min_distance,
        max_distance=GENERATOR_SETTINGS.max_distance,
        seed=seed,
        routes_file_path=routes_file_path,
    )


def test_generator_creates_routes_file_when_it_does_not_exist(tmp_path) -> None:
    routes_file_path = tmp_path / "routes.json"

    generator = _build_generator(str(routes_file_path), seed=42)
    routes_data = generator.generate_routes()

    assert routes_file_path.exists()
    with routes_file_path.open("r") as f:
        saved_routes = json.load(f)

    assert saved_routes == routes_data


def test_generator_uses_existing_routes_file_without_recreating_it(tmp_path) -> None:
    routes_file_path = tmp_path / "routes.json"
    existing_routes = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[1, 1], [2, 2]],
            },
            "properties": {
                "distance": 7777,
                "points": [
                    {"name": GENERATOR_SETTINGS.load_point_names[0]},
                    {"name": GENERATOR_SETTINGS.unload_point_names[0]},
                ],
            },
        }
    ]
    routes_file_path.write_text(json.dumps(existing_routes, indent=2))

    generator = _build_generator(str(routes_file_path), seed=999)
    routes_data = generator.generate_routes()

    assert routes_data == existing_routes
    with routes_file_path.open("r") as f:
        saved_routes = json.load(f)
    assert saved_routes == existing_routes
