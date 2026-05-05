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
        load_to_load_distance_range=GENERATOR_SETTINGS.load_to_load_distance_range.model_dump(),
        unload_to_unload_distance_range=GENERATOR_SETTINGS.unload_to_unload_distance_range.model_dump(),
        load_to_unload_distance_range=GENERATOR_SETTINGS.load_to_unload_distance_range.model_dump(),
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


def test_generator_uses_separate_distance_ranges_between_port_types(tmp_path) -> None:
    routes_file_path = tmp_path / "routes.json"
    generator = InputDataGenerator(
        load_point_names=["L1", "L2"],
        unload_point_names=["U1", "U2"],
        requests_num_min=1,
        requests_num_max=2,
        trucks_num=1,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, '%d.%m.%Y'),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, '%d.%m.%Y'),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        load_to_load_distance_range={"min": 300, "max": 350},
        unload_to_unload_distance_range={"min": 400, "max": 450},
        load_to_unload_distance_range={"min": 900, "max": 950},
        seed=42,
        routes_file_path=str(routes_file_path),
    )

    routes_data = generator.generate_routes()
    distances_by_pair = {
        tuple(point["name"] for point in route["properties"]["points"]): route["properties"]["distance"]
        for route in routes_data
    }

    assert 300 <= distances_by_pair[("L1", "L2")] <= 350
    assert 400 <= distances_by_pair[("U1", "U2")] <= 450
    for point_from, point_to in [("L1", "U1"), ("L1", "U2"), ("L2", "U1"), ("L2", "U2")]:
        assert 900 <= distances_by_pair[(point_from, point_to)] <= 950
