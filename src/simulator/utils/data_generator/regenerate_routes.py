import argparse
from datetime import datetime
from pathlib import Path

from src.optimizer.settings import GENERATOR_SETTINGS
from src.simulator.utils.data_generator.generator import InputDataGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate routes.json from current generator settings.")
    parser.add_argument(
        "--routes-file-path",
        type=Path,
        default=Path("input/routes.json"),
        help="Path to the routes JSON file to recreate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for route generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.routes_file_path.exists():
        args.routes_file_path.unlink()

    generator = InputDataGenerator(
        load_point_names=GENERATOR_SETTINGS.load_point_names,
        unload_point_names=GENERATOR_SETTINGS.unload_point_names,
        requests_num_min=GENERATOR_SETTINGS.min_requests_num,
        requests_num_max=GENERATOR_SETTINGS.max_requests_num,
        trucks_num=GENERATOR_SETTINGS.max_truck_num,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, "%d.%m.%Y"),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, "%d.%m.%Y"),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        load_to_load_distance_range=GENERATOR_SETTINGS.load_to_load_distance_range.model_dump(),
        unload_to_unload_distance_range=GENERATOR_SETTINGS.unload_to_unload_distance_range.model_dump(),
        load_to_unload_distance_range=GENERATOR_SETTINGS.load_to_unload_distance_range.model_dump(),
        seed=args.seed,
        routes_file_path=str(args.routes_file_path),
    )
    routes = generator.generate_routes()
    distances = [route["properties"]["distance"] for route in routes]

    print(f"Regenerated routes: {args.routes_file_path}")
    print(f"Routes count: {len(routes)}")
    print(f"Distance range: {min(distances)}..{max(distances)}")


if __name__ == "__main__":
    main()
