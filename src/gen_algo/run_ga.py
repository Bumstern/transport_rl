import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from src.gen_algo.simple_model import GeneticAlgoSimple
from src.optimizer.settings import GENERATOR_SETTINGS
from src.simulator.builder import get_env, get_requests_constraints
from src.simulator.model.simulator import Simulator
from src.simulator.utils.data_generator.generator import InputDataGenerator


def build_generator(seed: int | None = None) -> InputDataGenerator:
    return InputDataGenerator(
        load_point_names=GENERATOR_SETTINGS.load_point_names,
        unload_point_names=GENERATOR_SETTINGS.unload_point_names,
        requests_num_min=GENERATOR_SETTINGS.min_requests_num,
        requests_num_max=GENERATOR_SETTINGS.max_requests_num,
        trucks_num=GENERATOR_SETTINGS.max_truck_num,
        simulator_start_date=datetime.strptime(GENERATOR_SETTINGS.simulator_start_date, "%d.%m.%Y"),
        simulator_end_date=datetime.strptime(GENERATOR_SETTINGS.simulator_end_date, "%d.%m.%Y"),
        capacities_variants=GENERATOR_SETTINGS.capacities_variants,
        min_distance=GENERATOR_SETTINGS.min_distance,
        max_distance=GENERATOR_SETTINGS.max_distance,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the plain genetic algorithm on a single generated instance.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for generator and random GA operations.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of GA iterations.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="Population size.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="Per-gene mutation probability.",
    )
    parser.add_argument(
        "--retain-rate",
        type=float,
        default=0.2,
        help="Fraction of best individuals retained as parents.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to save the result as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    generator = build_generator(seed=args.seed)
    input_data, routes_data = generator.generate_all(None)
    environment = get_env(input_data, routes_data)
    simulator = Simulator(environment)
    requests_constraints = get_requests_constraints(environment, with_missed=True)

    ga = GeneticAlgoSimple(
        simulator=simulator,
        requests_constrains=requests_constraints,
        popul_size=args.population_size,
        mutation_rate=args.mutation_rate,
        retain_rate=args.retain_rate,
    )
    best_genome = ga.fit(iterations=args.iterations)
    missed_requests_ids, truck_positions, truck_available_times = simulator.run(tuple(best_genome))
    served_requests = environment.requests_num - len(missed_requests_ids)

    result = {
        "seed": args.seed,
        "iterations": args.iterations,
        "population_size": args.population_size,
        "mutation_rate": args.mutation_rate,
        "retain_rate": args.retain_rate,
        "requests_num": environment.requests_num,
        "served_requests": served_requests,
        "missed_requests": len(missed_requests_ids),
        "missed_requests_ids": missed_requests_ids,
        "best_genome": best_genome,
        "truck_positions": [point.name for point in truck_positions],
        "truck_available_times": truck_available_times,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
