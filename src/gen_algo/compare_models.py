import argparse
import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np

from src.gen_algo.model_rl_init import GeneticAlgoWithRLInit
from src.gen_algo.model_rl_mutator import GeneticAlgoWithRlMutator
from src.gen_algo.model_rl_mutator import GeneticAlgoWithRlTailMutator
from src.gen_algo.model_rl_mutator import GeneticAlgoWithInitAndRlMutator
from src.gen_algo.model_rl_mutator import GeneticAlgoWithInitAndRlTailMutator
from src.gen_algo.simple_model import GeneticAlgoSimple
from src.optimizer.settings import GENERATOR_SETTINGS
from src.simulator.builder import get_env, get_requests_constraints
from src.simulator.model.simulator import Simulator
from src.simulator.utils.data_generator.generator import InputDataGenerator

ALGORITHM_NAMES = (
    "ga",
    "ga_with_rl_init",
    "ga_with_rl_mutator",
    "ga_with_rl_tail_mutator",
    "ga_with_rl_init_and_mutator",
    "ga_with_rl_init_and_tail_mutator",
)


@dataclass(frozen=True)
class ComparisonConfig:
    model_path: Path
    output_path: Path | None
    test_instances: int
    test_seed: int
    ga_iterations: int
    population_size: int
    mutation_rate: float
    retain_rate: float
    max_workers: int


@dataclass(frozen=True)
class AlgorithmRunResult:
    algorithm: str
    instance_id: int
    served_requests: int
    missed_requests: int
    missed_ratio: float
    fitness: int


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


def build_fixed_test_instances(count: int, seed: int) -> list[tuple[dict, list[dict]]]:
    return build_generator(seed=seed).generate_many(count)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_algorithm(
    algorithm: str,
    simulator: Simulator,
    environment,
    requests_constraints: list[list[int]],
    model_path: Path,
    population_size: int,
    mutation_rate: float,
    retain_rate: float,
):
    if algorithm == "ga":
        return GeneticAlgoSimple(
            simulator=simulator,
            requests_constrains=requests_constraints,
            popul_size=population_size,
            mutation_rate=mutation_rate,
            retain_rate=retain_rate,
        )
    if algorithm == "ga_with_rl_init":
        return GeneticAlgoWithRLInit.from_model_path(
            simulator=simulator,
            environment=environment,
            model_path=model_path,
            requests_constrains=requests_constraints,
            popul_size=population_size,
            mutation_rate=mutation_rate,
            retain_rate=retain_rate,
        )
    if algorithm == "ga_with_rl_mutator":
        return GeneticAlgoWithRlMutator.from_model_path(
            simulator=simulator,
            environment=environment,
            model_path=model_path,
            requests_constrains=requests_constraints,
            popul_size=population_size,
            mutation_rate=mutation_rate,
            retain_rate=retain_rate,
        )
    if algorithm == "ga_with_rl_tail_mutator":
        return GeneticAlgoWithRlTailMutator.from_model_path(
            simulator=simulator,
            environment=environment,
            model_path=model_path,
            requests_constrains=requests_constraints,
            popul_size=population_size,
            mutation_rate=mutation_rate,
            retain_rate=retain_rate,
        )
    if algorithm == "ga_with_rl_init_and_mutator":
        return GeneticAlgoWithInitAndRlMutator.from_model_path(
            simulator=simulator,
            environment=environment,
            model_path=model_path,
            requests_constrains=requests_constraints,
            popul_size=population_size,
            mutation_rate=mutation_rate,
            retain_rate=retain_rate,
        )
    if algorithm == "ga_with_rl_init_and_tail_mutator":
        return GeneticAlgoWithInitAndRlTailMutator.from_model_path(
            simulator=simulator,
            environment=environment,
            model_path=model_path,
            requests_constrains=requests_constraints,
            popul_size=population_size,
            mutation_rate=mutation_rate,
            retain_rate=retain_rate,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_single_algorithm(
    algorithm: str,
    instance_id: int,
    input_data: dict,
    routes_data: list[dict],
    model_path: Path,
    ga_iterations: int,
    population_size: int,
    mutation_rate: float,
    retain_rate: float,
    seed: int,
) -> AlgorithmRunResult:
    _seed_everything(seed)
    print(f"Запустили {instance_id} для {algorithm}")

    environment = get_env(input_data, routes_data)
    requests_constraints = get_requests_constraints(environment, with_missed=True)
    simulator = Simulator(environment)

    ga = _build_algorithm(
        algorithm=algorithm,
        simulator=simulator,
        environment=environment,
        requests_constraints=requests_constraints,
        model_path=model_path,
        population_size=population_size,
        mutation_rate=mutation_rate,
        retain_rate=retain_rate,
    )
    best_genome = ga.fit(iterations=ga_iterations)
    missed_requests, _, _ = simulator.run(tuple(best_genome))
    served_requests = environment.requests_num - len(missed_requests)

    return AlgorithmRunResult(
        algorithm=algorithm,
        instance_id=instance_id,
        served_requests=served_requests,
        missed_requests=len(missed_requests),
        missed_ratio=(len(missed_requests) / environment.requests_num),
        fitness=served_requests,
    )


def evaluate_algorithms(config: ComparisonConfig) -> list[AlgorithmRunResult]:
    fixed_instances = build_fixed_test_instances(config.test_instances, config.test_seed)

    jobs = []
    for instance_id, (input_data, routes_data) in enumerate(fixed_instances):
        for algorithm_id, algorithm in enumerate(ALGORITHM_NAMES):
            jobs.append(
                (
                    algorithm,
                    instance_id,
                    input_data,
                    routes_data,
                    config.model_path,
                    config.ga_iterations,
                    config.population_size,
                    config.mutation_rate,
                    config.retain_rate,
                    config.test_seed + instance_id * 100 + algorithm_id,
                )
            )

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [executor.submit(run_single_algorithm, *job) for job in jobs]
        results = [future.result() for future in futures]

    results.sort(key=lambda result: (result.instance_id, result.algorithm))
    return results


def _format_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[col_id]) for row in rows) for col_id in range(len(rows[0]))]
    return "\n".join(
        " | ".join(value.ljust(widths[col_id]) for col_id, value in enumerate(row))
        for row in rows
    )


def print_results(results: list[AlgorithmRunResult]) -> None:
    header = ["instance", "algorithm", "served", "missed", "missed_ratio", "fitness"]
    rows = [header]
    for result in results:
        rows.append(
            [
                str(result.instance_id),
                result.algorithm,
                str(result.served_requests),
                str(result.missed_requests),
                f"{result.missed_ratio:.4f}",
                str(result.fitness),
            ]
        )
    print("Per-instance results:")
    print(_format_table(rows))

    summary_rows = [["algorithm", "avg_served", "avg_missed", "avg_missed_ratio", "avg_fitness"]]
    for algorithm in ALGORITHM_NAMES:
        algorithm_results = [result for result in results if result.algorithm == algorithm]
        summary_rows.append(
            [
                algorithm,
                f"{mean(result.served_requests for result in algorithm_results):.2f}",
                f"{mean(result.missed_requests for result in algorithm_results):.2f}",
                f"{mean(result.missed_ratio for result in algorithm_results):.4f}",
                f"{mean(result.fitness for result in algorithm_results):.2f}",
            ]
        )
    print("\nSummary:")
    print(_format_table(summary_rows))


def build_summary(results: list[AlgorithmRunResult]) -> list[dict]:
    summary = []
    for algorithm in ALGORITHM_NAMES:
        algorithm_results = [result for result in results if result.algorithm == algorithm]
        summary.append(
            {
                "algorithm": algorithm,
                "avg_served": mean(result.served_requests for result in algorithm_results),
                "avg_missed": mean(result.missed_requests for result in algorithm_results),
                "avg_missed_ratio": mean(result.missed_ratio for result in algorithm_results),
                "avg_fitness": mean(result.fitness for result in algorithm_results),
            }
        )
    return summary


def save_results(output_path: Path, results: list[AlgorithmRunResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(results)
    suffix = output_path.suffix.lower()

    if suffix == ".json":
        payload = {
            "results": [asdict(result) for result in results],
            "summary": summary,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if suffix == ".csv":
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["section", "instance_id", "algorithm", "served_requests", "missed_requests", "missed_ratio", "fitness"],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "section": "result",
                        "instance_id": result.instance_id,
                        "algorithm": result.algorithm,
                        "served_requests": result.served_requests,
                        "missed_requests": result.missed_requests,
                        "missed_ratio": result.missed_ratio,
                        "fitness": result.fitness,
                    }
                )
            for row in summary:
                writer.writerow(
                    {
                        "section": "summary",
                        "instance_id": "",
                        "algorithm": row["algorithm"],
                        "served_requests": row["avg_served"],
                        "missed_requests": row["avg_missed"],
                        "missed_ratio": row["avg_missed_ratio"],
                        "fitness": row["avg_fitness"],
                    }
                )
        return

    raise ValueError("output_path must end with .json or .csv")


def parse_args() -> ComparisonConfig:
    parser = argparse.ArgumentParser(description="Compare GA variants on the same fixed test instances.")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the trained RL model used by the RL-assisted GA variants.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default="output/compare_result.json",
        help="Optional path to save results as .json or .csv.",
    )
    parser.add_argument(
        "--test-instances",
        type=int,
        default=5,
        help="Number of fixed test instances to evaluate on.",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=42,
        help="Seed used to generate the fixed set of test instances.",
    )
    parser.add_argument(
        "--ga-iterations",
        type=int,
        default=20,
        help="Number of evolutionary iterations per algorithm run.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="Population size for all GA variants.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="Mutation rate for all GA variants.",
    )
    parser.add_argument(
        "--retain-rate",
        type=float,
        default=0.2,
        help="Selection retain rate for all GA variants.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=len(ALGORITHM_NAMES),
        help="Maximum number of parallel worker processes.",
    )
    args = parser.parse_args()

    return ComparisonConfig(
        model_path=args.model_path,
        output_path=args.output_path,
        test_instances=args.test_instances,
        test_seed=args.test_seed,
        ga_iterations=args.ga_iterations,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        retain_rate=args.retain_rate,
        max_workers=args.max_workers,
    )


def main() -> None:
    config = parse_args()
    results = evaluate_algorithms(config)
    print_results(results)
    if config.output_path is not None:
        save_results(config.output_path, results)


if __name__ == "__main__":
    main()
