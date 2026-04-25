import json
from pathlib import Path

from src.gen_algo.compare_models import AlgorithmRunResult
from src.gen_algo.compare_models import build_fixed_test_instances
from src.gen_algo.compare_models import build_summary
from src.gen_algo.compare_models import run_single_algorithm
from src.gen_algo.compare_models import save_results
from src.gen_algo.model_rl_init import GeneticAlgoWithRLInit
from src.gen_algo.model_rl_mutator import GeneticAlgoWithRlMutator
from src.gen_algo.model_rl_mutator import GeneticAlgoWithRlTailMutator
from src.gen_algo.model_rl_mutator import GeneticAlgoWithInitAndRlMutator
from src.gen_algo.model_rl_mutator import GeneticAlgoWithInitAndRlTailMutator


def test_gen_algo_from_model_path_loads_observation_config(
    simulator,
    environment,
    requests_constraints,
    monkeypatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "checkpoint.zip"
    model_path.write_text("stub")
    config_path = model_path.with_suffix(".config.json")
    config_path.write_text(
        json.dumps(
            {
                "config": {
                    "observation_feature_config": {
                        "use_time_windows": False,
                        "use_executed_requests": True,
                        "use_unfinished_ratio": False,
                        "use_current_selection": False,
                        "use_next_request_tw": True,
                        "use_pairwise_features": True,
                        "pairwise_lookahead_requests": 5,
                    }
                }
            }
        )
    )

    dummy_model = object()
    monkeypatch.setattr(
        "src.gen_algo.model_rl_init.MaskablePPO.load",
        lambda path: dummy_model,
    )

    ga = GeneticAlgoWithRLInit.from_model_path(
        simulator=simulator,
        environment=environment,
        model_path=model_path,
        requests_constrains=requests_constraints,
    )

    assert ga._rl_model is dummy_model
    assert ga._obs_builder._feature_config.use_time_windows is False
    assert ga._obs_builder._feature_config.use_current_selection is False
    assert ga._obs_builder._feature_config.pairwise_lookahead_requests == 5


def test_mutator_gen_algo_from_model_path_returns_subclass(
    simulator,
    environment,
    requests_constraints,
    monkeypatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "checkpoint.zip"
    model_path.write_text("stub")
    dummy_model = object()
    monkeypatch.setattr(
        "src.gen_algo.model_rl_init.MaskablePPO.load",
        lambda path: dummy_model,
    )

    ga = GeneticAlgoWithInitAndRlMutator.from_model_path(
        simulator=simulator,
        environment=environment,
        model_path=model_path,
        requests_constrains=requests_constraints,
    )

    assert isinstance(ga, GeneticAlgoWithInitAndRlMutator)
    assert ga._rl_model is dummy_model


def test_rl_mutator_only_gen_algo_from_model_path_returns_subclass(
    simulator,
    environment,
    requests_constraints,
    monkeypatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "checkpoint.zip"
    model_path.write_text("stub")
    dummy_model = object()
    monkeypatch.setattr(
        "src.gen_algo.model_rl_init.MaskablePPO.load",
        lambda path: dummy_model,
    )

    ga = GeneticAlgoWithRlMutator.from_model_path(
        simulator=simulator,
        environment=environment,
        model_path=model_path,
        requests_constrains=requests_constraints,
    )

    assert isinstance(ga, GeneticAlgoWithRlMutator)
    assert ga._rl_model is dummy_model


def test_rl_tail_mutator_gen_algo_from_model_path_returns_subclass(
    simulator,
    environment,
    requests_constraints,
    monkeypatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "checkpoint.zip"
    model_path.write_text("stub")
    dummy_model = object()
    monkeypatch.setattr(
        "src.gen_algo.model_rl_init.MaskablePPO.load",
        lambda path: dummy_model,
    )

    ga = GeneticAlgoWithRlTailMutator.from_model_path(
        simulator=simulator,
        environment=environment,
        model_path=model_path,
        requests_constrains=requests_constraints,
    )

    assert isinstance(ga, GeneticAlgoWithRlTailMutator)
    assert ga._rl_model is dummy_model


def test_init_and_rl_tail_mutator_gen_algo_from_model_path_returns_subclass(
    simulator,
    environment,
    requests_constraints,
    monkeypatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "checkpoint.zip"
    model_path.write_text("stub")
    dummy_model = object()
    monkeypatch.setattr(
        "src.gen_algo.model_rl_init.MaskablePPO.load",
        lambda path: dummy_model,
    )

    ga = GeneticAlgoWithInitAndRlTailMutator.from_model_path(
        simulator=simulator,
        environment=environment,
        model_path=model_path,
        requests_constrains=requests_constraints,
    )

    assert isinstance(ga, GeneticAlgoWithInitAndRlTailMutator)
    assert ga._rl_model is dummy_model


def test_rl_tail_mutator_rebuilds_tail_after_first_mutation(monkeypatch) -> None:
    class DummyObsBuilder:
        def __init__(self) -> None:
            self.selection_calls = []
            self.mask_calls = []

        def create_observation(self, missed_requests_ids, current_selection):
            self.selection_calls.append(list(current_selection))
            return {"selection_len": len(current_selection)}

        def create_action_mask(self, current_request_id):
            self.mask_calls.append(current_request_id)
            return [True, True, True]

    class DummyModel:
        def __init__(self) -> None:
            self.actions = iter([2, 3, 4, 1])

        def predict(self, obs, action_masks, deterministic):
            return next(self.actions), None

    ga = GeneticAlgoWithRlTailMutator(
        simulator=None,  # type: ignore[arg-type]
        rl_model=DummyModel(),
        obs_builder=DummyObsBuilder(),
        requests_constrains=[[-1, 0, 1]] * 4,
        popul_size=4,
        mutation_rate=1.0,
        retain_rate=0.5,
    )
    individual = [9, 9, 9, 9]

    monkeypatch.setattr("src.gen_algo.model_rl_mutator.random.random", lambda: 0.0)

    mutated = ga._mutation(individual)

    assert mutated == [1, 2, 3, 0]
    assert ga._obs_builder.selection_calls == [[], [1], [1, 2], [1, 2, 3]]
    assert ga._obs_builder.mask_calls == [0, 1, 2, 3]


def test_build_fixed_test_instances_is_deterministic() -> None:
    first_instances = build_fixed_test_instances(2, seed=123)
    second_instances = build_fixed_test_instances(2, seed=123)

    assert first_instances == second_instances


def test_run_single_algorithm_returns_metrics(monkeypatch, input_generator) -> None:
    input_data, routes_data = input_generator.generate_all(None)

    class DummyGA:
        def fit(self, iterations: int):
            return [-1] * len(input_data["requests"])

    monkeypatch.setattr(
        "src.gen_algo.compare_models._build_algorithm",
        lambda **kwargs: DummyGA(),
    )

    result = run_single_algorithm(
        algorithm="ga",
        instance_id=0,
        input_data=input_data,
        routes_data=routes_data,
        model_path=Path("output/models/2026-04-24_13-12-29.zip"),
        ga_iterations=3,
        population_size=10,
        mutation_rate=0.1,
        retain_rate=0.2,
        seed=42,
    )

    assert result.instance_id == 0
    assert result.algorithm == "ga"
    assert result.missed_requests == len(input_data["requests"])
    assert result.served_requests == 0
    assert result.fitness == 0


def test_save_results_writes_json_and_summary(tmp_path) -> None:
    output_path = tmp_path / "results.json"
    results = [
        AlgorithmRunResult("ga", 0, 10, 2, 10),
        AlgorithmRunResult("ga_with_rl_init", 0, 11, 1, 11),
        AlgorithmRunResult("ga_with_rl_mutator", 0, 11, 1, 11),
        AlgorithmRunResult("ga_with_rl_tail_mutator", 0, 11, 1, 11),
        AlgorithmRunResult("ga_with_rl_init_and_mutator", 0, 12, 0, 12),
        AlgorithmRunResult("ga_with_rl_init_and_tail_mutator", 0, 12, 0, 12),
    ]

    save_results(output_path, results)

    payload = json.loads(output_path.read_text())
    assert len(payload["results"]) == 6
    assert len(payload["summary"]) == 6
    assert payload["summary"][0]["algorithm"] == "ga"


def test_save_results_writes_csv(tmp_path) -> None:
    output_path = tmp_path / "results.csv"
    results = [
        AlgorithmRunResult("ga", 0, 10, 2, 10),
        AlgorithmRunResult("ga_with_rl_init", 0, 11, 1, 11),
        AlgorithmRunResult("ga_with_rl_mutator", 0, 11, 1, 11),
        AlgorithmRunResult("ga_with_rl_tail_mutator", 0, 11, 1, 11),
        AlgorithmRunResult("ga_with_rl_init_and_mutator", 0, 12, 0, 12),
        AlgorithmRunResult("ga_with_rl_init_and_tail_mutator", 0, 12, 0, 12),
    ]

    save_results(output_path, results)

    written = output_path.read_text()
    assert "section,instance_id,algorithm,served_requests,missed_requests,fitness" in written
    assert "result,0,ga,10,2,10" in written
    assert "summary,,ga," in written


def test_build_summary_includes_rl_mutator_variant() -> None:
    results = [
        AlgorithmRunResult("ga", 0, 10, 2, 10),
        AlgorithmRunResult("ga_with_rl_init", 0, 11, 1, 11),
        AlgorithmRunResult("ga_with_rl_mutator", 0, 12, 0, 12),
        AlgorithmRunResult("ga_with_rl_tail_mutator", 0, 8, 4, 8),
        AlgorithmRunResult("ga_with_rl_init_and_mutator", 0, 9, 3, 9),
        AlgorithmRunResult("ga_with_rl_init_and_tail_mutator", 0, 13, 1, 13),
    ]

    summary = build_summary(results)

    assert [row["algorithm"] for row in summary] == [
        "ga",
        "ga_with_rl_init",
        "ga_with_rl_mutator",
        "ga_with_rl_tail_mutator",
        "ga_with_rl_init_and_mutator",
        "ga_with_rl_init_and_tail_mutator",
    ]
