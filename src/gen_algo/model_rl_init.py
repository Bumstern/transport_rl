import json
from pathlib import Path

from gymnasium.core import ActType
from sb3_contrib import MaskablePPO

from src.gen_algo.simple_model import GeneticAlgoSimple, Genome
from src.optimizer.settings import DEFAULT_OBSERVATION_FEATURES, ObservationFeatureConfig
from src.optimizer.utils.observation_builder import ObservationBuilder
from src.simulator.environment import Environment
from src.simulator.model.simulator import Simulator


class GeneticAlgoWithRLInit(GeneticAlgoSimple):
    @staticmethod
    def _load_observation_feature_config(model_path: str | Path) -> ObservationFeatureConfig:
        config_path = Path(model_path).with_suffix(".config.json")
        if not config_path.exists():
            print("Конфиг файл не был найден. Используем по умолчанию")
            return DEFAULT_OBSERVATION_FEATURES

        raw_config = json.loads(config_path.read_text())
        config_payload = raw_config.get("config", raw_config)
        observation_feature_config = config_payload.get("observation_feature_config")
        if observation_feature_config is None:
            return DEFAULT_OBSERVATION_FEATURES
        return ObservationFeatureConfig(**observation_feature_config)

    @classmethod
    def from_model_path(
            cls,
            simulator: Simulator,
            environment: Environment,
            model_path: str | Path,
            requests_constrains: list[list[int]],
            popul_size: int = 100,
            mutation_rate: float = 0.1,
            retain_rate: float = 0.2
    ) -> "GeneticAlgoWithRLInit":
        observation_feature_config = cls._load_observation_feature_config(model_path)
        rl_model = MaskablePPO.load(str(model_path))
        obs_builder = ObservationBuilder(
            environment,
            requests_constrains,
            observation_feature_config,
        )
        return cls(
            simulator=simulator,
            rl_model=rl_model,
            obs_builder=obs_builder,
            requests_constrains=requests_constrains,
            popul_size=popul_size,
            mutation_rate=mutation_rate,
            retain_rate=retain_rate,
        )

    def __init__(
            self,
            simulator: Simulator,
            rl_model: MaskablePPO,
            obs_builder: ObservationBuilder,
            requests_constrains: list[list[int]],
            popul_size: int = 100,
            mutation_rate: float = 0.1,
            retain_rate: float = 0.2
    ):
        super().__init__(
            simulator=simulator,
            requests_constrains=requests_constrains,
            popul_size=popul_size,
            mutation_rate=mutation_rate,
            retain_rate=retain_rate
        )
        self._rl_model = rl_model
        self._obs_builder = obs_builder

    def _action_to_truck_id(self, action: ActType):
        # Переводим [0, max_truck_num + 1] вычитанием 1 -> [-1, max_truck_id]
        return action - 1

    def _predict_truck_id_with_rl_model(self, request_id: int, current_selection: list[int], missed_requests_ids: list[int]):
        obs = self._obs_builder.create_observation(missed_requests_ids=missed_requests_ids, current_selection=current_selection)
        mask = self._obs_builder.create_action_mask(request_id)

        action, _ = self._rl_model.predict(obs, action_masks=mask, deterministic=True)
        return self._action_to_truck_id(action)

    def _create_genome_with_rl(self, with_simulation: bool = False) -> Genome:
        current_selection = []
        missed_requests_ids = []
        while True:
            if len(current_selection) >= self._genome_length:
                break

            obs = self._obs_builder.create_observation(missed_requests_ids=missed_requests_ids, current_selection=current_selection)
            mask = self._obs_builder.create_action_mask(current_request_id=len(current_selection))
            # Ставим deterministic в False, чтобы популяция была более разнообразной
            action, _ = self._rl_model.predict(obs, action_masks=mask, deterministic=False)
            truck_id = self._action_to_truck_id(action)
            current_selection.append(truck_id)

            if with_simulation:
                # Если нужно для каждой хромосомы высчитывать пропущенные заявки
                # Возможно будут более хорошие предсказания, хоть и в ущерб скорости инициализации
                missed_requests_ids, _, _ = self._simulator.run(tuple(current_selection))
        return current_selection

    def _create_initial_population(self) -> list[Genome]:
        """Создает начальную популяцию из индексов машин с помощью RL."""
        return [self._create_genome_with_rl() for _ in range(self._popul_size)]
