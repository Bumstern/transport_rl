import copy
from typing import Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, ActType

from src.optimizer.utils.observation_builder import ObservationBuilder
from src.simulator.builder import get_env, get_requests_constraints
from src.simulator.environment import Environment
from src.simulator.model.simulator import Simulator
from src.optimizer.settings import (
    GENERATOR_SETTINGS,
    DEFAULT_OBSERVATION_FEATURES,
    ObservationFeatureConfig,
)
from src.simulator.utils.data_generator.generator import InputDataGenerator


class SimulatorEnv(gymnasium.Env):
    def __init__(
            self,
            input_generator: InputDataGenerator,
            observation_feature_config: ObservationFeatureConfig = DEFAULT_OBSERVATION_FEATURES,
            fixed_instances: list[tuple[dict, list[dict]]] | None = None,
            terminal_reward_multiplier: float = 0.0,
    ):
        # Пространство действий - это id машины [0, max_truck_num-1] + [-1]
        self.action_space = spaces.Discrete(GENERATOR_SETTINGS.max_truck_num+1)
        self._observation_feature_config = observation_feature_config
        self.observation_space = spaces.Dict(self._build_observation_space(observation_feature_config))

        self._static_obs = None     # Это неизменные наблюдения (как временные окна заявок и ограничения)
        self._simulator = Simulator()
        self._current_env: Environment = None
        self._current_requests_constrains = None
        self._generator = input_generator
        self._fixed_instances = fixed_instances or []
        self._fixed_instance_cursor = 0
        self._terminal_reward_multiplier = terminal_reward_multiplier
        self._max_selection_len = -1
        self._obs_builder = None

        self._current_selection = []
        self._current_step = 1
        self._current_observation = None

    @staticmethod
    def _build_observation_space(feature_config: ObservationFeatureConfig) -> dict:
        observation_space = {}
        pairwise_shape = (
            feature_config.pairwise_lookahead_requests,
            GENERATOR_SETTINGS.max_truck_num,
        )
        if feature_config.use_time_windows:
            observation_space["time_windows"] = spaces.Box(
                0.0, 1.0, shape=(GENERATOR_SETTINGS.max_requests_num, 2), dtype=np.float32
            )
        if feature_config.use_executed_requests:
            observation_space["executed_requests"] = spaces.MultiBinary(GENERATOR_SETTINGS.max_requests_num)
        if feature_config.use_unfinished_ratio:
            observation_space["unfinished_ratio"] = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        if feature_config.use_current_selection:
            observation_space["current_selection"] = spaces.Box(
                low=-1,
                high=GENERATOR_SETTINGS.max_truck_num - 1,
                shape=(GENERATOR_SETTINGS.max_requests_num,),
                dtype=np.int64
            )
        if feature_config.use_next_request_tw:
            observation_space["next_request_tw"] = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        if feature_config.use_pairwise_features:
            observation_space["travel_time_to_load"] = spaces.Box(
                0.0, 1.0, shape=pairwise_shape, dtype=np.float32
            )
            observation_space["travel_time_with_cargo_to_unload"] = spaces.Box(
                0.0, 1.0, shape=pairwise_shape, dtype=np.float32
            )
            observation_space["earliness_to_window_start"] = spaces.Box(
                0.0, 1.0, shape=pairwise_shape, dtype=np.float32
            )
            observation_space["lateness_to_window_start"] = spaces.Box(
                0.0, 1.0, shape=pairwise_shape, dtype=np.float32
            )
        return observation_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)

        if self._fixed_instances:
            if seed is not None:
                self._fixed_instance_cursor = seed % len(self._fixed_instances)
            input_data, routes_data = copy.deepcopy(self._fixed_instances[self._fixed_instance_cursor])
            self._fixed_instance_cursor = (self._fixed_instance_cursor + 1) % len(self._fixed_instances)
        else:
            if seed is not None:
                self._generator.reseed(seed)
            input_data, routes_data = self._generator.generate_all(None)
        self._current_env: Environment = get_env(input_data, routes_data)
        self._current_requests_constrains = get_requests_constraints(self._current_env, with_missed=True)
        self._obs_builder = ObservationBuilder(
            self._current_env,
            self._current_requests_constrains,
            self._observation_feature_config
        )

        self._current_selection = []
        self._current_step = 1

        observation = self._obs_builder.create_observation([], [])
        self._current_observation = observation
        info = {
            "missed_requests_num": 0,
            "unfinished_ratio": self._build_unfinished_ratio([], self._current_selection),
            "current_selection": self._current_selection
        }

        return observation, info

    def action_masks(self) -> np.ndarray[bool]:
        # Возвращаем любую маску при превышении лимита заявок
        if len(self._current_selection) >= self._current_env.requests_num:
            print("Превышен лимит заявок при попытке создать маску!")
            return [1] * (GENERATOR_SETTINGS.max_requests_num + 1)

        # Тк action_masks вызывается перед выполнением действия на следующую заявку, то
        # нужно выдать маску на следующую заявку
        current_request_id = len(self._current_selection)
        action_mask = self._obs_builder.create_action_mask(current_request_id=current_request_id)
        return action_mask

    def _apply_restrictions_to_selection(self, selection: list[int]) -> None:
        for request_id, truck_id in enumerate(selection):
            if truck_id not in self._current_requests_constrains[request_id]:
                selection[request_id] = -1
                print(f"Не выполнилось ограничение для заявки {request_id}! Пытались поставить {truck_id}.")

    @staticmethod
    def _build_unfinished_ratio(missed_requests_ids: list[int], current_selection: list[int]) -> np.ndarray:
        if len(current_selection) == 0:
            return np.array([0.0], dtype=np.float32)
        return np.array([len(missed_requests_ids) / len(current_selection)], dtype=np.float32)

    @staticmethod
    def _slack_penalty(slack: float) -> float:
        if -1.0 <= slack < 0.0:
            return -(abs(slack) ** 2)
        if 0.0 <= slack <= 1.0:
            return 1.0 - ((4.0 * slack) ** 0.5)
        raise ValueError(f"Slack must be in [-1, 1], got {slack}")

    def _get_slack_penalty_for_action(self, action: ActType, observation_before_action: ObsType) -> float:
        truck_id = self.__action_to_truck_id(action)
        if truck_id == -1:
            return 0.0

        chosen_slack = (
            float(observation_before_action["earliness_to_window_start"][0][truck_id])
            - float(observation_before_action["lateness_to_window_start"][0][truck_id])
        )
        return self._slack_penalty(chosen_slack)

    def _calculate_reward(
            self,
            action: ActType,
            observation_before_action: ObsType,
            current_selection: list[int],
            missed_requests: list[int],
            terminated: bool,
    ) -> float:
        last_request_id = len(current_selection) - 1
        chosen_truck_id = self.__action_to_truck_id(action)
        if chosen_truck_id == -1:
            reward = -2.0
        else:
            reward = -1.0 if last_request_id in missed_requests else 1.0
            reward += self._get_slack_penalty_for_action(action, observation_before_action)
        if terminated:
            served_ratio = 1.0 - (len(missed_requests) / self._current_env.requests_num)
            reward += self._terminal_reward_multiplier * served_ratio
        return reward

    def __action_to_truck_id(self, action: ActType):
        # Переводим [0, max_truck_num + 1] вычитанием 1 -> [-1, max_truck_id]
        return action - 1

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # Проверяем, не превышен ли лимит заявок
        if len(self._current_selection) >= self._current_env.requests_num:
            observation = self._current_observation
            info = {
                "missed_requests_num": 0,
                "unfinished_ratio": self._build_unfinished_ratio([], self._current_selection),
                "current_selection": self._current_selection
            }
            return observation, 0.0, True, False, info

        observation_before_action = self._current_observation

        # Ставим машину на текущую заявку
        self._current_selection.append(self.__action_to_truck_id(action))

        # Применяем ограничения (по идее с маской действий эта логика не нужна)
        # self._apply_restrictions_to_selection(self._current_selection)

        # Запускаем симуляцию выборки
        missed_requests_ids, truck_positions, truck_available_times = self._simulator.run(
            tuple(self._current_selection),
            self._current_env
        )
        observation = self._obs_builder.create_observation(
            missed_requests_ids,
            self._current_selection,
            truck_positions,
            truck_available_times
        )
        self._current_observation = observation

        # Проверяем нужно ли заканчивать предсказание
        terminated = len(self._current_selection) >= self._current_env.requests_num

        # Считаем награду
        reward = self._calculate_reward(
            action,
            observation_before_action,
            self._current_selection,
            missed_requests_ids,
            terminated,
        )

        truncated = False
        info = {
            "missed_requests_num": len(missed_requests_ids),
            "unfinished_ratio": self._build_unfinished_ratio(missed_requests_ids, self._current_selection),
            "current_selection": self._current_selection
        }

        return observation, reward, terminated, truncated, info
