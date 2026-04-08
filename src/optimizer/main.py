from typing import Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, ActType

from src.optimizer.utils.observation_builder import ObservationBuilder
from src.simulator.builder import get_env, get_requests_constraints
from src.simulator.environment import Environment
from src.simulator.model.simulator import Simulator
from src.optimizer.settings import GENERATOR_SETTINGS
from src.simulator.utils.data_generator.generator import InputDataGenerator


class SimulatorEnv(gymnasium.Env):
    def __init__(self, input_generator: InputDataGenerator):
        # Пространство действий - это id машины [0, max_truck_num-1] + [-1]
        self.action_space = spaces.Discrete(GENERATOR_SETTINGS.max_truck_num+1)
        self.observation_space = spaces.Dict({
            # Массив с отнормированным временными окнами у заявок
            "time_windows": spaces.Box(0.0, 1.0, shape=(GENERATOR_SETTINGS.max_requests_num, 2), dtype=np.float32),
            # Бинарная маска с выполненными и невыполненными заявками
            "executed_requests": spaces.MultiBinary(GENERATOR_SETTINGS.max_requests_num),
            # Отношение выполненных заявок к невыполненным
            "unfinished_ratio": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            # Текущее распределение
            "current_selection": spaces.Box(low=-1, high=GENERATOR_SETTINGS.max_truck_num - 1, shape=(GENERATOR_SETTINGS.max_requests_num,), dtype=np.int64),
            # Временные окна следующей задачи
            "next_request_tw": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
        })

        self._static_obs = None     # Это неизменные наблюдения (как временные окна заявок и ограничения)
        self._simulator = Simulator()
        self._current_env: Environment = None
        self._current_requests_constrains = None
        self._generator = input_generator
        self._max_selection_len = -1
        self._obs_builder = None

        self._current_selection = []
        self._current_step = 1

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)

        input_data, routes_data = self._generator.generate_all(None)
        self._current_env: Environment = get_env(input_data, routes_data)
        self._current_requests_constrains = get_requests_constraints(self._current_env, with_missed=True)
        self._obs_builder = ObservationBuilder(self._current_env, self._current_requests_constrains)

        self._current_selection = []
        self._current_step = 1

        observation = self._obs_builder.create_observation([], [])
        info = {
            "missed_requests_num": 0,
            "unfinished_ratio": observation["unfinished_ratio"],
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

    def _calculate_reward(self, current_selection: list[int], missed_requests: list[int]) -> float:
        # return (-missed_requests_len + (current_selection_len - missed_requests_len)) / current_selection_len
        last_request_id = len(current_selection) - 1
        reward = -1 if last_request_id in missed_requests else 1
        return reward

    def __action_to_truck_id(self, action: ActType):
        # Переводим [0, max_truck_num + 1] вычитанием 1 -> [-1, max_truck_id]
        return action - 1

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # Проверяем, не превышен ли лимит заявок
        if len(self._current_selection) >= self._current_env.requests_num:
            observation = self._obs_builder.create_observation([], self._current_selection)
            return observation, 0.0, True, False, {
                "missed_requests_num": 0,
                "unfinished_ratio": observation["unfinished_ratio"],
                "current_selection": self._current_selection
            }

        # Ставим машину на текущую заявку
        self._current_selection.append(self.__action_to_truck_id(action))

        # Применяем ограничения (по идее с маской действий эта логика не нужна)
        self._apply_restrictions_to_selection(self._current_selection)

        # Запускаем симуляцию выборки
        missed_requests_ids = self._simulator.run(tuple(self._current_selection), self._current_env)
        observation = self._obs_builder.create_observation(missed_requests_ids, self._current_selection)

        # Считаем награду
        reward = self._calculate_reward(self._current_selection, missed_requests_ids)

        # Проверяем нужно ли заканчивать предсказание
        terminated = len(self._current_selection) >= self._current_env.requests_num

        truncated = False
        info = {
            "missed_requests_num": len(missed_requests_ids),
            "unfinished_ratio": observation["unfinished_ratio"],
            "current_selection": self._current_selection
        }

        return observation, reward, terminated, truncated, info
