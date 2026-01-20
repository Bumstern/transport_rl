from typing import Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
from numpy import floating, integer
from sympy.strategies.core import switch

from simulator.builder import get_env, get_requests_constrains
from simulator.environment import Environment
from simulator.model.simulator import Simulator
from optimizer.settings import ENV_SETTINGS, GENERATOR_SETTINGS
from simulator.units.request import Request
from simulator.utils.data_generator.generator import InputDataGenerator


class SimulatorEnv(gymnasium.Env):
    def __init__(self, input_generator: InputDataGenerator):
        self.action_space = spaces.Box(low=-1, high=GENERATOR_SETTINGS.max_truck_num, shape=(1,), dtype=integer)
        self.observation_space = spaces.Dict({
            # Бинарная матрица разрешенных машин на заявках
            "requests_constraints": spaces.MultiBinary([GENERATOR_SETTINGS.max_requests_num, GENERATOR_SETTINGS.max_truck_num]),
            # Массив с отнормированным временными окнами у заявок
            "time_windows": spaces.Box(0.0, 1.0, shape=(GENERATOR_SETTINGS.max_requests_num, 2), dtype=floating),
            # Бинарная маска с выполненными и невыполненными заявками
            "executed_requests": spaces.MultiBinary(GENERATOR_SETTINGS.max_requests_num),
            # Отношение выполненных заявок к невыполненным
            "unfinished_ratio": spaces.Box(0.0, 1.0, shape=(1,), dtype=floating),
            # Бинарная маска с текущим распределением
            # "is_active": spaces.MultiBinary(GENERATOR_SETTINGS.max_requests_num),
            # Текущее распределение
            "current_selection": spaces.Box(low=-1, high=GENERATOR_SETTINGS.max_truck_num - 1, shape=(GENERATOR_SETTINGS.max_requests_num,), dtype=integer),
            # Временные окна следующей задачи
            "next_request_tw": spaces.Box(0.0, 1.0, shape=(2,), dtype=floating),
        })

        self._static_obs = None     # Это неизменные наблюдения (как временные окна заявок и ограничения)
        self._simulator = Simulator()
        self._current_env: Environment = None
        self._current_requests_constrains = None
        self._generator = input_generator
        self._max_selection_len = -1

        self._current_selection = []
        self._current_step = 1

    def __get_binary_list_with_position_mask(self, list_len: int,  mask: list[int], mask_is_zeros: bool):
        if mask_is_zeros:
            binary_list = [1] * list_len
        else:
            binary_list = [0] * list_len

        for position in mask:
            binary_list[position] = 0 if mask_is_zeros else 1
        return binary_list

    def __make_obs_from_current_selection(self, selection: list[int]) -> list[int]:
        # Дополним текущую выборку до максимального кол-ва заявок с помощью -1
        # Это нужно чтобы можно было его передать как наблюдение в модель
        remaining_requests_num = GENERATOR_SETTINGS.max_requests_num - len(selection)
        observation = selection + [-1] * remaining_requests_num
        return observation

    def __make_active_list_requests(self, current_selection: list[int]) -> list[int]:
        # Дополним бинарную маску до максимального кол-ва заявок с помощью -1
        # Это нужно чтобы можно было его передать как наблюдение в модель
        remaining_requests_num = GENERATOR_SETTINGS.max_requests_num - len(current_selection)
        active_list = [1] * len(current_selection) + [0] * remaining_requests_num
        return active_list

    def _convert_obs_to_numpy(self, observation: dict) -> dict:
        return {
            "requests_constraints": np.array(observation["requests_constraints"], dtype=np.int8),
            "time_windows": np.array(observation["time_windows"], dtype=np.float64),
            "executed_requests": np.array(observation["executed_requests"], dtype=np.int8),
            "unfinished_ratio": np.array(observation["unfinished_ratio"], dtype=np.float64),
            "current_selection": np.array(observation["current_selection"], dtype=np.int64),
            "next_request_tw": np.array(observation["next_request_tw"], dtype=np.float64),
        }

    def _make_normalized_observation(self, missed_requests_ids: list[int], current_selection: list[int]):
        # Получаем бинарную маску с выполненными и невыполненными заявками
        not_started_requests_ids = [i for i in range(len(current_selection), GENERATOR_SETTINGS.max_requests_num)]
        executed_requests = self.__get_binary_list_with_position_mask(
            list_len=GENERATOR_SETTINGS.max_requests_num,
            mask=missed_requests_ids + not_started_requests_ids,
            mask_is_zeros=True
        )
        if len(current_selection) > 0:
            unfinished_ratio = [ len(missed_requests_ids) / len(current_selection) ]
        else:
            unfinished_ratio = [0]
        selection_obs = self.__make_obs_from_current_selection(current_selection)
        # is_active = self.__make_active_list_requests(current_selection)
        if len(current_selection) < self._current_env.requests_num:
            next_request = self._current_env.requests[len(current_selection)]
            next_request_tw = [next_request.point_to_load.date_start_window, next_request.point_to_load.date_end_window]
        else:
            next_request_tw = [0, 0]
        observation = {
            "executed_requests": executed_requests,
            "unfinished_ratio": unfinished_ratio,
            # "is_active": is_active,
            "current_selection": selection_obs,
            "next_request_tw": next_request_tw
        }
        self._normalize_observation(observation)
        observation.update(self._static_obs)

        return self._convert_obs_to_numpy(observation)

    def _normalize_observation(self, observation: dict) -> None:
        for key in observation.keys():
            match key:
                case "executed_requests":
                    # Ничего не нужно нормализовать
                    pass
                case "unfinished_ratio":
                    # Ничего не нужно нормализовать
                    pass
                case "current_selection":
                    pass
                case "is_active":
                    pass
                case "next_request_tw":
                    observation[key] = [observation[key][0] / self._current_env.end_date, observation[key][1] / self._current_env.end_date]
                case _:
                    raise NotImplementedError

    def _make_normalized_static_observation(self) -> dict:
        max_and_current_requests_len_delta = GENERATOR_SETTINGS.max_requests_num - self._current_env.requests_num

        # Создаем бинарную матрицу разрешенных машин на заявках
        constrains_matrix_requests_by_trucks = []
        for request_id, constrains in enumerate(self._current_requests_constrains):
            request_binary_constrains = self.__get_binary_list_with_position_mask(
                list_len=len(self._current_env.trucks),
                mask=constrains,
                mask_is_zeros=False
            )
            constrains_matrix_requests_by_trucks.append(request_binary_constrains)
        constrains_matrix_requests_by_trucks += [[0] * GENERATOR_SETTINGS.max_truck_num] * max_and_current_requests_len_delta

        time_windows = []
        for request in self._current_env.requests:
            request_window = [request.point_to_load.date_start_window, request.point_to_load.date_end_window]
            time_windows.append(request_window)
        time_windows += [[0, 0]] * max_and_current_requests_len_delta

        static_observation = {
            "requests_constraints": constrains_matrix_requests_by_trucks,
            "time_windows": time_windows
        }
        self._normalize_static_observation(static_observation)

        return static_observation

    def _normalize_static_observation(self, observation: dict) -> None:
        for key in observation.keys():
            match key:
                case "requests_constraints":
                    # Ничего не нужно нормализовать
                    pass
                case "time_windows":
                    observation[key] = list(map(
                        lambda x: [x[0] / self._current_env.end_date, x[1] / self._current_env.end_date],
                        observation[key]
                    )
                    )
                case _:
                    raise NotImplementedError

    def _apply_restrictions_to_selection(self, selection: list[int]) -> None:
        for request_id, truck_id in enumerate(selection):
            if truck_id not in self._current_requests_constrains[request_id]:
                selection[request_id] = -1

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)

        input_data, routes_data = self._generator.generate_all(None)
        self._current_env: Environment = get_env(input_data, routes_data)
        self._current_requests_constrains = get_requests_constrains(self._current_env, with_missed=True)
        self._static_obs = self._make_normalized_static_observation()

        self._current_selection = []
        self._current_step = 1

        observation = self._make_normalized_observation([], [])
        info = {
            "missed_requests_num": 0,
            "unfinished_ratio": observation["unfinished_ratio"],
            "current_selection": self._current_selection
        }

        return observation, info

    def _calculate_reward(self, current_selection: list[int], missed_requests: list[int]) -> float:
        # return (-missed_requests_len + (current_selection_len - missed_requests_len)) / current_selection_len
        last_request_id = len(current_selection) - 1
        reward = -1 if last_request_id in missed_requests else 1
        return reward

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # Проверяем, не превышен ли лимит заявок
        if len(self._current_selection) >= self._current_env.requests_num:
            observation = self._make_normalized_observation([], self._current_selection)
            return observation, 0.0, True, False, {
                "missed_requests_num": 0,
                "unfinished_ratio": observation["unfinished_ratio"],
                "current_selection": self._current_selection
            }

        self._current_selection.append(int(action[0]))

        # selection = (action - 1).tolist()
        self._apply_restrictions_to_selection(self._current_selection)

        missed_requests_ids = self._simulator.run(tuple(self._current_selection), self._current_env)
        observation = self._make_normalized_observation(missed_requests_ids, self._current_selection)

        reward = self._calculate_reward(self._current_selection, missed_requests_ids)

        terminated = len(self._current_selection) >= self._current_env.requests_num

        truncated = False
        info = {
            "missed_requests_num": len(missed_requests_ids),
            "unfinished_ratio": observation["unfinished_ratio"],
            "current_selection": self._current_selection
        }

        return observation, reward, terminated, truncated, info
