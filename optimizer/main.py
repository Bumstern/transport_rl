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
        self.action_space = spaces.MultiDiscrete([GENERATOR_SETTINGS.max_truck_num + 1] * GENERATOR_SETTINGS.max_requests_num)
        self.observation_space = spaces.Dict({
            "requests_constraints": spaces.MultiBinary([GENERATOR_SETTINGS.max_requests_num, GENERATOR_SETTINGS.max_truck_num]),
            "time_windows": spaces.Box(0.0, 1.0, shape=(GENERATOR_SETTINGS.max_requests_num, 2), dtype=floating),
            "executed_requests": spaces.MultiBinary(GENERATOR_SETTINGS.max_requests_num),
            "unfinished_ratio": spaces.Box(0.0, 1.0, shape=(1,), dtype=floating),
            # "is_active": spaces.MultiBinary(SETTINGS.max_requests_num),  # если число заявок переменное
        })

        self._static_obs = None
        self._simulator = Simulator()
        self._current_env: Environment = None
        self._current_requests_constrains = None
        self._generator = input_generator

        self._current_step = 1

    def __get_binary_list_with_position_mask(self, list_len: int,  mask: list[int], mask_is_zeros: bool):
        if mask_is_zeros:
            binary_list = [1] * list_len
        else:
            binary_list = [0] * list_len

        for position in mask:
            binary_list[position] = 0 if mask_is_zeros else 1
        return binary_list

    def _make_normalized_observation(self, missed_requests_ids: list[int]):
        executed_requests = self.__get_binary_list_with_position_mask(
            list_len=len(self._current_env.requests),
            mask=missed_requests_ids,
            mask_is_zeros=True
        )
        unfinished_ratio = [ len(missed_requests_ids) / len(self._current_env.requests) ]
        observation = {
            "executed_requests": executed_requests,
            "unfinished_ratio": unfinished_ratio,
        }
        self._normalize_observation(observation)
        observation.update(self._static_obs)

        return observation

    def _normalize_observation(self, observation: dict) -> None:
        for key in observation.keys():
            match key:
                case "executed_requests":
                    # Ничего не нужно нормализовать
                    pass
                case "unfinished_ratio":
                    # Ничего не нужно нормализовать
                    pass
                case _:
                    raise NotImplementedError

    def _make_normalized_static_observation(self) -> dict:
        constrains_matrix_requests_by_trucks = []
        for request_id, constrains in enumerate(self._current_requests_constrains):
            request_binary_constrains = self.__get_binary_list_with_position_mask(
                list_len=len(self._current_env.trucks),
                mask=constrains,
                mask_is_zeros=False
            )
            constrains_matrix_requests_by_trucks.append(request_binary_constrains)

        time_windows = []
        for request in self._current_env.requests:
            request_window = [request.point_to_load.date_start_window, request.point_to_load.date_end_window]
            time_windows.append(request_window)

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

    def _apply_restrictions_to_action(self, action: ActType) -> None:
        for request_id, truck_id in enumerate(action):
            if truck_id not in self._current_requests_constrains[request_id]:
                action[request_id] = -1

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)

        input_data, routes_data = self._generator.generate_all(None)
        self._current_env: Environment = get_env(input_data, routes_data)
        self._current_requests_constrains = get_requests_constrains(self._current_env, with_missed=False)
        self._static_obs = self._make_normalized_static_observation()

        self._current_step = 1

        # selection = (self.action_space.sample() - 1).tolist()
        selection = [-1] * len(self._current_env.requests)
        self._apply_restrictions_to_action(selection)
        selection = tuple(selection)

        missed_requests_ids = self._simulator.run(selection, self._current_env)
        observation = self._make_normalized_observation(missed_requests_ids)
        info = {
            "missed_requests_num": len(missed_requests_ids),
            "unfinished_ratio": observation["unfinished_ratio"]
        }

        return observation, info

    def _calculate_reward(self, missed_requests_ids: list[int]):
        return (-len(missed_requests_ids) + (len(self._current_env.requests) - len(missed_requests_ids))) / len(self._current_env.requests)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        selection = (action - 1).tolist()
        self._apply_restrictions_to_action(selection)
        selection = tuple(selection)

        missed_requests_ids = self._simulator.run(selection, self._current_env)
        observation = self._make_normalized_observation(missed_requests_ids)

        reward = self._calculate_reward(missed_requests_ids)

        terminated = False
        if self._current_step >= ENV_SETTINGS.max_num_of_steps:
            terminated = True
        self._current_step += 1

        truncated = False
        info = {
            "missed_requests_num": len(missed_requests_ids),
            "unfinished_ratio": observation["unfinished_ratio"]
        }

        return observation, reward, terminated, truncated, info
