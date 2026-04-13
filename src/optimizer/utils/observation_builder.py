import numpy as np

from src.optimizer.settings import GENERATOR_SETTINGS
from src.simulator.environment import Environment
from src.simulator.units.point import Point


class ObservationBuilder:

    def __init__(self, env: Environment, requests_constrains: list[list[int]]):
        """ Создатель наблюдений и разрешенной маски
        :param env: Объект Environment с заявками, машинами и прочим
        :param requests_constrains: Нужно передавать список разрешенных машин с добавленным [-1]
        """
        self._env = env
        self._req_constrains = requests_constrains
        self._static_obs = self._make_normalized_static_observation()

    @staticmethod
    def __get_binary_list_with_position_mask(list_len: int, mask: list[int], mask_is_zeros: bool):
        if mask_is_zeros:
            binary_list = [1] * list_len
        else:
            binary_list = [0] * list_len

        for position in mask:
            binary_list[position] = 0 if mask_is_zeros else 1
        return binary_list

    @staticmethod
    def __make_obs_from_current_selection(selection: list[int]) -> list[int]:
        # Дополним текущую выборку до максимального кол-ва заявок с помощью -1
        # Это нужно чтобы можно было его передать как наблюдение в модель
        remaining_requests_num = GENERATOR_SETTINGS.max_requests_num - len(selection)
        observation = selection + [-1] * remaining_requests_num
        return observation

    @staticmethod
    def __make_active_list_requests(current_selection: list[int]) -> list[int]:
        # Дополним бинарную маску до максимального кол-ва заявок с помощью -1
        # Это нужно чтобы можно было его передать как наблюдение в модель
        remaining_requests_num = GENERATOR_SETTINGS.max_requests_num - len(current_selection)
        active_list = [1] * len(current_selection) + [0] * remaining_requests_num
        return active_list

    @staticmethod
    def _convert_obs_to_numpy(observation: dict) -> dict:

        return {
            "time_windows": np.array(observation["time_windows"], dtype=np.float32),
            "executed_requests": np.array(observation["executed_requests"], dtype=np.int8),
            "unfinished_ratio": np.array(observation["unfinished_ratio"], dtype=np.float32),
            "current_selection": np.array(observation["current_selection"], dtype=np.int64),
            "next_request_tw": np.array(observation["next_request_tw"], dtype=np.float32),
            "travel_time_to_load": np.array(observation["travel_time_to_load"], dtype=np.float32),
            "time_slack_to_window_start": np.array(observation["time_slack_to_window_start"], dtype=np.float32),
        }

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
                    observation[key] = [observation[key][0] / self._env.end_date, observation[key][1] / self._env.end_date]
                case "travel_time_to_load":
                    observation[key] = [min(travel_time / self._env.end_date, 1.0) for travel_time in observation[key]]
                case "time_slack_to_window_start":
                    observation[key] = [
                        min(max(slack / self._env.end_date, -1.0), 1.0)
                        for slack in observation[key]
                    ]
                case _:
                    raise NotImplementedError

    def _make_normalized_static_observation(self) -> dict:
        max_and_current_requests_len_delta = GENERATOR_SETTINGS.max_requests_num - self._env.requests_num

        time_windows = []
        for request in self._env.requests:
            # Нормализуем временные окна в интервале [0, 1]
            request_window = [request.point_to_load.date_start_window / self._env.end_date, request.point_to_load.date_end_window / self._env.end_date]
            time_windows.append(request_window)
        time_windows += [[0, 0]] * max_and_current_requests_len_delta

        static_observation = {
            "time_windows": time_windows
        }

        return static_observation

    def create_action_mask(self, current_request_id: int) -> np.ndarray[bool]:
        # Тк действие модели на каждом шаге - это выбор id машины на текущую заявку
        # (по сути число из [-1, max_truck_num]), то
        # мне нужно знать ограничения текущей заявки, чтобы вернуть бинарный вектор shape=(1, max_truck_num+1)
        current_req_constrains = np.array(self._req_constrains[current_request_id]) + 1
        current_req_constrains = list(current_req_constrains)
        binary_action_mask = self.__get_binary_list_with_position_mask(
            list_len=GENERATOR_SETTINGS.max_truck_num + 1,
            mask=current_req_constrains,
            mask_is_zeros=False
        )
        return np.array(binary_action_mask, dtype=bool)

    def _get_current_truck_positions(self, truck_positions: list[Point] | None) -> list[Point]:
        if truck_positions is not None:
            return truck_positions
        return [truck.position.current_point.model_copy(deep=True) for truck in self._env.trucks]

    def _get_travel_time_to_load(self, current_selection: list[int], truck_positions: list[Point] | None) -> list[int]:
        travel_time_to_load = [0] * GENERATOR_SETTINGS.max_truck_num
        if len(current_selection) >= self._env.requests_num:
            return travel_time_to_load

        current_truck_positions = self._get_current_truck_positions(truck_positions)
        next_request = self._env.requests[len(current_selection)]
        for truck_id in range(len(self._env.trucks)):
            if truck_id not in self._req_constrains[len(current_selection)]:
                travel_time_to_load[truck_id] = self._env.end_date
                continue
            travel_time_to_load[truck_id] = self._env.route_manager.calculate_travel_time_to_point(
                truck=self._env.trucks[truck_id],
                with_cargo=False,
                request=next_request,
                departure_point=current_truck_positions[truck_id],
                destination_point=next_request.point_to_load
            )
        return travel_time_to_load

    def _get_time_slack_to_window_start(
            self,
            current_selection: list[int],
            travel_time_to_load: list[int],
            truck_available_times: list[int] | None
    ) -> list[int]:
        time_slack_to_window_start = [0] * GENERATOR_SETTINGS.max_truck_num
        if len(current_selection) >= self._env.requests_num:
            return time_slack_to_window_start

        next_request = self._env.requests[len(current_selection)]
        current_truck_available_times = truck_available_times or [0] * len(self._env.trucks)
        for truck_id in range(len(self._env.trucks)):
            if truck_id not in self._req_constrains[len(current_selection)]:
                time_slack_to_window_start[truck_id] = -self._env.end_date
                continue
            time_slack_to_window_start[truck_id] = (
                next_request.point_to_load.date_start_window -
                (current_truck_available_times[truck_id] + travel_time_to_load[truck_id])
            )
        return time_slack_to_window_start

    def create_observation(
            self,
            missed_requests_ids: list[int],
            current_selection: list[int],
            truck_positions: list[Point] | None = None,
            truck_available_times: list[int] | None = None
    ) -> dict:
        # Получаем бинарную маску с выполненными и невыполненными заявками
        not_started_requests_ids = [i for i in range(len(current_selection), GENERATOR_SETTINGS.max_requests_num)]
        executed_requests = self.__get_binary_list_with_position_mask(
            list_len=GENERATOR_SETTINGS.max_requests_num,
            mask=missed_requests_ids + not_started_requests_ids,
            mask_is_zeros=True
        )
        # Считаем отношение невыполненных заявок к выполненным
        if len(current_selection) > 0:
            unfinished_ratio = [ len(missed_requests_ids) / len(current_selection) ]
        else:
            unfinished_ratio = [0]
        # Дополняем выборку незначащими -1 до кол-ва требуемого в наблюдении
        selection_obs = self.__make_obs_from_current_selection(current_selection)
        # Вычисляем временное окно следующей заявки
        if len(current_selection) < self._env.requests_num:
            next_request = self._env.requests[len(current_selection)]
            next_request_tw = [next_request.point_to_load.date_start_window, next_request.point_to_load.date_end_window]
        else:
            next_request_tw = [0, 0]
        travel_time_to_load = self._get_travel_time_to_load(current_selection, truck_positions)
        time_slack_to_window_start = self._get_time_slack_to_window_start(
            current_selection,
            travel_time_to_load,
            truck_available_times
        )
        observation = {
            "executed_requests": executed_requests,
            "unfinished_ratio": unfinished_ratio,
            "current_selection": selection_obs,
            "next_request_tw": next_request_tw,
            "travel_time_to_load": travel_time_to_load,
            "time_slack_to_window_start": time_slack_to_window_start,
        }
        self._normalize_observation(observation)
        observation.update(self._static_obs)

        return self._convert_obs_to_numpy(observation)


