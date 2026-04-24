import numpy as np

from src.optimizer.settings import GENERATOR_SETTINGS, DEFAULT_OBSERVATION_FEATURES, ObservationFeatureConfig
from src.simulator.environment import Environment
from src.simulator.units.point import Point


class ObservationBuilder:
    _EMPTY_TRAVEL_TIME = GENERATOR_SETTINGS.max_requests_num


    def __init__(
            self,
            env: Environment,
            requests_constrains: list[list[int]],
            feature_config: ObservationFeatureConfig = DEFAULT_OBSERVATION_FEATURES
    ):
        """ Создатель наблюдений и разрешенной маски
        :param env: Объект Environment с заявками, машинами и прочим
        :param requests_constrains: Нужно передавать список разрешенных машин с добавленным [-1]
        """
        self._env = env
        self._req_constrains = requests_constrains
        self._feature_config = feature_config
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
        dtype_by_key = {
            "time_windows": np.float32,
            "executed_requests": np.int8,
            "unfinished_ratio": np.float32,
            "current_selection": np.int64,
            "next_request_tw": np.float32,
            "travel_time_to_load": np.float32,
            "travel_time_with_cargo_to_unload": np.float32,
            "earliness_to_window_start": np.float32,
            "lateness_to_window_start": np.float32,
        }
        return {
            key: np.array(value, dtype=dtype_by_key[key])
            for key, value in observation.items()
        }

    def _normalize_observation(self, observation: dict) -> None:
        def _normalize_matrix(values: list[list[float]], mapper) -> list[list[float]]:
            return [
                [mapper(value) for value in row]
                for row in values
            ]

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
                    observation[key] = _normalize_matrix(
                        observation[key],
                        lambda travel_time: min(travel_time / self._env.end_date, 1.0)
                    )
                case "travel_time_with_cargo_to_unload":
                    observation[key] = _normalize_matrix(
                        observation[key],
                        lambda travel_time: min(travel_time / self._env.end_date, 1.0)
                    )
                case "earliness_to_window_start":
                    observation[key] = _normalize_matrix(
                        observation[key],
                        lambda value: min(max(value / self._env.end_date, 0.0), 1.0)
                    )
                case "lateness_to_window_start":
                    observation[key] = _normalize_matrix(
                        observation[key],
                        lambda value: min(max(value / self._env.end_date, 0.0), 1.0)
                    )
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

        static_observation = {}
        if self._feature_config.use_time_windows:
            static_observation["time_windows"] = time_windows

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

    def _build_empty_pairwise_row(
            self,
            *,
            travel_time_value: int | None = None,
            slack_value: int | None = None,
    ) -> list[int]:
        if travel_time_value is not None:
            return [travel_time_value] * GENERATOR_SETTINGS.max_truck_num
        if slack_value is not None:
            return [slack_value] * GENERATOR_SETTINGS.max_truck_num
        raise ValueError("Either travel_time_value or slack_value must be provided")

    def _get_travel_time_to_load_for_request(
            self,
            request_id: int,
            truck_positions: list[Point] | None,
    ) -> list[int]:
        travel_time_to_load = [0] * GENERATOR_SETTINGS.max_truck_num
        if request_id >= self._env.requests_num:
            return self._build_empty_pairwise_row(travel_time_value=self._env.end_date)

        current_truck_positions = self._get_current_truck_positions(truck_positions)
        next_request = self._env.requests[request_id]
        for truck_id in range(len(self._env.trucks)):
            if truck_id not in self._req_constrains[request_id]:
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
            request_id: int,
            travel_time_to_load: list[int],
            truck_available_times: list[int] | None
    ) -> list[int]:
        time_slack_to_window_start = [0] * GENERATOR_SETTINGS.max_truck_num
        if request_id >= self._env.requests_num:
            return self._build_empty_pairwise_row(slack_value=-self._env.end_date)

        next_request = self._env.requests[request_id]
        current_truck_available_times = truck_available_times or [0] * len(self._env.trucks)
        for truck_id in range(len(self._env.trucks)):
            if truck_id not in self._req_constrains[request_id]:
                time_slack_to_window_start[truck_id] = -self._env.end_date
                continue
            time_slack_to_window_start[truck_id] = (
                next_request.point_to_load.date_start_window -
                (current_truck_available_times[truck_id] + travel_time_to_load[truck_id])
            )
        return time_slack_to_window_start

    def _get_travel_time_with_cargo_to_unload_for_request(self, request_id: int) -> list[int]:
        travel_time_with_cargo_to_unload = [0] * GENERATOR_SETTINGS.max_truck_num
        if request_id >= self._env.requests_num:
            return self._build_empty_pairwise_row(travel_time_value=self._env.end_date)

        next_request = self._env.requests[request_id]
        for truck_id in range(len(self._env.trucks)):
            if truck_id not in self._req_constrains[request_id]:
                travel_time_with_cargo_to_unload[truck_id] = self._env.end_date
                continue
            travel_time_with_cargo_to_unload[truck_id] = self._env.route_manager.calculate_travel_time_to_point(
                truck=self._env.trucks[truck_id],
                with_cargo=True,
                request=next_request,
                departure_point=next_request.point_to_load,
                destination_point=next_request.point_to_unload
            )
        return travel_time_with_cargo_to_unload

    @staticmethod
    def _split_slack_to_earliness_and_lateness(time_slack_to_window_start: list[int]) -> tuple[list[int], list[int]]:
        earliness_to_window_start = [max(slack, 0) for slack in time_slack_to_window_start]
        lateness_to_window_start = [max(-slack, 0) for slack in time_slack_to_window_start]
        return earliness_to_window_start, lateness_to_window_start

    def _get_pairwise_feature_matrices(
            self,
            current_selection: list[int],
            truck_positions: list[Point] | None,
            truck_available_times: list[int] | None,
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
        travel_time_to_load = []
        travel_time_with_cargo_to_unload = []
        earliness_to_window_start = []
        lateness_to_window_start = []

        current_request_id = len(current_selection)
        for lookahead_offset in range(self._feature_config.pairwise_lookahead_requests):
            request_id = current_request_id + lookahead_offset
            current_travel_time_to_load = self._get_travel_time_to_load_for_request(
                request_id=request_id,
                truck_positions=truck_positions,
            )
            current_travel_time_with_cargo_to_unload = self._get_travel_time_with_cargo_to_unload_for_request(
                request_id=request_id
            )
            current_time_slack_to_window_start = self._get_time_slack_to_window_start(
                request_id=request_id,
                travel_time_to_load=current_travel_time_to_load,
                truck_available_times=truck_available_times,
            )
            current_earliness_to_window_start, current_lateness_to_window_start = (
                self._split_slack_to_earliness_and_lateness(current_time_slack_to_window_start)
            )

            travel_time_to_load.append(current_travel_time_to_load)
            travel_time_with_cargo_to_unload.append(current_travel_time_with_cargo_to_unload)
            earliness_to_window_start.append(current_earliness_to_window_start)
            lateness_to_window_start.append(current_lateness_to_window_start)

        return (
            travel_time_to_load,
            travel_time_with_cargo_to_unload,
            earliness_to_window_start,
            lateness_to_window_start,
        )

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
        (
            travel_time_to_load,
            travel_time_with_cargo_to_unload,
            earliness_to_window_start,
            lateness_to_window_start,
        ) = self._get_pairwise_feature_matrices(
            current_selection=current_selection,
            truck_positions=truck_positions,
            truck_available_times=truck_available_times,
        )
        observation = {}
        if self._feature_config.use_executed_requests:
            observation["executed_requests"] = executed_requests
        if self._feature_config.use_unfinished_ratio:
            observation["unfinished_ratio"] = unfinished_ratio
        if self._feature_config.use_current_selection:
            observation["current_selection"] = selection_obs
        if self._feature_config.use_next_request_tw:
            observation["next_request_tw"] = next_request_tw
        if self._feature_config.use_pairwise_features:
            observation["travel_time_to_load"] = travel_time_to_load
            observation["travel_time_with_cargo_to_unload"] = travel_time_with_cargo_to_unload
            observation["earliness_to_window_start"] = earliness_to_window_start
            observation["lateness_to_window_start"] = lateness_to_window_start
        self._normalize_observation(observation)
        observation.update(self._static_obs)

        return self._convert_obs_to_numpy(observation)
