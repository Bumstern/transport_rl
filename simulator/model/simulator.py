import math
import copy

from pydantic.v1.generics import replace_types

from simulator.environment import Environment
from simulator.managers.task_manager import TaskManager
from simulator.units.point import LoadPoint, Point
from simulator.units.request import Request
from simulator.units.truck import Truck, Position


class Simulator:
    def __init__(self, env: Environment):
        self._env = env

    def _request_simulation(
            self,
            truck: Truck,
            request: Request,
            current_time: int
    ) -> (bool, int):
        task_completed, request_time = self._load_process(
            truck=truck,
            request=request,
            current_time=current_time
        )

        if task_completed:
            request_time += self._unload_process(
                truck=truck,
                request=request
            )
        else:
            return task_completed, 0
        return task_completed, request_time

    def __check_truck_be_on_time_for_request(
            self,
            request: Request,
            current_time: int,
            request_time: int,
            travel_time: int
    ) -> bool:
        if current_time + request_time + travel_time <= request.point_to_load.date_start_window:
            return True
        else:
            return False

    def _cargo_process(
            self,
            truck: Truck,
            request: Request,
            is_loading_process: bool
    ):
        if is_loading_process:
            process_speed = truck.cargo_params.loading_speed
        else:
            process_speed = truck.cargo_params.unloading_speed

        cargo_time = math.ceil(request.volume / process_speed)
        return cargo_time

    def _load_process(
            self,
            truck: Truck,
            request: Request,
            current_time: int
    ) -> (bool, int):
        request_time = 0
        current_point: Point = truck.position.current_point
        next_point: Point = request.point_to_load

        travel_time = self._env.route_manager.calculate_travel_time_to_point(
            truck=truck,
            with_cargo=False,
            request=request,
            departure_point=current_point,
            destination_point=next_point
        )

        task_completed_flag = self.__check_truck_be_on_time_for_request(
            request=request,
            current_time=current_time,
            request_time=request_time,
            travel_time=travel_time
        )

        if task_completed_flag:
            truck.position.set_current_point(next_point)
            request_time += travel_time
            cargo_time = self._cargo_process(
                truck=truck,
                request=request,
                is_loading_process=True
            )
            request_time += cargo_time
        else:
            return task_completed_flag, 0

        return task_completed_flag, request_time

    def _unload_process(
            self,
            truck: Truck,
            request: Request
    ) -> int:
        request_time = 0
        current_point: Point = truck.position.current_point
        next_point: Point = request.point_to_unload

        travel_time = self._env.route_manager.calculate_travel_time_to_point(
            truck=truck,
            with_cargo=True,
            request=request,
            departure_point=current_point,
            destination_point=next_point
        )

        truck.position.set_current_point(next_point)
        request_time += travel_time
        cargo_time = self._cargo_process(
            truck=truck,
            request=request,
            is_loading_process=False
        )
        request_time += cargo_time

        return request_time

    def _save_state(
            self,
            request_time: int,
            current_time: int
    ) -> int:
        current_time += request_time
        return current_time

    def _reset_state(
            self,
            truck: Truck,
            saved_position: Position,
            count_of_missed_requests: int
    ) -> (Truck, int):
        truck.position = saved_position
        count_of_missed_requests += 1
        return truck, count_of_missed_requests

    def __get_copy_of_trucks(self) -> list[Truck]:
        copied_trucks = []
        for truck in self._env.trucks:
            copied_truck = copy.copy(truck)
            copied_trucks.append(copied_truck)
        return copied_trucks

    def run(self, selection: tuple[int]):
        # Присваиваем каждой машине список своих заказов с помощью TaskManager
        task_manager = TaskManager(selection, self._env)

        # Получаем копии машин, чтобы не изменить их в env
        trucks = self.__get_copy_of_trucks()

        count_of_missed_requests = 0
        # В цикле по каждой машине
        for truck in trucks:
            truck: Truck
            current_time = 0

            # Для каждого заказа машины
            for request in task_manager.iter_by(truck.info.name):
                # Сохраняем состояние машины до выполнения заказа
                saved_position: Position = truck.position.model_copy(deep=True)

                # Симулируем выполнение заказа
                task_completed, request_time = self._request_simulation(
                    truck=truck,
                    request=request,
                    current_time=current_time
                )

                if task_completed:
                    # Сохраняем состояние, если заказ выполнен
                    current_time = self._save_state(request_time=request_time, current_time=current_time)
                else:
                    # Откатываем изменения, если заказ не выполнен
                    count_of_missed_requests = self._reset_state(
                        truck=truck,
                        saved_position=saved_position,
                        count_of_missed_requests=count_of_missed_requests
                    )

        return count_of_missed_requests
