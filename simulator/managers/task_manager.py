from collections import deque

from simulator.environment import Environment
from simulator.units.request import Request
from simulator.units.truck import Truck


class TaskManager:
    def __init__(self, selection: tuple[int], env: Environment):
        self._requests_per_truck = self.__set_requests_per_truck(selection, env)

    def __set_requests_per_truck(self, selection: tuple[int], env: Environment) -> dict[str, list[Request]]:
        requests_per_truck: dict[str, list[Request]] = {}
        for request_id, truck_id in enumerate(selection):
            if truck_id == -1:
                continue
            truck: Truck = env.trucks[truck_id]
            request: Request = env.requests[request_id]
            if truck.info.name not in requests_per_truck:
                requests_per_truck[truck.info.name] = []
            requests_per_truck[truck.info.name].append(request)

        for request_list in requests_per_truck. values():
            request_list.sort(key=lambda req: req.point_to_load.date_start_window)

        return requests_per_truck

    def iter_by(self, truck_name: str):
        return self._requests_per_truck.get(truck_name, []).copy()
