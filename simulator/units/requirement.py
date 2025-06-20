from simulator.units.entities import Entities
from simulator.units.request import Request
from simulator.units.truck import Truck


def apply_requirements(requests: Entities, trucks: Entities, with_missed: bool):
    requests_constrains = [[i for i in range(len(trucks))]] * len(requests)
    requirements = [apply_trucks_capacity_requirement]

    for requirement in requirements:
        requests_constrains = requirement(requests_constrains, requests, trucks)

    if with_missed:
        for constrain in requests_constrains:
            constrain.append(-1)

    return requests_constrains


def apply_trucks_capacity_requirement(requests_constrains: list[list[int]], requests: Entities, trucks: Entities):
    fixed_requests_constrains = [[] for _ in range(len(requests))]
    for request_id, constrains in enumerate(requests_constrains):
        for truck_id in constrains:
            truck: Truck = trucks[truck_id]
            request: Request = requests[request_id]
            if truck.cargo_params.capacity >= request.volume:
                fixed_requests_constrains[request_id].append(truck_id)
    return fixed_requests_constrains
