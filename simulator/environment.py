from pydantic import BaseModel, ConfigDict, field_validator, ValidationError

from simulator.managers.route_manager import RouteManager
from simulator.units.entities import Entities
from simulator.units.request import Request
from simulator.units.route import Route
from simulator.units.truck import Truck


class Environment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed = True)

    route_manager: RouteManager
    trucks: Entities
    requests: Entities

    @field_validator("trucks", mode="before")
    @classmethod
    def __init_trucks(cls, data: list[dict]) -> Entities:
        try:
            return Entities(data, Truck)
        except ValidationError as e:
            print(e.json())
            raise

    @field_validator("requests", mode="before")
    @classmethod
    def __init_requests(cls, data: list[dict]) -> Entities:
        try:
            return Entities(data, Request)
        except ValidationError as e:
            print(e.json())
            raise

    @field_validator("route_manager", mode="before")
    @classmethod
    def __init_route_manager(cls, routes_data: list[dict]) -> RouteManager:
        try:
            routes = []
            for route_elem_data in routes_data:
                route = Route(**route_elem_data)
                routes.append(route)
            return RouteManager(routes)
        except ValidationError as e:
            print(e.json())
            raise
