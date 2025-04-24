from pydantic import BaseModel

from simulator.managers.route_manager import RouteManager
from simulator.units.entities import Entities
from simulator.units.request import Request
from simulator.units.truck import Truck


class Environment(BaseModel):
    route_manager: RouteManager
    trucks: Entities
    requests: Entities
