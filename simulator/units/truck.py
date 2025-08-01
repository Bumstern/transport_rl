import copy

from pydantic import BaseModel, ConfigDict

from simulator.units.point import Point


class Position(BaseModel):
    current_point: Point

    def set_current_point(self, point: Point):
        self.current_point.name = point.name

class CargoParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    capacity: int
    loading_speed: int
    unloading_speed: int


class MovingParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    speed_with_cargo: int
    speed_without_cargo: int

    def calculate_speed(self, with_cargo: bool) -> int:
        if with_cargo:
            return self.speed_with_cargo
        else:
            return self.speed_without_cargo


class TruckInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str


class Truck(BaseModel):
    id: int
    info: TruckInfo
    position: Position
    cargo_params: CargoParams
    moving_params: MovingParams
