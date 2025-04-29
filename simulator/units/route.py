from pydantic import BaseModel, field_validator, ConfigDict

from simulator.units.point import RoutePoint


class Geometry(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    coordinates: list[list[float]]


class Properties(BaseModel):
    model_config = ConfigDict(frozen=True)

    distance: float
    points: list[RoutePoint]

    # @field_validator("points", mode="after")
    # def __init_points(self, points: list[RoutePoint]) -> list[RoutePoint]:
    #     return sorted(points, key=lambda x: x.name)

    @property
    def name(self) -> str:
        return f"{self.points[0].name}_{self.points[1].name}"


class Route(BaseModel):
    type: str
    geometry: Geometry
    properties: Properties
