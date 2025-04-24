from typing import Optional

from pydantic import BaseModel, ConfigDict

from simulator.units.point import LoadPoint, UnloadPoint


class RequestInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str


class Request(BaseModel):
    id: int
    info: RequestInfo
    point_to_load: LoadPoint
    point_to_unload: UnloadPoint
    fix_route: Optional[str]
    volume: int

    def has_fix_route(self) -> bool:
        return self.fix_route is not None
