from pydantic import BaseModel, ConfigDict


class Point(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str


class RoutePoint(Point):
    pass


class LoadPoint(Point):
    model_config = ConfigDict(frozen=True)

    date_start_window: int
    date_end_window: int


class UnloadPoint(Point):
    model_config = ConfigDict(frozen=True)

    pass
