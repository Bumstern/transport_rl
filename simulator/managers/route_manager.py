from math import ceil

from simulator.units.point import Point
from simulator.units.request import Request
from simulator.units.route import Route
from simulator.units.truck import Truck


class RouteManager:
    def __init__(self, routes: list[Route]):
        self._matrix = self.__get_route_matrix(routes)
        self._routes = self.__get_route_dict(routes)

    def __get_route_matrix(self, routes: list[Route]) -> dict[str, dict[str, Route]]:
        matrix = {}
        for route in routes:
            point_0_name = route.properties.points[0].name
            point_1_name = route.properties.points[-1].name

            # Сохраняем маршрут в матрицу маршрутов
            if point_0_name not in matrix:
                matrix[point_0_name] = {}
            if point_1_name not in matrix:
                matrix[point_1_name] = {}
            matrix[point_0_name][point_1_name] = route
            matrix[point_1_name][point_0_name] = route
        return matrix

    def __get_route_dict(self, routes: list[Route]) -> dict[str, Route]:
        routes_dict = {route.properties.name: route for route in routes}
        return routes_dict

    def find_route(
            self,
            request: Request,
            departure_point: Point,
            destination_point: Point
    ) -> Route | None:
        if departure_point.name == destination_point.name:
            return None

        truck_route = None
        if request.has_fix_route():
            truck_route = self._routes[request.fix_route]
        else:
            truck_route = self._matrix[destination_point.name][departure_point.name]
        return truck_route

    def calculate_distance_to_point(
            self,
            request: Request,
            departure_point: Point,
            destination_point: Point
    ) -> float:
        route = self.find_route(request, departure_point, destination_point)
        return route.properties.distance

    def calculate_travel_time_to_point(
            self,
            truck: Truck,
            with_cargo: bool,
            request: Request,
            departure_point: Point,
            destination_point: Point
    ) -> int:
        distance = self.calculate_distance_to_point(request, departure_point, destination_point)
        speed = truck.moving_params.calculate_speed(with_cargo)
        return ceil(distance / speed)