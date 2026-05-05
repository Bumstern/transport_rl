from src.simulator.environment import Environment
from src.simulator.model.simulator import Simulator


def _build_route(point_from_name: str, point_to_name: str, distance: int) -> dict:
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[1, 1], [2, 2]],
        },
        "properties": {
            "distance": distance,
            "points": [
                {"name": point_from_name},
                {"name": point_to_name},
            ],
        },
    }


def _build_environment(
    *,
    trucks: list[dict] | None = None,
    requests: list[dict] | None = None,
    routes: list[dict] | None = None,
) -> Environment:
    default_trucks = [
        {
            "info": {"name": "Truck_0"},
            "position": {"current_point": {"name": "DEPOT"}},
            "cargo_params": {
                "capacity": 100,
                "loading_speed": 10,
                "unloading_speed": 10,
            },
            "moving_params": {
                "speed_with_cargo": 10,
                "speed_without_cargo": 10,
            },
        },
        {
            "info": {"name": "Truck_1"},
            "position": {"current_point": {"name": "DEPOT"}},
            "cargo_params": {
                "capacity": 100,
                "loading_speed": 10,
                "unloading_speed": 10,
            },
            "moving_params": {
                "speed_with_cargo": 10,
                "speed_without_cargo": 10,
            },
        },
    ]
    default_requests = [
        {
            "info": {"name": "Request_A"},
            "point_to_load": {
                "name": "LOAD_A",
                "date_start_window": 10,
                "date_end_window": 20,
            },
            "point_to_unload": {"name": "UNLOAD_A"},
            "fix_route": None,
            "volume": 10,
        },
        {
            "info": {"name": "Request_B"},
            "point_to_load": {
                "name": "LOAD_B",
                "date_start_window": 8,
                "date_end_window": 20,
            },
            "point_to_unload": {"name": "UNLOAD_B"},
            "fix_route": None,
            "volume": 10,
        },
    ]
    default_routes = [
        _build_route("DEPOT", "LOAD_A", 50),
        _build_route("DEPOT", "LOAD_B", 20),
        _build_route("DEPOT", "UNLOAD_A", 70),
        _build_route("DEPOT", "UNLOAD_B", 40),
        _build_route("LOAD_A", "UNLOAD_A", 20),
        _build_route("LOAD_B", "UNLOAD_B", 30),
        _build_route("LOAD_A", "LOAD_B", 30),
        _build_route("LOAD_A", "UNLOAD_B", 35),
        _build_route("LOAD_B", "UNLOAD_A", 25),
        _build_route("UNLOAD_A", "UNLOAD_B", 15),
    ]

    return Environment(
        end_date=100,
        route_manager=routes or default_routes,
        trucks=trucks or default_trucks,
        requests=requests or default_requests,
    )


def test_simulator_distributes_all_selected_requests_between_trucks() -> None:
    environment = _build_environment()
    simulator = Simulator(environment)

    missed_requests_ids, truck_positions, truck_available_times = simulator.run((0, 1))

    assert missed_requests_ids == []
    assert truck_positions[0].name == "UNLOAD_B"
    assert truck_positions[1].name == "UNLOAD_A"
    assert truck_available_times == [13, 14]


def test_simulator_does_not_assign_skipped_request_to_any_truck() -> None:
    environment = _build_environment()
    simulator = Simulator(environment)

    missed_requests_ids, truck_positions, truck_available_times = simulator.run((-1, 1))

    assert missed_requests_ids == [0]
    assert truck_positions[0].name == "DEPOT"
    assert truck_positions[1].name == "UNLOAD_A"
    assert truck_available_times == [0, 14]


def test_request_started_at_window_start_when_truck_arrives_early() -> None:
    environment = _build_environment(
        requests=[
            {
                "info": {"name": "Request_Early"},
                "point_to_load": {
                    "name": "LOAD_A",
                    "date_start_window": 10,
                    "date_end_window": 20,
                },
                "point_to_unload": {"name": "UNLOAD_A"},
                "fix_route": None,
                "volume": 10,
            },
        ]
    )
    simulator = Simulator(environment)
    truck = environment.trucks[0].model_copy(deep=True)

    completed, request_time = simulator._request_simulation(
        truck=truck,
        request=environment.requests[0],
        current_time=0,
    )

    # travel=5h, wait=5h, load=1h, unload-travel=2h, unload=1h
    assert completed is True
    assert request_time == 14


def test_request_started_immediately_when_truck_arrives_inside_window() -> None:
    environment = _build_environment(
        requests=[
            {
                "info": {"name": "Request_Inside_Window"},
                "point_to_load": {
                    "name": "LOAD_A",
                    "date_start_window": 10,
                    "date_end_window": 20,
                },
                "point_to_unload": {"name": "UNLOAD_A"},
                "fix_route": None,
                "volume": 10,
            },
        ]
    )
    simulator = Simulator(environment)
    truck = environment.trucks[0].model_copy(deep=True)

    completed, request_time = simulator._request_simulation(
        truck=truck,
        request=environment.requests[0],
        current_time=7,
    )

    # travel=5h, wait=0h, load=1h, unload-travel=2h, unload=1h
    assert completed is True
    assert request_time == 9


def test_late_request_is_not_executed_and_truck_state_is_rolled_back() -> None:
    environment = _build_environment(
        requests=[
            {
                "info": {"name": "Request_Late"},
                "point_to_load": {
                    "name": "LOAD_A",
                    "date_start_window": 10,
                    "date_end_window": 20,
                },
                "point_to_unload": {"name": "UNLOAD_A"},
                "fix_route": None,
                "volume": 10,
            },
        ],
        routes=[
            _build_route("DEPOT", "LOAD_A", 250),
            _build_route("DEPOT", "UNLOAD_A", 270),
            _build_route("LOAD_A", "UNLOAD_A", 20),
        ],
    )
    simulator = Simulator(environment)

    missed_requests_ids, truck_positions, truck_available_times = simulator.run((0,))

    assert missed_requests_ids == [0]
    assert truck_positions[0].name == "DEPOT"
    assert truck_available_times[0] == 0
    assert environment.trucks[0].position.current_point.name == "DEPOT"
