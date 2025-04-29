import random
from pathlib import Path

from simulator.builder import Builder


def main():
    random.seed(0)
    builder = Builder(
        input_path=Path('input/input.json'),
        routes_path=Path('input/routes.json')
    )
    builder.run_selection(selection=[random.randint(0, len(builder._env.trucks) - 1) for i in range(len(builder._env.requests))])


if __name__ == '__main__':
    main()
