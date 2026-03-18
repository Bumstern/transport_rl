from abc import ABC, abstractmethod


type Genome = list[int]


class GeneticAlgoBase(ABC):

    @abstractmethod
    def _create_initial_population(self) -> list[Genome]:
        pass

    @abstractmethod
    def _evaluate_population(self, population: list[Genome]) -> list[int|float]:
        pass

    @abstractmethod
    def _selection(self, population: list[Genome], fitnesses: list[int|float]) -> list[Genome]:
        pass

    @abstractmethod
    def _crossover(self, parent1: Genome, parent2: Genome) -> tuple[Genome, Genome]:
        pass

    @abstractmethod
    def _mutation(self, individual: Genome) -> Genome:
        pass

    @abstractmethod
    def fit(self, iterations: int) -> Genome:
        pass
