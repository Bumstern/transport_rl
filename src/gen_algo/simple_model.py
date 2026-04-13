import random

from src.gen_algo.base import GeneticAlgoBase, Genome
from src.simulator.model.simulator import Simulator


class GeneticAlgoSimple(GeneticAlgoBase):
    """ Простой генетический алгоритм

        :arg _genome_length: Длина хромосомы (кол-во заявок)
                (вычисляется на основе requests_constrains)

    """

    def __init__(
            self,
            simulator: Simulator,
            requests_constrains: list[list[int]],
            popul_size: int = 100,
            mutation_rate: float = 0.1,
            retain_rate: float = 0.2
    ):
        """
        :param simulator: Модель симулятора бизнесс-логики выборки машин
        :param requests_constrains: Список из разрешенных индексов машин для каждой заявки
        :param popul_size: Размер популяции
        :param mutation_rate: Вероятность мутации каждого гена
        :param retain_rate: Доля лучших особей, которые выживают и становятся родителями
        """
        self._genome_length = len(requests_constrains)
        self._popul_size = popul_size
        assert 0 <= mutation_rate <= 1, "Вероятность мутации должна быть в пределах [0,1]"
        self._mutation_rate = mutation_rate
        assert 0 <= retain_rate <= 1, "Доля лучших особей должна быть в пределах [0,1]"
        self._retain_rate = retain_rate
        self._simulator = simulator
        self._requests_constrains = requests_constrains

    def _create_initial_population(self) -> list[Genome]:
        """Создает начальную популяцию из случайных индексов машин."""
        return [[random.choice(self._requests_constrains[i]) for i in range(self._genome_length)] for _ in range(self._popul_size)]

    def _fitness_function(self, individual: Genome) -> int:
        """Целевая функция: считаем количество выполненных заявок."""
        missed_requests_ids, _, _ = self._simulator.run(tuple(individual))
        return self._genome_length - len(missed_requests_ids)

    def _evaluate_population(self, population: list[Genome]) -> list[int]:
        """Оценивает всю популяцию."""
        return [self._fitness_function(ind) for ind in population]

    def _selection(self, population: list[Genome], fitnesses: list[int]) -> list[Genome]:
        """Отбор лучших особей (Truncation selection)."""
        # Объединяем особей с их оценками и сортируем по убыванию приспособленности
        pop_fit = list(zip(population, fitnesses))
        pop_fit.sort(key=lambda x: x[1], reverse=True)

        # Оставляем только топ X процентов для размножения
        retain_length = int(len(population) * self._retain_rate)
        parents = [ind for ind, fit in pop_fit[:retain_length]]
        return parents

    def _crossover(self, parent1: Genome, parent2: Genome) -> tuple[Genome, Genome]:
        """Одноточечное скрещивание."""
        # Выбираем случайную точку разреза
        point = random.randint(1, self._genome_length - 1)

        # Создаем двух потомков, обмениваясь частями родителей
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutation(self, individual: Genome) -> Genome:
        """Случайная мутация: случайная машина с заданной вероятностью."""
        individual = individual.copy()
        for i in range(self._genome_length):
            if random.random() < self._mutation_rate:
                individual[i] = random.choice(self._requests_constrains[i])
        return individual

    def fit(self, iterations: int) -> Genome:
        """Основной цикл эволюции."""
        population = self._create_initial_population()

        for iter in range(iterations):
            # 1. Оценка текущего поколения
            fitnesses = self._evaluate_population(population)
            best_fitness = max(fitnesses)
            best_individual = population[fitnesses.index(best_fitness)]

            print(f"Поколение {iter + 1}: Лучшая приспособленность = {best_fitness}")

            # 2. Селекция (отбор родителей)
            parents = self._selection(population, fitnesses)

            # 3. Формирование нового поколения
            next_generation = []

            # Элитизм: гарантированно переносим лучшую особь в следующее поколение
            next_generation.append(best_individual)

            # 4. Скрещивание и мутация для заполнения остальной популяции
            while len(next_generation) < self._popul_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)

                c1, c2 = self._crossover(p1, p2)

                next_generation.append(self._mutation(c1))
                if len(next_generation) < self._popul_size:
                    next_generation.append(self._mutation(c2))

            population = next_generation

        # Оценка последнего поколения после завершения цикла
        final_fitnesses = self._evaluate_population(population)
        best_idx = final_fitnesses.index(max(final_fitnesses))
        return population[best_idx]
