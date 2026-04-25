import random

from src.gen_algo.simple_model import GeneticAlgoSimple
from src.gen_algo.base import Genome
from src.gen_algo.model_rl_init import GeneticAlgoWithRLInit


class _RlMutatorMixin:

    def _mutation(self, individual: Genome) -> Genome:
        individual = individual.copy()
        for i in range(self._genome_length):
            if random.random() < self._mutation_rate:
                # Обрезаем выборку (хромосому) до i невключительно, чтобы просимулировать, что
                # будущие заявки еще не расставлены и модель видит только предыдущие
                parted_selection = individual[:i]
                obs = self._obs_builder.create_observation(missed_requests_ids=[], current_selection=parted_selection)
                mask = self._obs_builder.create_action_mask(i)
                action, _ = self._rl_model.predict(obs, action_masks=mask, deterministic=True)
                individual[i] = self._action_to_truck_id(action)
        return individual


class _RlTailMutatorMixin:

    def _mutation(self, individual: Genome) -> Genome:
        individual = individual.copy()
        for i in range(self._genome_length):
            if random.random() < self._mutation_rate:
                for request_id in range(i, self._genome_length):
                    # Перестраиваем весь хвост начиная с точки мутации:
                    # модель видит только уже зафиксированный префикс до request_id.
                    parted_selection = individual[:request_id]
                    obs = self._obs_builder.create_observation(
                        missed_requests_ids=[],
                        current_selection=parted_selection
                    )
                    mask = self._obs_builder.create_action_mask(request_id)
                    action, _ = self._rl_model.predict(obs, action_masks=mask, deterministic=True)
                    individual[request_id] = self._action_to_truck_id(action)
                break
        return individual


class GeneticAlgoWithRlMutator(_RlMutatorMixin, GeneticAlgoWithRLInit):

    def _create_initial_population(self) -> list[Genome]:
        return GeneticAlgoSimple._create_initial_population(self)


class GeneticAlgoWithRlTailMutator(_RlTailMutatorMixin, GeneticAlgoWithRLInit):

    def _create_initial_population(self) -> list[Genome]:
        return GeneticAlgoSimple._create_initial_population(self)


class GeneticAlgoWithInitAndRlMutator(_RlMutatorMixin, GeneticAlgoWithRLInit):
    pass
