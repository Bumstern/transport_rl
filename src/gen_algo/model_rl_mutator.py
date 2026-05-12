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


class _RlMissedRequestsSequentialMutatorMixin:

    def _build_prefix_missed_requests_ids(
        self,
        request_id: int,
        pending_missed_requests_ids: list[int],
    ) -> list[int]:
        return [missed_request_id for missed_request_id in pending_missed_requests_ids if missed_request_id < request_id]

    def _mutate_missed_requests_sequentially(
        self,
        individual: Genome,
        missed_requests_ids: list[int],
    ) -> Genome:
        mutated_individual = individual.copy()
        pending_missed_requests_ids = list(missed_requests_ids)

        for request_id in missed_requests_ids:
            parted_selection = mutated_individual[:request_id]
            obs = self._obs_builder.create_observation(
                missed_requests_ids=self._build_prefix_missed_requests_ids(
                    request_id,
                    pending_missed_requests_ids,
                ),
                current_selection=parted_selection,
            )
            mask = self._obs_builder.create_action_mask(request_id)
            action, _ = self._rl_model.predict(obs, action_masks=mask, deterministic=True)
            mutated_individual[request_id] = self._action_to_truck_id(action)
            if request_id in pending_missed_requests_ids:
                pending_missed_requests_ids.remove(request_id)

        return mutated_individual

    def _select_mutation_result(
        self,
        original_individual: Genome,
        mutated_individual: Genome,
    ) -> Genome:
        return mutated_individual

    def _mutation(self, individual: Genome) -> Genome:
        original_individual = individual.copy()
        if random.random() >= self._mutation_rate:
            return original_individual

        missed_requests_ids, _, _ = self._simulator.run(tuple(original_individual))
        if not missed_requests_ids:
            return original_individual

        mutated_individual = self._mutate_missed_requests_sequentially(
            original_individual,
            missed_requests_ids,
        )
        return self._select_mutation_result(original_individual, mutated_individual)


class GeneticAlgoWithRlMutator(_RlMutatorMixin, GeneticAlgoWithRLInit):

    def _create_initial_population(self) -> list[Genome]:
        return GeneticAlgoSimple._create_initial_population(self)


class GeneticAlgoWithRlTailMutator(_RlTailMutatorMixin, GeneticAlgoWithRLInit):

    def _create_initial_population(self) -> list[Genome]:
        return GeneticAlgoSimple._create_initial_population(self)


class GeneticAlgoWithInitAndRlMutator(_RlMutatorMixin, GeneticAlgoWithRLInit):
    pass


class GeneticAlgoWithInitAndRlTailMutator(_RlTailMutatorMixin, GeneticAlgoWithRLInit):
    pass


class GeneticAlgoWithRlMissedRequestsMutator(_RlMissedRequestsSequentialMutatorMixin, GeneticAlgoWithRLInit):

    def _create_initial_population(self) -> list[Genome]:
        return GeneticAlgoSimple._create_initial_population(self)


class GeneticAlgoWithInitAndRlMissedRequestsMutator(_RlMissedRequestsSequentialMutatorMixin, GeneticAlgoWithRLInit):
    pass


class _RlMissedRequestsAcceptedByFitnessMutatorMixin(_RlMissedRequestsSequentialMutatorMixin):

    def _select_mutation_result(
        self,
        original_individual: Genome,
        mutated_individual: Genome,
    ) -> Genome:
        original_missed_requests_ids, _, _ = self._simulator.run(tuple(original_individual))
        mutated_missed_requests_ids, _, _ = self._simulator.run(tuple(mutated_individual))
        original_served_requests = self._genome_length - len(original_missed_requests_ids)
        mutated_served_requests = self._genome_length - len(mutated_missed_requests_ids)
        return mutated_individual if mutated_served_requests > original_served_requests else original_individual


class GeneticAlgoWithRlMissedRequestsAcceptedByFitnessMutator(
    _RlMissedRequestsAcceptedByFitnessMutatorMixin,
    GeneticAlgoWithRLInit,
):

    def _create_initial_population(self) -> list[Genome]:
        return GeneticAlgoSimple._create_initial_population(self)


class GeneticAlgoWithInitAndRlMissedRequestsAcceptedByFitnessMutator(
    _RlMissedRequestsAcceptedByFitnessMutatorMixin,
    GeneticAlgoWithRLInit,
):
    pass
