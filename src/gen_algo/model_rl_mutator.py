import random

from src.gen_algo.base import Genome
from src.gen_algo.model_rl_init import GeneticAlgoWithRLInit


class GeneticAlgoWithInitAndRlMutator(GeneticAlgoWithRLInit):

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
