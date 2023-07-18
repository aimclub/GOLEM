from typing import List

from golem.core.optimisers.adaptive.operator_agent import ObsType


class RewardAgent:
    """ Agent to process raw fitness values. """
    def __init__(self, window_size: int = 1, decaying_factor: float = 1.):
        self._window_size = window_size
        self._decaying_factor = decaying_factor

    def get_rewards_for_arms(self, obs: List[ObsType], arms: List[int]) -> List[float]:
        decay_values = self.get_decay_values(obs, arms)
        frr = self.get_fitness_rank_rate(decay_values)
        frr_values = []
        for arm in arms:
            frr_values.append(frr[arms.index(arm)])
        return frr_values

    def get_decay_values(self, rewards: List[ObsType], arms: List[int]) -> List[float]:
        decays = dict.fromkeys(set(arms), 0.0)
        for i, reward in enumerate(rewards):
            decays[arms[i]] += reward
        decays.update((key, value * self._decaying_factor) for key, value in decays.items())
        return list(decays.values())

    @staticmethod
    def get_fitness_rank_rate(decay_values: List[float]) -> List[float]:
        total_decay_sum = abs(sum(decay_values))
        return [decay / total_decay_sum for decay in decay_values] if total_decay_sum != 0 else [0.]
