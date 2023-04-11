import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Sequence, Hashable, Tuple, Optional, List

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.log import default_log
from golem.core.optimisers.opt_history_objects.individual import Individual

ObsType = Graph
ActType = Hashable


class MutationAgentTypeEnum(Enum):
    default = 'default'
    random = 'random'
    bandit = 'bandit'
    contextual_bandit = 'contextual_bandit'


class ExperienceBuffer:
    """Buffer for learning experience of ``OperatorAgent``.
    Keeps (State, Action, Reward) lists until retrieval."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._observations = []
        self._actions = []
        self._rewards = []
        self._prev_pop = set()
        self._next_pop = set()

    def collect_results(self, results: Sequence[Individual]):
        for ind in results:
            self.collect_result(ind)

    def collect_result(self, result: Individual):
        if result.uid in self._prev_pop:
            return
        if not result.parent_operator or result.parent_operator.type_ != 'mutation':
            return
        self._next_pop.add(result.uid)
        obs = result.graph
        # TODO: store this action in correct format in ParentOperators.operators
        action = result.parent_operator.operators[0]
        prev_fitness = result.parent_operator.parent_individuals[0].fitness.value
        # we're minimising the fitness, that's why less is better
        reward = prev_fitness - result.fitness.value if prev_fitness is not None else 0.
        self.collect_experience(obs, action, reward)

    def collect_experience(self, obs: ObsType, action: ActType, reward: float):
        self._observations.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)

    def retrieve_experience(self) -> Tuple[List[ActType], List[float]]:
        """Get all collected experience and clear the experience buffer."""
        actions, rewards = self._actions, self._rewards
        next_pop = self._next_pop
        self.reset()
        self._prev_pop = next_pop
        return actions, rewards


class OperatorAgent(ABC):
    @abstractmethod
    def partial_fit(self, experience: ExperienceBuffer):
        raise NotImplementedError()

    @abstractmethod
    def choose_action(self, obs: Optional[ObsType]) -> ActType:
        raise NotImplementedError()

    @abstractmethod
    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        raise NotImplementedError()

    @abstractmethod
    def get_action_probs(self, obs: Optional[ObsType]) -> Sequence[float]:
        raise NotImplementedError()

    @abstractmethod
    def get_action_values(self, obs: Optional[ObsType]) -> Sequence[float]:
        raise NotImplementedError()


class RandomAgent(OperatorAgent):
    def __init__(self,
                 actions: Sequence[ActType],
                 probs: Optional[Sequence[float]] = None,
                 enable_logging: bool = True):
        self.actions = list(actions)
        self._probs = [1./ len(actions)] * len(actions)
        self._enable_logging = enable_logging
        self._log = default_log(self)

    def choose_action(self, obs: ObsType) -> ActType:
        action = np.random.choice(self.actions, p=self.get_action_probs(obs))
        return action

    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        subject_nodes = random.sample(graph.nodes, k=num_nodes)
        return subject_nodes[0] if num_nodes == 1 else subject_nodes

    def partial_fit(self, experience: ExperienceBuffer):
        actions, rewards = experience.retrieve_experience()
        self._dbg_log(actions, rewards)

    def get_action_probs(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        return self._probs

    def get_action_values(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        return self._probs

    def _dbg_log(self, actions, rewards):
        if self._enable_logging:
            rr = np.array(rewards).round(4)
            nonzero = rr[rr.nonzero()]
            msg = f'len={len(rr)} nonzero={len(nonzero)} '
            if len(nonzero) > 0:
                msg += (f'avg={nonzero.mean()} std={nonzero.std()} '
                        f'min={nonzero.min()} max={nonzero.max()} ')
            self._log.info(msg)
            self._log.info(f'actions/rewards: {list(zip(actions, rr))}')
