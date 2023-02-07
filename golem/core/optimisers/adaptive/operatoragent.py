import random
from abc import ABC, abstractmethod
from typing import Union, Sequence, Hashable, Any, Dict, Tuple, Optional

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.opt_history_objects.individual import Individual

ObsType = Graph
ActType = Hashable


class ExperienceBuffer(ABC):
    def __init__(self):
        self.reset()

    def reset(self):
        self._actions = []
        self._old_fitness = []
        self._rewards = []

    def log_actions(self, obs: Sequence[Individual], action: Sequence[ActType]):
        self._actions = list(action)
        self._old_fitness = [ind.fitness.value or 1. for ind in obs]

    def log_result(self, obs: Sequence[Individual]):
        new_fitness = [ind.fitness.value or 0. for ind in obs]
        rewards = np.subtract(new_fitness, self._old_fitness)
        self._rewards = rewards

    def get_experience(self) -> Tuple[Sequence[ActType], Sequence[float]]:
        actions, rewards = self._actions, self._rewards
        self.reset()
        return actions, rewards


class OperatorAgent(ABC):
    @abstractmethod
    def choose_action(self, obs: ObsType) -> ActType:
        raise NotImplementedError()

    @abstractmethod
    def get_action_probs(self, obs: ObsType) -> Sequence[float]:
        raise NotImplementedError()

    @abstractmethod
    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        raise NotImplementedError()

    @abstractmethod
    # def partial_fit(self, actions: Sequence[ActType], rewards: Sequence[float]):
    def partial_fit(self, experience: ExperienceBuffer):
        raise NotImplementedError()


class RandomAgent(OperatorAgent):
    def __init__(self, actions: Sequence[ActType], probs: Optional[Sequence[float]] = None):
        self.actions = list(actions)
        self._probs = probs

    def choose_action(self, obs: ObsType) -> ActType:
        action = np.random.choice(self.actions, p=self.get_action_probs(obs))
        return action

    def get_action_probs(self, obs: ObsType) -> Optional[Sequence[float]]:
        return self._probs

    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        subject_nodes = random.sample(graph.nodes, k=num_nodes)
        return subject_nodes[0] if num_nodes == 1 else subject_nodes

    def partial_fit(self, experience: ExperienceBuffer):
        pass
