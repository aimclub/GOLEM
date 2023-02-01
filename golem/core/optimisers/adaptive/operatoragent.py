import random
from abc import ABC, abstractmethod
from typing import Union, Sequence, Hashable, Any, Dict, Tuple, Optional

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode


ObsType = Graph
ActType = Hashable


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
