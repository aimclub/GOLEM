import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Sequence, Optional

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.log import default_log
from golem.core.optimisers.adaptive.common_types import ObsType, ActType
from golem.core.optimisers.adaptive.experience_buffer import ExperienceBuffer


class MutationAgentTypeEnum(Enum):
    default = 'default'
    random = 'random'
    bandit = 'bandit'
    contextual_bandit = 'contextual_bandit'
    neural_bandit = 'neural_bandit'


class OperatorAgent(ABC):
    def __init__(self, actions: Sequence[ActType], enable_logging: bool = True):
        self.actions = list(actions)
        self._enable_logging = enable_logging
        self._log = default_log(self)

    @property
    def available_actions(self) -> Sequence[ActType]:
        return self.actions

    @abstractmethod
    def partial_fit(self, experience: ExperienceBuffer):
        raise NotImplementedError()

    @abstractmethod
    def choose_action(self, obs: Optional[ObsType]) -> ActType:
        raise NotImplementedError()

    @abstractmethod
    def choose_nodes(self, graph: ObsType, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        raise NotImplementedError()

    @abstractmethod
    def get_action_probs(self, obs: Optional[ObsType]) -> Sequence[float]:
        raise NotImplementedError()

    @abstractmethod
    def get_action_values(self, obs: Optional[ObsType]) -> Sequence[float]:
        raise NotImplementedError()

    def _dbg_log(self, obs, actions, rewards):
        if self._enable_logging:
            prec = 4
            rr = np.array(rewards).round(prec)
            nonzero = rr[rr.nonzero()]
            msg = f'len={len(rr)} nonzero={len(nonzero)} '
            if len(nonzero) > 0:
                msg += (f'avg={nonzero.mean():.3f} std={nonzero.std():.3f} '
                        f'min={nonzero.min():.3f} max={nonzero.max():.3f} ')

            def get_name(obj):
                return getattr(obj, '__name__', str(obj))

            self._log.info(msg)
            action_strs = map(get_name, actions)
            self._log.info(f'actions/rewards: {list(zip(action_strs, rr))}')

            action_values = list(map(self.get_action_values, obs))
            action_probs = list(map(self.get_action_probs, obs))
            action_values = np.round(np.mean(action_values, axis=0), prec)
            action_probs = np.round(np.mean(action_probs, axis=0), prec)

            self._log.info(f'exp={action_values} '
                           f'probs={action_probs}')


class RandomAgent(OperatorAgent):
    def __init__(self,
                 actions: Sequence[ActType],
                 probs: Optional[Sequence[float]] = None,
                 enable_logging: bool = True):
        super().__init__(actions, enable_logging)
        self._probs = probs or [1. / len(actions)] * len(actions)

    def choose_action(self, obs: Graph) -> ActType:
        action = np.random.choice(self.actions, p=self.get_action_probs(obs))
        return action

    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        subject_nodes = random.sample(graph.nodes, k=num_nodes)
        return subject_nodes[0] if num_nodes == 1 else subject_nodes

    def partial_fit(self, experience: ExperienceBuffer):
        obs, actions, rewards = experience.retrieve_experience()
        self._dbg_log(obs, actions, rewards)

    def get_action_probs(self, obs: Optional[Graph] = None) -> Sequence[float]:
        return self._probs

    def get_action_values(self, obs: Optional[Graph] = None) -> Sequence[float]:
        return self._probs
