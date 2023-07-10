import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Sequence, Hashable, Tuple, Optional, List, Iterable

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.log import default_log
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory

ObsType = Union[Individual, Graph]
ActType = Hashable
# Trajectory step includes (past observation, action, reward)
TrajectoryStep = Tuple[Individual, ActType, float]
# Trajectory is a sequence of applied mutations and received rewards
GraphTrajectory = Sequence[TrajectoryStep]


class MutationAgentTypeEnum(Enum):
    default = 'default'
    random = 'random'
    bandit = 'bandit'
    contextual_bandit = 'contextual_bandit'
    neural_bandit = 'neural_bandit'


class ExperienceBuffer:
    """Buffer for learning experience of ``OperatorAgent``.
    Keeps (State, Action, Reward) lists until retrieval."""

    @staticmethod
    def from_history(history: OptHistory) -> 'ExperienceBuffer':
        exp = ExperienceBuffer()
        exp.collect_history(history)
        return exp

    def __init__(self, inds=None, actions=None, rewards=None):
        self.reset(inds, actions, rewards)

    def reset(self, inds=None, actions=None, rewards=None):
        if not (len(inds) == len(actions) == len(rewards)):
            raise ValueError('lengths of buffers do not mathch')
        self._individuals = inds or []
        self._actions = actions or []
        self._rewards = rewards or []
        self._prev_pop = set()
        self._next_pop = set()

    @staticmethod
    def unroll_action_step(result: Individual) -> TrajectoryStep:
        """Unrolls individual's history to get its source individual, action and resulting reward."""
        if not result.parent_operator or result.parent_operator.type_ != 'mutation':
            return None, None, np.nan
        source_ind = result.parent_operator.parent_individuals[0]
        action = result.parent_operator.operators[0]
        # we're minimising the fitness, that's why less is better
        reward = source_ind.fitness.value - result.fitness.value if source_ind.fitness is not None else 0.
        return source_ind, action, reward

    @staticmethod
    def unroll_trajectories(history: OptHistory) -> List[GraphTrajectory]:
        """Iterates through history and find continuous sequences of applied operator actions."""
        trajectories = []
        seen_uids = set()
        for terminal_individual in history.final_choices:
            trajectory = []
            next_ind = terminal_individual
            while True:
                seen_uids.add(next_ind.uid)
                source_ind, action, reward = ExperienceBuffer.unroll_action_step(next_ind)
                if source_ind is None or source_ind.uid in seen_uids:
                    break
                # prepend step to keep historical direction
                trajectory.insert(0, (source_ind, action, reward))
                next_ind = source_ind
            trajectories.append(trajectory)
        return trajectories

    def __len__(self):
        return len(self._individuals)

    def __str__(self):
        return f'{self.__class__.__name__}({len(self)})'

    def collect_history(self, history: OptHistory):
        seen = set()
        # We don't need the initial assumptions, as they have no parent operators, hence [1:]
        for generation in history.generations[1:]:
            for ind in generation:
                if ind.uid not in seen:
                    seen.add(ind.uid)
                    self.collect_result(ind)

    def collect_results(self, results: Iterable[Individual]):
        for ind in results:
            self.collect_result(ind)

    def collect_result(self, result: Individual):
        if result.uid in self._prev_pop:
            # avoid collecting results from indiiduals that didn't change
            return
        self._next_pop.add(result.uid)

        source_ind, action, reward = self.unroll_action_step(result)
        if action is None:
            return
        self.collect_experience(source_ind, action, reward)

    def collect_experience(self, obs: Individual, action: ActType, reward: float):
        self._individuals.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)

    def retrieve_experience(self, as_graphs: bool = True) -> Tuple[List[ObsType], List[ActType], List[float]]:
        """Get all collected experience and clear the experience buffer.
        Args:
            as_graphs: if True (by default) returns observations as graphs, otherwise as individuals.
        Return:
             Unzipped trajectories (tuple of lists of observations, actions, rewards).
        """
        individuals, actions, rewards = self._individuals, self._actions, self._rewards
        observations = [ind.graph for ind in individuals] if as_graphs else individuals
        next_pop = self._next_pop
        self.reset()
        self._prev_pop = next_pop
        return observations, actions, rewards

    def retrieve_trajectories(self) -> GraphTrajectory:
        """Same as `retrieve_experience` but in the form of zipped trajectories that consist from steps."""
        trajectories = list(zip(self.retrieve_experience(as_graphs=False)))
        return trajectories

    def split(self, ratio: float = 0.8, shuffle: bool = False
              ) -> Tuple['ExperienceBuffer', 'ExperienceBuffer']:
        """Splits buffer in 2 parts, useful for train/validation split."""
        mask_train = np.full_like(self._individuals, True, dtype=bool)
        num_train = int(len(self._individuals) * ratio)
        mask_train[-num_train:] = False
        if shuffle:
            np.random.default_rng().shuffle(mask_train)
        buffer_train = ExperienceBuffer(inds=self._individuals[mask_train].aslist(),
                                        actions=self._actions[mask_train].aslist(),
                                        rewards=self._rewards[mask_train].aslist())
        buffer_val = ExperienceBuffer(inds=self._individuals[~mask_train].aslist(),
                                      actions=self._actions[~mask_train].aslist(),
                                      rewards=self._rewards[~mask_train].aslist())
        return buffer_train, buffer_val


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
                msg += (f'avg={nonzero.mean()} std={nonzero.std()} '
                        f'min={nonzero.min()} max={nonzero.max()} ')

            self._log.info(msg)
            self._log.info(f'actions/rewards: {list(zip(actions, rr))}')

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
