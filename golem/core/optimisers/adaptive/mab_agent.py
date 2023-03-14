import random
from typing import Union, Sequence, Optional

import numpy as np
from mabwiser.mab import MAB, LearningPolicy

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.log import default_log
from golem.core.optimisers.adaptive.operatoragent import OperatorAgent, ActType, ObsType, ExperienceBuffer


class MultiArmedBanditAgent(OperatorAgent):
    def __init__(self,
                 actions: Sequence[ActType],
                 enable_logging: bool = True):
        self.actions = list(actions)
        self._indices = list(range(len(actions)))
        self._arm_by_action = dict(zip(actions, self._indices))
        self._agent = MAB(arms=self._indices,
                          learning_policy=LearningPolicy.EpsilonGreedy(0.15),
                          n_jobs=1)
        self._initial_fit()
        self._enable_logging = enable_logging
        self._log = default_log(self)

    def _initial_fit(self):
        n = len(self.actions)
        uniform_rewards = [1. / n] * n
        self._agent.fit(decisions=self._indices, rewards=uniform_rewards)

    def _action_values(self) -> Sequence[float]:
        return self._agent.predict_expectations()

    def choose_action(self, obs: ObsType) -> ActType:
        arm = self._agent.predict()
        action = self.actions[arm]
        # if self._enable_logging:
        #     self._log.info(f'action={action} expectations={self._action_values()}')
        return action

    def get_action_probs(self, obs: ObsType) -> Optional[Sequence[float]]:
        # TODO: normalise expectations in softmax?
        return self._action_values()

    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        subject_nodes = random.sample(graph.nodes, k=num_nodes)
        return subject_nodes[0] if num_nodes == 1 else subject_nodes

    def partial_fit(self, experience: ExperienceBuffer):
        actions, rewards = experience.get_experience()
        self._dbg_log(actions, rewards)
        arms = [self._arm_by_action[action] for action in actions]
        self._agent.partial_fit(decisions=arms, rewards=rewards)

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

            self._log.info(f'expectations={self._action_values()}')
