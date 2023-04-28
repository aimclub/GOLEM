import random
from typing import Union, Sequence, Optional

from mabwiser.mab import MAB, LearningPolicy
from scipy.special import softmax

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.optimisers.adaptive.operator_agent import OperatorAgent, ActType, ObsType, ExperienceBuffer


class MultiArmedBanditAgent(OperatorAgent):
    def __init__(self,
                 actions: Sequence[ActType],
                 n_jobs: int = 1,
                 enable_logging: bool = True):
        super().__init__(enable_logging)
        self.actions = list(actions)
        self._indices = list(range(len(actions)))
        self._arm_by_action = dict(zip(actions, self._indices))
        self._agent = MAB(arms=self._indices,
                          learning_policy=LearningPolicy.UCB1(alpha=1.25),
                          n_jobs=n_jobs)
        self._initial_fit()

    def _initial_fit(self):
        n = len(self.actions)
        uniform_rewards = [1. / n] * n
        self._agent.fit(decisions=self._indices, rewards=uniform_rewards)

    def choose_action(self, obs: ObsType) -> ActType:
        arm = self._agent.predict()
        action = self.actions[arm]
        return action

    def get_action_values(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        prob_dict = self._agent.predict_expectations()
        prob_list = [prob_dict[i] for i in range(len(prob_dict))]
        return prob_list

    def get_action_probs(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        return softmax(self.get_action_values())

    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        subject_nodes = random.sample(graph.nodes, k=num_nodes)
        return subject_nodes[0] if num_nodes == 1 else subject_nodes

    def partial_fit(self, experience: ExperienceBuffer):
        """Continues learning of underlying agent with new experience."""
        obs, actions, rewards = experience.retrieve_experience()
        self._dbg_log(obs, actions, rewards)
        arms = [self._arm_by_action[action] for action in actions]
        self._agent.partial_fit(decisions=arms, rewards=rewards)
