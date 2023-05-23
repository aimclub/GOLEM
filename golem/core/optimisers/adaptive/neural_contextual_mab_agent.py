import random
from typing import Sequence, Union, Optional, List, Any

from scipy.special import softmax

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.optimisers.adaptive.neural_mab import NeuralMAB
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum, ContextAgentsRepository
from golem.core.optimisers.adaptive.mab_agent import MultiArmedBanditAgent
from golem.core.optimisers.adaptive.operator_agent import ActType, ExperienceBuffer, ObsType


class NeuralContextualMultiArmedBanditAgent(MultiArmedBanditAgent):
    def __init__(self,
                 actions: Sequence[ActType],
                 n_jobs: int = 1,
                 context_agent: ContextAgentTypeEnum = ContextAgentTypeEnum.feather_graph,
                 enable_logging: bool = True):
        super().__init__(actions=actions, enable_logging=enable_logging)
        self._agent = NeuralMAB(arms=self._indices,
                                n_jobs=n_jobs)
        self._context_agent = context_agent

    def choose_action(self, obs: ObsType) -> ActType:
        obs_emb = self._get_obs_embedding(obs=[obs])
        arm = self._agent.predict(context=obs_emb)
        action = self.actions[arm]
        return action

    def get_action_values(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        if not isinstance(obs, list):
            obs = [obs]
        obs_emb = self._get_obs_embedding(obs=obs)
        prob_dict = self._agent.predict_expectations(context=obs_emb)
        prob_list = [prob_dict[i].item() for i in range(len(prob_dict))]
        return prob_list

    def get_action_probs(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        return softmax(self.get_action_values(obs=obs))

    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        subject_nodes = random.sample(graph.nodes, k=num_nodes)
        return subject_nodes[0] if num_nodes == 1 else subject_nodes

    def partial_fit(self, experience: ExperienceBuffer):
        """Continues learning of underlying agent with new experience."""
        obs, actions, rewards = experience.retrieve_experience()
        self._dbg_log(obs, actions, rewards)
        arms = [self._arm_by_action[action] for action in actions]
        obs_embs = self._get_obs_embedding(obs=obs)
        self._agent.partial_fit(decisions=arms, rewards=rewards, contexts=obs_embs)

    def _get_obs_embedding(self, obs: List[ObsType]) -> List[Any]:
        context_agent = ContextAgentsRepository.agent_class_by_id(self._context_agent)
        obs_contexts = []
        for ob in obs:
            obs_contexts.append(context_agent(ob))
        return obs_contexts
