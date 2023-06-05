from typing import Sequence

from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.neural_mab import NeuralMAB
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
from golem.core.optimisers.adaptive.operator_agent import ActType


class NeuralContextualMultiArmedBanditAgent(ContextualMultiArmedBanditAgent):
    """ Neural Contextual Multi-Armed bandit. Observations can be encoded with the use of Neural Networks,
    but still there are some restrictions to guarantee convergence. """
    def __init__(self,
                 actions: Sequence[ActType],
                 n_jobs: int = 1,
                 context_agent_type: ContextAgentTypeEnum = ContextAgentTypeEnum.nodes_num,
                 enable_logging: bool = True):
        super().__init__(actions=actions, n_jobs=n_jobs,
                         enable_logging=enable_logging, context_agent_type=context_agent_type)
        self._agent = NeuralMAB(arms=self._indices,
                                n_jobs=n_jobs)
