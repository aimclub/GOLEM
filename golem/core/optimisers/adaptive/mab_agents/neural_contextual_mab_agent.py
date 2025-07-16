from typing import Sequence, List

from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.neural_mab import NeuralMAB
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
from golem.core.optimisers.adaptive.common_types import ActType


def get_context_size(context_agent_type: ContextAgentTypeEnum, available_operations: List[str]) -> int:
    """
    Returns the context size based on the context agent type and available operations.
    """
    if context_agent_type == ContextAgentTypeEnum.nodes_num:
        return 1  # returns 1 as nodes_num context represents the number of nodes in the graph
    elif context_agent_type == ContextAgentTypeEnum.operations_quantity:
        return len(available_operations)
    elif context_agent_type == ContextAgentTypeEnum.adjacency_matrix:
        return len(available_operations) ** 2
    elif context_agent_type == ContextAgentTypeEnum.feather_graph:
        return 500  # embedding size for FeatherGraph with default parameters
    elif context_agent_type == ContextAgentTypeEnum.labeled_edges:
        return 100  # fixed context size; can be edited in labeled_edges function
    else:
        raise ValueError(f"Context agent type {context_agent_type} is not supported for NeuralContextualMABAgent.")


class NeuralContextualMultiArmedBanditAgent(ContextualMultiArmedBanditAgent):
    """ Neural Contextual Multi-Armed bandit.
    Observations can be encoded with the use of Neural Networks, but still there are some restrictions
    to guarantee convergence. """

    def __init__(self,
                 actions: Sequence[ActType],
                 context_agent_type: ContextAgentTypeEnum,
                 available_operations: List[str],
                 n_jobs: int = 1,
                 enable_logging: bool = True,
                 decaying_factor: float = 1.0):
        super().__init__(actions=actions, n_jobs=n_jobs,
                         context_agent_type=context_agent_type, available_operations=available_operations,
                         enable_logging=enable_logging, decaying_factor=decaying_factor)
        context_size = get_context_size(context_agent_type, available_operations)

        self._agent = NeuralMAB(arms=self._indices,
                                context_size=context_size,
                                n_jobs=n_jobs)
