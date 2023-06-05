from enum import Enum

from typing import List, Callable, Any

from karateclub import FeatherGraph

from golem.core.adapter.nx_adapter import BanditNetworkxAdapter
from golem.core.optimisers.opt_history_objects.individual import Individual


def feather_graph(obs: Any) -> List[float]:
    """ Returns embedding based on an implementation of `"FEATHER-G" <https://arxiv.org/abs/2005.07959>`_.
    The procedure uses characteristic functions of node features with random walk weights to describe
    node neighborhoods. These node level features are pooled by mean pooling to
    create graph level statistics. """
    descriptor = FeatherGraph()
    nx_graph = BanditNetworkxAdapter().restore(obs)
    descriptor.fit([nx_graph])
    return descriptor.get_embedding()[:20]


def nodes_num(obs: Any) -> int:
    """ Returns number of nodes in graph. """
    if isinstance(obs, Individual):
        return len(obs.graph.nodes)
    else:
        return len(obs.nodes)


class ContextAgentTypeEnum(Enum):
    feather_graph = 'feather_graph'
    nodes_num = 'nodes_num'


class ContextAgentsRepository:
    """ Repository of functions to encode observations. """
    _agents_implementations = {
        ContextAgentTypeEnum.feather_graph: feather_graph,
        ContextAgentTypeEnum.nodes_num: nodes_num
    }

    @staticmethod
    def agent_class_by_id(agent_id: ContextAgentTypeEnum) -> Callable:
        return ContextAgentsRepository._agents_implementations[agent_id]

