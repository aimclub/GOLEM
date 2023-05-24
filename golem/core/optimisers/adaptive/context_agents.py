from enum import Enum

from typing import List, Callable

from karateclub import FeatherGraph

from golem.core.adapter.nx_adapter import BanditNetworkxAdapter
from golem.core.dag.graph import Graph


def feather_graph(graph: Graph) -> List[float]:
    """ Returns embedding based on an implementation of `"FEATHER-G" <https://arxiv.org/abs/2005.07959>`_.
    The procedure uses characteristic functions of node features with random walk weights to describe
    node neighborhoods. These node level features are pooled by mean pooling to
    create graph level statistics. """
    descriptor = FeatherGraph()
    nx_graph = BanditNetworkxAdapter().restore(graph)
    descriptor.fit([nx_graph])
    return descriptor.get_embedding()[:20]


class ContextAgentTypeEnum(Enum):
    feather_graph = 'feather_graph'


class ContextAgentsRepository:
    _agents_implementations = {
        ContextAgentTypeEnum.feather_graph: feather_graph
    }

    @staticmethod
    def agent_class_by_id(agent_id: ContextAgentTypeEnum) -> Callable:
        return ContextAgentsRepository._agents_implementations[agent_id]

