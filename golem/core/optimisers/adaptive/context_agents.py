from enum import Enum

from typing import List, Callable, Any

import networkx as nx
from karateclub import FeatherGraph

from golem.core.adapter.nx_adapter import BanditNetworkxAdapter, BaseNetworkxAdapter
from golem.core.optimisers.opt_history_objects.individual import Individual


def adapter_func(func):
    """ Decorator function to adapt observation to networkx graphs. """
    def wrapper(obs):
        nx_graph = BanditNetworkxAdapter().restore(obs)
        embedding = func(nx_graph)
        return embedding
    return wrapper


def encode_operations(operations: List[str], available_operations: List[str], mode: str = 'label'):
    """ Encoding of operations.
    :param operations: operations to encode
    :param available_operations: list of all available operations
    :param mode: mode of encoding. Available type: 'ohe' and 'label', default -- 'label'
    """
    encoded = []
    for operation in operations:
        if mode == 'label':
            encoding = available_operations.index(operation)
        else:
            encoding = [0]*len(available_operations)
            encoding[available_operations.index(operation)] = 1
        encoded.append(encoding)
    return encoded


@adapter_func
def feather_graph(obs: Any, available_operations: List[str]) -> List[float]:
    """ Returns embedding based on an implementation of `"FEATHER-G" <https://arxiv.org/abs/2005.07959>`_.
    The procedure uses characteristic functions of node features with random walk weights to describe
    node neighborhoods. These node level features are pooled by mean pooling to
    create graph level statistics. """
    descriptor = FeatherGraph()
    descriptor.fit([obs])
    emb = descriptor.get_embedding().reshape(-1, 1)
    embd = [i[0] for i in emb]
    return embd


def nodes_num(obs: Any, available_operations: List[str]) -> List[int]:
    """ Returns number of nodes in graph. """
    if isinstance(obs, Individual):
        return [len(obs.graph.nodes)]
    else:
        return [len(obs.nodes)]


def labeled_edges(obs: Any, available_operations: List[str]) -> List[int]:
    """ Encodes graph with its edges with nodes labels. """
    operations = []
    for node in obs.graph.nodes:
        for node_ in node.nodes_from:
            operations.append(node_.name)
            operations.append(node.name)
    return encode_operations(operations=operations, available_operations=available_operations)


def operations_encoding(obs: Any, available_operations: List[str]) -> List[int]:
    """ Encodes graphs as vectors with quantity of each operation. """
    encoding = [0] * len(available_operations)
    if isinstance(obs, Individual):
        graph = obs.graph
    else:
        graph = obs
    for node in graph.nodes:
        encoding[available_operations.index(node.name)] += 1
    return encoding


class ContextAgentTypeEnum(Enum):
    feather_graph = 'feather_graph'
    nodes_num = 'nodes_num'
    labeled_edges = 'labeled_edges'
    operations_encoding = 'operations_encoding'


class ContextAgentsRepository:
    """ Repository of functions to encode observations. """
    _agents_implementations = {
        ContextAgentTypeEnum.feather_graph: feather_graph,
        ContextAgentTypeEnum.nodes_num: nodes_num,
        ContextAgentTypeEnum.labeled_edges: labeled_edges,
        ContextAgentTypeEnum.operations_encoding: operations_encoding
    }

    @staticmethod
    def agent_class_by_id(agent_id: ContextAgentTypeEnum) -> Callable:
        return ContextAgentsRepository._agents_implementations[agent_id]
