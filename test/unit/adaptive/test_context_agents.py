import pytest

from golem.core.adapter.nx_adapter import BanditNetworkxAdapter
from golem.core.optimisers.adaptive.context_agents import ContextAgentsRepository, ContextAgentTypeEnum
from golem.core.optimisers.graph import OptNode, OptGraph
from test.unit.adaptive.test_contextual_mab_agents import get_large_and_small_graphs

adapter = BanditNetworkxAdapter()
available_operations = ['1', '2', '3', '4', '5']


def get_opt_graph():
    node = OptNode('1')
    node2 = OptNode('2')
    node4 = OptNode('4', nodes_from=[node, node2])
    node2_2 = OptNode('2')
    node3 = OptNode('3')
    node5 = OptNode('5', nodes_from=[node4, node2_2, node3])
    graph = OptGraph(node5)
    return graph


@pytest.mark.parametrize('context_agent_enum, result_encoding',
                         [(ContextAgentTypeEnum.operations_quantity,
                           [1, 2, 1, 1, 1]),
                          (ContextAgentTypeEnum.labeled_edges,
                           [3, 4, 1, 4, 2, 4, 0, 3, 1, 3] + [5] * 90),
                          (ContextAgentTypeEnum.adjacency_matrix,
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]),
                          (ContextAgentTypeEnum.nodes_num,
                           [6])
                          ])
def test_operations_encoding(context_agent_enum, result_encoding):
    """ Checks the correctness of context agents output. """
    graph = get_opt_graph()
    repo = ContextAgentsRepository()
    context_agent = repo.agent_class_by_id(context_agent_enum)
    encoding = context_agent(obs=graph, available_operations=available_operations)
    assert encoding == result_encoding


@pytest.mark.parametrize('context_agent, context_size',
                         [(ContextAgentTypeEnum.operations_quantity, len(available_operations)),
                          (ContextAgentTypeEnum.adjacency_matrix, len(available_operations) ** 2),
                          (ContextAgentTypeEnum.feather_graph, 500),
                          (ContextAgentTypeEnum.labeled_edges, 100),
                          (ContextAgentTypeEnum.nodes_num, 1)])
def test_context_size(context_agent, context_size):
    """ Checks the correctness of context size, returned by context agents. """

    small_graph, large_graph = get_large_and_small_graphs()

    context_small_graph = ContextAgentsRepository.agent_class_by_id(context_agent)(small_graph, available_operations)
    context_large_graph = ContextAgentsRepository.agent_class_by_id(context_agent)(large_graph, available_operations)

    assert len(context_small_graph) == context_size
    assert len(context_large_graph) == context_size
