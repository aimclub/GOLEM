import pytest

from golem.core.optimisers.adaptive.context_agents import ContextAgentsRepository, ContextAgentTypeEnum
from golem.core.optimisers.graph import OptNode, OptGraph


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
                         [(ContextAgentTypeEnum.operations_encoding,
                           [1, 2, 1, 1, 1]),
                          (ContextAgentTypeEnum.labeled_edges,
                           [3, 4, 1, 4, 2, 4, 0, 3, 1, 3]),
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
    encoding = context_agent(obs=graph, available_operations=['1', '2', '3', '4', '5'])
    assert encoding == result_encoding
