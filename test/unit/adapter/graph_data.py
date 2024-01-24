import networkx as nx

from golem.core.adapter.nx_adapter import DumbNetworkxAdapter, BaseNetworkxAdapter
from test.unit.mocks.common_mocks import MockNode, MockDomainStructure, MockAdapter


def get_adapters():
    return [MockAdapter(),
            BaseNetworkxAdapter(),
            DumbNetworkxAdapter()]


def get_graphs():
    node_a = MockNode('operation_a')
    node_b = MockNode('operation_a', nodes_from=[node_a])
    node_c = MockNode('operation_a', nodes_from=[node_b, node_a])
    skip_connection_structure = MockDomainStructure([node_c])

    one_node_graph = MockDomainStructure([node_a])

    node_d = MockNode('operation_a', nodes_from=[node_b])
    linear_graph = MockDomainStructure([node_d])

    node_f = MockNode('operation_f')
    node_g = MockNode('operation_g', nodes_from=[node_b, node_f])
    branching_structure = MockDomainStructure([node_g])

    node_k = MockNode('operation_k', nodes_from=[node_f])
    node_m = MockNode('operation_m', nodes_from=[node_b, node_k])
    node_r = MockNode('operation_r', nodes_from=[node_m])
    branching_structure2 = MockDomainStructure([node_r])

    return [one_node_graph, linear_graph,
            branching_structure, branching_structure2,
            skip_connection_structure]


def get_optgraphs():
    adapter = MockAdapter()
    return [adapter.adapt(g) for g in get_graphs()]


def graph_with_custom_parameters(alpha_value):
    node_a = MockNode('a')
    node_b = MockNode('b')
    node_c = MockNode('c', nodes_from=[node_a])
    node_d = MockNode('d', nodes_from=[node_b])
    node_final = MockNode('e', nodes_from=[node_c, node_d])
    node_final.content['params'] = {'alpha': alpha_value}
    graph = MockDomainStructure([node_final])

    return graph


def networkx_graph_with_parameters(alpha_value):
    graph = nx.DiGraph()
    graph.add_node(0, name='a')
    graph.add_node(1, name='b')
    graph.add_node(2, name='c', alpha=alpha_value)
    graph.add_edges_from([(0, 2), (1, 2)])
    return graph


def get_complex_graph():
    node_a = MockNode('a')
    node_b = MockNode('b', nodes_from=[node_a])
    node_c = MockNode('c', nodes_from=[node_b])
    node_d = MockNode('d', nodes_from=[node_b, node_c])
    node_e = MockNode('e', nodes_from=[node_d])
    node_final = MockNode('a', nodes_from=[node_c, node_e])
    graph = MockDomainStructure([node_final])
    return graph
