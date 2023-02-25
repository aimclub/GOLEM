import pytest

from golem.core.dag.verification_rules import has_no_cycle, has_no_isolated_nodes, ERROR_PREFIX, \
    has_no_self_cycled_nodes, has_no_isolated_components
from test.unit.mocks.common_mocks import MockNode, MockDomainStructure
from test.unit.utils import graph_first


def graph_with_cycle():
    first = MockNode('a')
    second = MockNode('b', nodes_from=[first])
    third = MockNode('c', nodes_from=[second, first])
    second.nodes_from.append(third)
    graph = MockDomainStructure([third])
    return graph


def graph_with_isolated_nodes():
    first = MockNode('a')
    second = MockNode('b', nodes_from=[first])
    third = MockNode('c', nodes_from=[second])
    isolated = MockNode('d', nodes_from=[])
    graph = MockDomainStructure([third, isolated])
    return graph


def graph_with_cycled_node():
    first = MockNode('a')
    second = MockNode('b', nodes_from=[first])
    second.nodes_from.append(second)
    graph = MockDomainStructure([first, second])
    return graph


def graph_with_isolated_components():
    first = MockNode('a')
    second = MockNode('b', nodes_from=[first])
    third = MockNode('c', nodes_from=[])
    fourth = MockNode('d', nodes_from=[third])
    graph = MockDomainStructure([second, fourth])
    return graph


def test_graph_with_cycle_raise_exception():
    graph = graph_with_cycle()
    with pytest.raises(Exception) as exc:
        assert has_no_cycle(graph)
    assert str(exc.value) == f'{ERROR_PREFIX} Graph has cycles'


def test_graph_without_cycles_correct():
    graph = graph_first()

    assert has_no_cycle(graph)


def test_graph_with_isolated_nodes_raise_exception():
    graph = graph_with_isolated_nodes()
    with pytest.raises(ValueError) as exc:
        assert has_no_isolated_nodes(graph)
    assert str(exc.value) == f'{ERROR_PREFIX} Graph has isolated nodes'


def test_graph_with_self_cycled_nodes_raise_exception():
    graph = graph_with_cycled_node()
    with pytest.raises(Exception) as exc:
        assert has_no_self_cycled_nodes(graph)
    assert str(exc.value) == f'{ERROR_PREFIX} Graph has self-cycled nodes'


def test_graph_with_isolated_components_raise_exception():
    graph = graph_with_isolated_components()
    with pytest.raises(Exception) as exc:
        assert has_no_isolated_components(graph)
    assert str(exc.value) == f'{ERROR_PREFIX} Graph has isolated components'

