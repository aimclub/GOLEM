import pytest

from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from test.unit.adapter.mock_adapter import MockNode, MockDomainStructure, MockAdapter


def get_valid_graph():
    first_node = MockNode(content='n1')
    second_node = MockNode(content='n2', nodes_from=[first_node])
    third_node = MockNode(content='n3', nodes_from=[second_node])

    graph = MockDomainStructure([third_node])
    adapter = MockAdapter()
    opt_graph = adapter.adapt(graph)
    return opt_graph, graph, adapter


@pytest.mark.parametrize('rule', DEFAULT_DAG_RULES)
def test_adapt_verification_rules_dag(rule):
    """Test that dag verification rules behave as expected with new adapter.
    They accept any graphs, so the new adapter must see them as native
    and shouldn't change them on the call to adapt."""

    opt_graph, graph, adapter = get_valid_graph()

    # sanity check
    assert rule(graph)

    adapted_rule = adapter.adapt_func(rule)

    assert adapted_rule(opt_graph)
    assert id(rule) == id(adapted_rule)
