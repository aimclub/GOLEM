import datetime

import pytest

from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.random_graph_factory import DefaultRandomOptGraphFactory


@pytest.mark.parametrize('max_depth', [1, 3, 5])
def test_gp_composer_random_graph_generation_looping(max_depth):
    """ Test checks DefaultRandomOptGraphFactory valid generation. """
    available_node_types = ['a', 'b', 'c', 'd', 'e']
    requirements = GraphRequirements(
        timeout=datetime.timedelta(seconds=300),
        max_depth=max_depth,
        max_arity=2,
        num_of_generations=5)
    verifier = GraphVerifier(DEFAULT_DAG_RULES)
    opt_node_factory = DefaultOptNodeFactory(available_node_types)
    random_graph_factory = DefaultRandomOptGraphFactory(verifier, opt_node_factory)

    graphs = [random_graph_factory(requirements, max_depth=None) for _ in range(4)]
    for graph in graphs:
        for node in graph.nodes:
            assert node.content['name'] in available_node_types
        assert verifier(graph) is True
        assert graph.depth <= requirements.max_depth
