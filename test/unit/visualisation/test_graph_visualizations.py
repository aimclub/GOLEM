from pathlib import Path
from typing import Type, Union

import pytest
from golem.core.dag.graph import Graph
from golem.core.dag.graph_delegate import GraphDelegate
from golem.core.dag.linked_graph import LinkedGraph
from golem.core.dag.linked_graph_node import LinkedGraphNode
from golem.core.optimisers.graph import OptGraph, OptNode
from test.unit.mocks.common_mocks import MockDomainStructure, MockNode


@pytest.fixture(scope='module', params=[GraphDelegate, LinkedGraph, MockDomainStructure, OptGraph])
def graph(request):
    graph_type: Union[Type[Graph], Type[MockDomainStructure], Type[OptGraph]] = request.param
    nodes_kwargs = [{'content': {'name': f'n{i + 1}'}} for i in range(4)]
    nodes_kwargs[-1]['nodes_from'] = range(len(nodes_kwargs) - 1)
    if graph_type in [LinkedGraph, GraphDelegate]:
        node_type = LinkedGraphNode
    elif graph_type is MockDomainStructure:
        node_type = MockNode
    else:
        node_type = OptNode
    nodes = []
    for i, kwargs in enumerate(nodes_kwargs):
        if 'nodes_from' in kwargs:
            kwargs['nodes_from'] = [nodes[j] for j in kwargs['nodes_from']]
        else:
            kwargs['nodes_from'] = []
        nodes.append(node_type(**kwargs))
    return graph_type(nodes[-1])


# @pytest.mark.parametrize('engine', ('matplotlib', 'pyvis', 'graphviz'))
def test_graph_show_saving_plots(graph, tmp_path):
    engine = 'matplotlib'
    save_path = Path(tmp_path, engine)
    save_path = save_path.with_suffix('.html') if engine == 'pyvis' else save_path.with_suffix('.png')
    try:
        graph.show(engine=engine, save_path=save_path, dpi=100)
        assert save_path.exists()
    except ImportError:
        assert engine == 'graphviz'
