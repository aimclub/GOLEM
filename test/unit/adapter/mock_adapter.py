from copy import deepcopy
from typing import Optional, Dict, Any, Union, Iterable

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.dag.graph_utils import map_dag_nodes
from golem.core.dag.linked_graph import LinkedGraph
from golem.core.dag.linked_graph_node import LinkedGraphNode
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import ObjectiveEvaluate
from test.unit.utils import nodes_same


class MockNode(LinkedGraphNode):
    def __init__(self, content: Union[dict, str], nodes_from: Optional[Iterable['MockNode']] = None):
        super().__init__(content, nodes_from=nodes_from)
        self.content['intermadiate_metric'] = None


class MockDomainStructure(LinkedGraph):
    """Mock domain structure for testing adapt/restore logic.
    Represents just a list of nodes."""

    def __init__(self, nodes: Iterable[MockNode], *args, **kwargs):
        super().__init__(nodes, *args, **kwargs)

    def __eq__(self, other):
        return nodes_same(self.nodes, other.nodes)


class MockAdapter(BaseOptimizationAdapter[MockDomainStructure]):
    def __init__(self):
        super().__init__(base_graph_class=MockDomainStructure)

    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> MockDomainStructure:
        nodes = map_dag_nodes(self._opt_to_mock_node, opt_graph.nodes)
        return MockDomainStructure(nodes)

    def _adapt(self, adaptee: MockDomainStructure) -> OptGraph:
        nodes = map_dag_nodes(self._mock_to_opt_node, adaptee.nodes)
        return OptGraph(nodes)

    @staticmethod
    def _mock_to_opt_node(node: MockNode):
        return OptNode(deepcopy(node.content))

    @staticmethod
    def _opt_to_mock_node(node: OptNode):
        return MockNode(deepcopy(node.content))


class MockObjectiveEvaluate(ObjectiveEvaluate):
    def evaluate_intermediate_metrics(self, graph: MockDomainStructure):
        for node in graph.nodes:
            intermediate_graph = MockDomainStructure([node])
            node.content['intermediate_metric'] = self._objective(intermediate_graph)
