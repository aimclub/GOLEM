from copy import deepcopy
from typing import Optional, Dict, Any, Iterable

import networkx as nx

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.dag.graph_node import GraphNode
from golem.core.optimisers.graph import OptGraph, OptNode


class BaseNetworkxAdapter(BaseOptimizationAdapter[nx.DiGraph]):
    """Base class for adaptation of networkx.DiGraph to optimization graph.
    Allows to use NetworkX directed graphs with our optimizers.

    For custom networkx adapters overwrite methods responsible for
    transformation of single nodes (`_node_adapt` & `_node_restore`).
    """

    def __init__(self):
        super().__init__(base_graph_class=nx.DiGraph)

    def _node_restore(self, node: GraphNode) -> Dict:
        """Transforms GraphNode to dict of NetworkX node attributes.
        Override for custom behavior."""
        parameters = {}
        if hasattr(node, 'parameters'):
            parameters = deepcopy(node.parameters)

        if node.name:
            parameters['name'] = node.name

        return parameters

    def _node_adapt(self, data: Dict) -> OptNode:
        """Transforms a dict of NetworkX node attributes to GraphNode.
        Override for custom behavior."""
        data = deepcopy(data)
        name = data.pop('name', None)
        return OptNode(content={'name': name, 'params': data})

    def _adapt(self, adaptee: nx.DiGraph) -> OptGraph:
        mapped_nodes = {}

        def map_predecessors(node_id) -> Iterable[OptNode]:
            for pred_id in adaptee.predecessors(node_id):
                yield mapped_nodes[pred_id]

        # map nodes
        for node_id, node_data in adaptee.nodes.items():
            # transform node
            node = self._node_adapt(node_data)
            mapped_nodes[node_id] = node

        # map parent nodes
        for node_id, node in mapped_nodes.items():
            # append its parent edges
            node.nodes_from = map_predecessors(node_id)

        return OptGraph(mapped_nodes.values())

    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        nx_node_data = {}

        # add nodes
        for node in opt_graph.nodes:
            nx_node_data[node.uid] = self._node_restore(node)
            nx_graph.add_node(node.uid)

        # add edges
        for node in opt_graph.nodes:
            for parent in node.nodes_from:
                nx_graph.add_edge(parent.uid, node.uid)

        # add nodes ad labels
        nx.set_node_attributes(nx_graph, nx_node_data)

        return nx_graph


_NX_NODE_KEY = 'data'


class DumbNetworkxAdapter(BaseNetworkxAdapter):
    """Simple version of networkx adapter that just stores
    `OptNode` as an attribute of NetworkX graph node."""

    def _node_restore(self, node: GraphNode) -> Dict:
        return {_NX_NODE_KEY: node}

    def _node_adapt(self, data: Dict) -> OptNode:
        return data[_NX_NODE_KEY]


class BanditNetworkxAdapter(BaseNetworkxAdapter):
    """ Classic networkx adapter with nodes indexes in names instead of uids.
    It is needed since some frameworks (e.g. karateclub) have asserts in which node
    names should consist only of its indexes.
    """
    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        nx_node_data = {}

        # add nodes
        for node in opt_graph.nodes:
            nx_node_data[node.uid] = self._node_restore(node)
            nx_graph.add_node(opt_graph.nodes.index(node))

        # add edges
        for node in opt_graph.nodes:
            for parent in node.nodes_from:
                nx_graph.add_edge(opt_graph.nodes.index(parent), opt_graph.nodes.index(node))

        # add nodes ad labels
        nx.set_node_attributes(nx_graph, nx_node_data)

        return nx_graph
