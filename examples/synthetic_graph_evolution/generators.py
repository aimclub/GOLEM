from typing import Dict, Callable, Collection, Sequence

import networkx as nx
import numpy as np

from golem.core.adapter.nx_adapter import nx_to_directed

NumNodes = int
DiGraphGenerator = Callable[[NumNodes], nx.DiGraph]


graph_generators: Dict[str, DiGraphGenerator] = {
    'star': lambda n: nx_to_directed(nx.star_graph(n)),
    'grid2d': lambda n: nx.grid_2d_graph(int(np.sqrt(n)), int(np.sqrt(n))),
    '2ring': lambda n: nx_to_directed(nx.circular_ladder_graph(n)),
    'hypercube': lambda n: nx_to_directed(nx.hypercube_graph(int(np.log2(n).round()))),
    'gnp': lambda n: nx_to_directed(nx.gnp_random_graph(n, p=0.08)),
    'line': lambda n: nx_to_directed(nx.path_graph(n, create_using=nx.DiGraph)),
    'tree': lambda n: nx.random_tree(n, create_using=nx.DiGraph),
}


def relabel_nx_graph(graph: nx.Graph, available_names: Collection[str]) -> nx.Graph:
    """Randomly label nodes with 'name' attribute in nx.Graph
    given list of available labels"""
    names = np.random.choice(available_names, size=graph.number_of_nodes())
    attributes = {node_id: {'name': name}
                  for node_id, name in zip(graph.nodes, names)}
    nx.set_node_attributes(graph, attributes)
    return graph


def generate_labeled_graph(kind: str,
                           size: int,
                           node_labels: Sequence[str] = ('x',)):
    """Generate randomly labeled graph of the specified kind and size"""
    generator = graph_generators[kind]
    graph = generator(size).reverse()
    # Label the nodes randomly from available nodes
    graph = relabel_nx_graph(graph, node_labels)
    return graph


