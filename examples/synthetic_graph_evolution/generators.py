from typing import Dict, Callable, Collection, Sequence

import networkx as nx
import numpy as np

NumNodes = int
DiGraphGenerator = Callable[[NumNodes], nx.DiGraph]


def nx_to_directed(graph: nx.Graph) -> nx.DiGraph:
    """Randomly chooses a direction for each edge."""
    dedges = set()
    digraph = nx.DiGraph()

    for node, data in graph.nodes(data=True):
        digraph.add_node(node, **data)

    for u, v, data in graph.edges.data():
        edge = (u, v)
        inv_edge = (v, u)
        if edge in dedges or inv_edge in dedges:
            continue

        if np.random.default_rng().random() > 0.5:
            digraph.add_edge(*edge, **data)
            dedges.add(edge)
        else:
            digraph.add_edge(*inv_edge, **data)
            dedges.add(inv_edge)
    return digraph


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


