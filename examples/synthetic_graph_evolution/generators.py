from typing import Dict, Callable, Collection, Sequence, Optional

import networkx as nx
import numpy as np

from examples.synthetic_graph_evolution.utils import draw_graphs_subplots

NumNodes = int
DiGraphGenerator = Callable[[NumNodes], nx.DiGraph]


graph_generators: Dict[str, DiGraphGenerator] = {
    'line': lambda n: nx.path_graph(n, create_using=nx.DiGraph),
    'tree': lambda n: nx.random_tree(n, create_using=nx.DiGraph),
    'gnp': lambda n: nx.gnp_random_graph(n, p=0.1),
    'dag': lambda n: generate_dag(n),
    'star': nx.star_graph,
    '2ring': nx.circular_ladder_graph,
    'grid2d': lambda n: nx.grid_2d_graph(int(np.sqrt(n)), int(np.sqrt(n))),
    'hypercube': lambda n: nx.hypercube_graph(int(np.log2(n).round())),
}

graph_kinds: Sequence[str] = tuple(graph_generators.keys())


def generate_dag(n):
    """ Works good for small graphs (up to n=100000) """
    g = nx.gnp_random_graph(n, p=0.5, directed=True)
    g = nx.DiGraph([(u, v) for (u, v) in g.edges() if u < v])
    return g


def nx_to_directed(graph: nx.Graph) -> nx.DiGraph:
    """Randomly chooses a direction for each edge."""
    if isinstance(graph, nx.DiGraph):
        return graph

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


def largest_component(graph: nx.Graph) -> nx.Graph:
    get_components = nx.weakly_connected_components \
        if isinstance(graph, nx.DiGraph) else nx.connected_components
    largest = max(get_components(graph), key=len)
    return graph.subgraph(largest)


def relabel_nx_graph(graph: nx.Graph, available_names: Collection[str]) -> nx.Graph:
    """Randomly label nodes with 'name' attribute in nx.Graph
    given list of available labels"""
    names = np.random.choice(available_names, size=graph.number_of_nodes())
    attributes = {node_id: {'name': name}
                  for node_id, name in zip(graph.nodes, names)}
    nx.set_node_attributes(graph, attributes)
    return graph


def postprocess_nx_graph(graph: nx.Graph,
                         node_labels: Optional[Sequence[str]] = ('x',),
                         connected: bool = True,
                         directed: bool = True):
    """Generate randomly labeled graph, enforce connectedness and direction."""
    # Remove unconnected components
    if connected:
        graph = largest_component(graph)
    # Optionally choose random directions for each edge
    if directed:
        # reverse() is mainly for trees
        # to make them growing towards root
        graph = nx_to_directed(graph).reverse()
    # Label the nodes randomly from available nodes
    if node_labels:
        graph = relabel_nx_graph(graph, node_labels)
    return graph


def generate_labeled_graph(kind: str,
                           size: int,
                           node_labels: Optional[Sequence[str]] = ('x',),
                           connected: bool = True,
                           directed: bool = True):
    """Generate randomly labeled graph of the specified kind and size,
    optionally enforce connectedness and direction. Important! With small specified size
    some methods can generate smaller graphs due to removal of unconnected components."""
    nx_graph = graph_generators[kind](size)
    graph = postprocess_nx_graph(nx_graph, node_labels, connected, directed)
    return graph


def _draw_sample_graphs(kind: str = 'gnp', sizes=tuple(range(5, 50, 5))):
    graphs = [generate_labeled_graph(kind, n) for n in sizes]
    draw_graphs_subplots(*graphs)


if __name__ == '__main__':
    _draw_sample_graphs('gnp')
