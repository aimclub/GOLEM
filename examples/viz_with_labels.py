from golem.core.optimisers.graph import OptNode, OptGraph


def get_opt_graph():
    node = OptNode('1')
    node2 = OptNode('2')
    node4 = OptNode('4', nodes_from=[node, node2])
    node3 = OptNode('3')
    node5 = OptNode('5', nodes_from=[node4, node3])
    graph = OptGraph(node5)
    return graph


if __name__ == "__main__":
    # Shows how to set labels for nodes and edges.
    graph = get_opt_graph()

    nodes_labels = {}
    for i, node in enumerate(graph.nodes):
        nodes_labels[i] = f'{node.name}_node'

    edges_labels = {}
    for i, edge in enumerate(graph.get_edges()):
        edges_labels[i] = f'{edge[0].name}_{edge[1].name}'

    graph.show(nodes_labels=nodes_labels, edges_labels=edges_labels)
