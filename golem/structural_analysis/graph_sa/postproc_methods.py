from golem.core.log import default_log
from golem.core.optimisers.graph import OptGraph, OptNode


def nodes_deletion(graph: OptGraph, worst_result: dict) -> OptGraph:
    """ Extracts the node index from the entity key and removes it from the graph """

    node_to_delete = worst_result["entity"]

    graph.delete_node(get_same_node_from_graph(graph=graph, node=node_to_delete))
    default_log('NodeDeletion').message(f'{node_to_delete.name} was deleted')

    return graph


def nodes_replacement(graph: OptGraph, worst_result: dict) -> OptGraph:
    """ Extracts the node index and the operation to which it needs to be replaced from the entity key
    and replaces the node with a new one """

    # get the node that will be replaced
    node_to_replace = worst_result["entity"]
    # get node to replace to
    new_node = worst_result["entity_to_replace_to"]

    # actualize node to current instance of graph
    node_to_replace = get_same_node_from_graph(graph=graph, node=node_to_replace)

    graph.update_node(old_node=node_to_replace, new_node=new_node)

    default_log('NodeReplacement').message(f'{node_to_replace.name} was replaced with {new_node.name}')

    return graph


def subtree_deletion(graph: OptGraph, worst_result: dict) -> OptGraph:
    """ Extracts the node index from the entity key and removes its subtree from the graph """

    node_to_delete = worst_result["entity"]
    graph.delete_subtree(node_to_delete)
    default_log('SubtreeDeletion').message(f'{node_to_delete.name} subtree was deleted')

    return graph


def edges_deletion(graph: OptGraph, worst_result: dict) -> OptGraph:
    """ Extracts the edge's nodes indices from the entity key and removes edge from the graph """

    parent_node = worst_result['entity'].parent_node
    child_node = worst_result['entity'].child_node
    graph.disconnect_nodes(parent_node, child_node)
    default_log('EdgeDeletion').message(f'Edge from {parent_node.name} to {child_node.name} was deleted')

    return graph


def edges_replacement(graph: OptGraph, worst_result: dict) -> OptGraph:
    """ Extracts the edge's nodes indices and the new edge to which it needs to be replaced from the entity key
    and replaces the edge with a new one """

    # get the edge that will be replaced
    parent_node = worst_result['entity'].parent_node
    child_node = worst_result['entity'].child_node

    graph.disconnect_nodes(parent_node, child_node)

    # get an edge to replace
    next_parent_node = worst_result['entity_to_replace_to'].parent_node
    next_child_node = worst_result['entity_to_replace_to'].child_node

    graph.connect_nodes(next_parent_node, next_child_node)
    default_log('EdgeReplacement').message(f'Edge from {parent_node.name} to {child_node.name} was replaced with '
                                           f'edge from {next_parent_node.name} to {next_child_node.name}')

    return graph


def get_same_node_from_graph(graph: OptGraph, node: OptNode) -> OptNode:
    """ Returns the same node but from particular graph. """
    for cur_node in graph.nodes:
        if cur_node.description() == node.description():
            return cur_node
