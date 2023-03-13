import numpy as np

from golem.core.dag.graph_node import GraphNode
from golem.core.log import default_log
from golem.core.optimisers.graph import OptGraph
from golem.structural_analysis.graph_sa.edge_sa_approaches import EdgeReplaceOperationAnalyze
from golem.structural_analysis.graph_sa.node_sa_approaches import NodeReplaceOperationAnalyze
from golem.structural_analysis.graph_sa.result_presenting_structures.sa_analysis_results import SAAnalysisResults


def nodes_deletion(graph: OptGraph, worst_result: dict) -> OptGraph:
    """ Extracts the node index from the entity key and removes it from the graph """

    node_to_delete = worst_result["entity"]
    graph.delete_node(node_to_delete)
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
    for node in graph.nodes:
        if node.description() == node_to_replace.description():
            node_to_replace = node

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

    # next_parent_node = [node for node in graph.nodes
    #                     if graph.nodes.index(node) == next_parent_node_index][0]
    # next_child_node = [node for node in graph.nodes
    #                    if graph.nodes.index(node) == next_child_node_index][0]

    graph.connect_nodes(next_parent_node, next_child_node)
    default_log('EdgeReplacement').message(f'Edge from {parent_node.name} to {child_node.name} was replaced with '
                                           f'edge from {next_parent_node.name} to {next_child_node.name}')

    return graph


def _get_edge_nodes(graph: OptGraph, parent_index: str, child_index: str) -> (GraphNode, GraphNode):
    """ Function to get the nodes of a given edge """
    parent_node = [node for node in graph.nodes
                   if f'parent_node id = {graph.nodes.index(node)}' == parent_index][0]
    child_node = [node for node in graph.nodes
                  if f' child_node id = {graph.nodes.index(node)}' == child_index][0]
    return parent_node, child_node


def extract_result_values(approaches: list, results):
    """ Calculates the average of the results obtained by the approach.
    We subtract one, since loss=metric_after/metric_before and for visualization it is more convenient
    to have the horizontal axis at the level of zero than one """

    gathered_results = []
    for approach in approaches:
        approach_result = [np.mean(result[f'{approach.__name__}']['loss']) - 1 for result in results.values()]
        gathered_results.append(approach_result)

    return gathered_results
