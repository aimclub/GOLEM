import numpy as np

from golem.core.dag.graph_node import GraphNode
from golem.core.log import default_log
from golem.core.optimisers.graph import OptGraph
from golem.structural_analysis.pipeline_sa.edge_sa_approaches import EdgeReplaceOperationAnalyze
from golem.structural_analysis.pipeline_sa.node_sa_approaches import NodeReplaceOperationAnalyze


def nodes_deletion(pipeline: OptGraph, entity: str, **kwargs) -> OptGraph:
    """ Extracts the node index from the entity key and removes it from the pipeline """

    node_to_delete = [node for node in pipeline.nodes
                      if f'id = {pipeline.nodes.index(node)}, operation = {str(node)}' == entity][0]
    pipeline.delete_node(node_to_delete)
    default_log('NodeDeletion').message(f'{node_to_delete.name} was deleted')

    return pipeline


def nodes_replacement(results: dict, pipeline: OptGraph, entity: str) -> OptGraph:
    """ Extracts the node index and the operation to which it needs to be replaced from the entity key
    and replaces the node with a new one """

    # get the node that will be replaced
    node_to_replace = \
        [node for node in pipeline.nodes
         if f'id = {pipeline.nodes.index(node)}, operation = {str(node)}' == entity][0]

    results = results['nodes_result']
    # get the node to replace
    new_node_operation = \
        results[entity][f'{NodeReplaceOperationAnalyze.__name__}']['new_node_operation']

    old_node_type = type(node_to_replace)
    new_node = old_node_type(operation_type=new_node_operation, nodes_from=node_to_replace.nodes_from)
    pipeline.update_node(old_node=node_to_replace, new_node=new_node)
    default_log('NodeReplacement').message(f'{node_to_replace.name} was replaced with {new_node.name}')

    return pipeline


def subtree_deletion(pipeline: OptGraph, entity: str, **kwargs) -> OptGraph:
    """ Extracts the node index from the entity key and removes its subtree from the pipeline """

    node_to_delete = [node for node in pipeline.nodes
                      if f'id = {pipeline.nodes.index(node)}, operation = {str(node)}' == entity][0]
    pipeline.delete_subtree(node_to_delete)
    default_log('SubtreeDeletion').message(f'{node_to_delete.name} subtree was deleted')

    return pipeline


def edges_deletion(pipeline: OptGraph, entity: str, **kwargs) -> OptGraph:
    """ Extracts the edge's nodes indices from the entity key and removes edge from the pipeline """

    # get an edge to delete
    parent_node, child_node = _get_edge_nodes(pipeline=pipeline,
                                              parent_index=entity.split(',')[0],
                                              child_index=entity.split(',')[1])
    pipeline.disconnect_nodes(parent_node, child_node)
    default_log('EdgeDeletion').message(f'Edge from {parent_node.name} to {child_node.name} was deleted')

    return pipeline


def edges_replacement(results: dict, pipeline: OptGraph, entity: str) -> OptGraph:
    """ Extracts the edge's nodes indices and the new edge to which it needs to be replaced from the entity key
    and replaces the edge with a new one """

    # get the edge that will be replaced
    parent_node, child_node = _get_edge_nodes(pipeline=pipeline,
                                              parent_index=entity.split(',')[0],
                                              child_index=entity.split(',')[1])
    pipeline.disconnect_nodes(parent_node, child_node)

    results = results['edges_result']
    # get an edge to replace
    edge_nodes_idxs = results[entity][f'{EdgeReplaceOperationAnalyze.__name__}']['edge_node_idx_to_replace_to']
    next_parent_node_index = edge_nodes_idxs['parent_node_id']
    next_child_node_index = edge_nodes_idxs['child_node_id']

    next_parent_node = [node for node in pipeline.nodes
                        if pipeline.nodes.index(node) == next_parent_node_index][0]
    next_child_node = [node for node in pipeline.nodes
                       if pipeline.nodes.index(node) == next_child_node_index][0]

    pipeline.connect_nodes(next_parent_node, next_child_node)
    default_log('EdgeReplacement').message(f'Edge from {parent_node.name} to {child_node.name} was replaced with '
                                           f'edge from {next_parent_node.name} to {next_child_node.name}')

    return pipeline


def _get_edge_nodes(pipeline: OptGraph, parent_index: str, child_index: str) -> (GraphNode, GraphNode):
    """ Function to get the nodes of a given edge """
    parent_node = [node for node in pipeline.nodes
                   if f'parent_node id = {pipeline.nodes.index(node)}' == parent_index][0]
    child_node = [node for node in pipeline.nodes
                  if f' child_node id = {pipeline.nodes.index(node)}' == child_index][0]
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
