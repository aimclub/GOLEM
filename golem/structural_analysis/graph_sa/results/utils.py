from typing import Union

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.structural_analysis.graph_sa.entities.edge import Edge


def get_entity_str(entity: Union[GraphNode, Edge]) -> str:
    if isinstance(entity, GraphNode):
        entity_str = str(hash(entity.descriptive_id))
    else:
        entity_str = str(hash(entity.parent_node.descriptive_id)) + '_' + \
                     str(hash(entity.child_node.descriptive_id))
    return entity_str


def get_entity_by_str(graph: Graph, entity_str: str) -> Union[GraphNode, Edge]:
    entities = graph.nodes + Edge.from_tuple(graph.get_edges())
    for entity in entities:
        if entity_str == get_entity_str(entity=entity):
            return entity
