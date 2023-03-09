from typing import Tuple, List

from golem.core.optimisers.graph import OptNode


class Edge:
    def __init__(self, parent_node: OptNode, child_node: OptNode):
        self.parent_node = parent_node
        self.child_node = child_node
        self.rating = None

    @staticmethod
    def from_tuple(edges_in_tuple: List[Tuple[OptNode, OptNode]]) -> List['Edge']:
        edges = []
        for edge in edges_in_tuple:
            edges.append(Edge(parent_node=edge[0], child_node=edge[1]))
        return edges
