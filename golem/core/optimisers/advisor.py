from typing import List, Any, TypeVar, Generic

from golem.utilities.data_structures import ComparableEnum as Enum

NodeType = TypeVar('NodeType')


class RemoveType(Enum):
    """Defines allowed kinds of removals in Graph. Used by mutations."""
    forbidden = 'forbidden'
    node_only = 'node_only'
    node_rewire = 'node_rewire'
    with_direct_children = 'with_direct_children'
    with_parents = 'with_parents'


class DefaultChangeAdvisor(Generic[NodeType]):
    """
    Class for advising of graph changes during evolution
    """

    def __init__(self, task=None):
        self.task = task

    def propose_change(self, node: NodeType, possible_operations: List[Any]) -> List[Any]:
        return possible_operations

    def can_be_removed(self, node: NodeType) -> RemoveType:
        return RemoveType.node_rewire

    def can_be_sink(self, node: NodeType) -> bool:
        """Whether the node is acceptable as the final (sink) node of a graph.

        Crossover consults this before placing a subtree head into the sink
        position, so domains whose sinks are constrained - e.g. a pipeline
        must end with a model - do not breed offspring that verification is
        certain to reject.
        """
        return True

    def propose_parent(self, node: NodeType, possible_operations: List[Any]) -> List[Any]:
        return possible_operations
