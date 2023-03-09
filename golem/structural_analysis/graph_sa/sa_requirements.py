from collections import namedtuple
from typing import List, Optional

from golem.core.optimisers.graph import OptNode
from golem.structural_analysis.graph_sa.entities.edge import Edge

HyperparamsAnalysisMetaParams = namedtuple('HyperparamsAnalysisMetaParams', ['analyze_method',
                                                                             'sample_method',
                                                                             'sample_size'])

ReplacementAnalysisMetaParams = namedtuple('ReplacementAnalysisMetaParams', ['edges_to_replace_to',
                                                                             'number_of_random_operations_edges',
                                                                             'nodes_to_replace_to',
                                                                             'number_of_random_operations_nodes'])


class StructuralAnalysisRequirements:
    """
    Use this object to pass all the requirements needed for SA.

    :param replacement_nodes_to_replace_to: defines nodes which is used in replacement analysis
    :param replacement_number_of_random_operations_nodes: if replacement_nodes_to_replace_to is not filled, \
    define the number of randomly chosen operations used in replacement analysis
    :param replacement_edges_to_replace_to: defines edges which is used in replacement analysis
    :param replacement_number_of_random_operations_edges: if replacement_edges_to_replace_to is not filled, \
    define the number of randomly chosen operations used in replacement analysis
    :param is_visualize: defines whether the SA visualization needs to be saved to .png files
    :param is_save_results_to_json: defines whether the SA indices needs to be saved to .json file
    """

    def __init__(self,
                 replacement_nodes_to_replace_to: Optional[List[OptNode]] = None,
                 replacement_number_of_random_operations_nodes: Optional[int] = None,
                 replacement_edges_to_replace_to: Optional[List[Edge]] = None,
                 replacement_number_of_random_operations_edges: Optional[int] = None,
                 is_visualize: bool = True,
                 is_save_results_to_json: bool = True):

        self.replacement_meta = ReplacementAnalysisMetaParams(replacement_edges_to_replace_to,
                                                              replacement_number_of_random_operations_edges,
                                                              replacement_nodes_to_replace_to,
                                                              replacement_number_of_random_operations_nodes)

        self.is_visualize = is_visualize
        self.is_save = is_save_results_to_json
