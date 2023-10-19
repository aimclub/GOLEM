import math
from copy import deepcopy
from functools import partial
from random import choice, randint, random, sample
from typing import TYPE_CHECKING
from datetime import datetime
from golem.core.adapter import register_native
from golem.core.dag.graph import ReconnectType
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_utils import distance_to_root_level, ordered_subnodes_hierarchy, distance_to_primary_level
from golem.core.optimisers.advisor import RemoveType
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters
from golem.core.utilities.data_structures import ComparableEnum as Enum

from functools import partial

if TYPE_CHECKING:
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters


class MutationStrengthEnum(Enum):
    weak = 0.2
    mean = 1.0
    f = 5.0
    strong = f


class MutationTypesEnum(Enum):
    simple = 'simple'
    growth = 'growth'
    local_growth = 'local_growth'
    tree_growth = 'tree_growth'
    reduce = 'reduce'
    single_add = 'single_add'
    single_change = 'single_change'
    single_drop = 'single_drop'
    single_edge = 'single_edge'
    single_edge_add = 'single_edge_add'
    single_edge_del = 'single_edge_del'

    change_label = 'change_label'
    change_label_to_1 = 'change_label_to_1'
    change_label_to_0 = 'change_label_to_0'
    change_label_to_diff = 'change_label_to_diff'

    star_edge = 'star_edge'
    path_edge='path_edge'
    cycle_edge='cycle_edge'


    batch_edge_5='batch_edge_5'
    batch_edge_10='batch_edge_10'
    batch_edge_15='batch_edge_15'
    batch_edge_20='batch_edge_20'
    batch_edge_25='batch_edge_25'
    batch_edge_30='batch_edge_30'
    batch_edge_35='batch_edge_35'
    batch_edge_40='batch_edge_40'
    batch_edge_45='batch_edge_45'
    batch_edge_50='batch_edge_50'
    batch_edge_55='batch_edge_55'


    star_edge_5='star_edge_5'
    star_edge_10='star_edge_10'
    star_edge_15='star_edge_15'
    star_edge_20='star_edge_20'
    star_edge_25='star_edge_25'
    star_edge_30='star_edge_30'
    star_edge_35='star_edge_35'
    star_edge_40='star_edge_40'
    star_edge_45='star_edge_45'
    star_edge_50='star_edge_50'
    star_edge_55='star_edge_55'

    change_label_5='change_label_5'
    change_label_10='change_label_10'
    change_label_15='change_label_15'
    change_label_20='change_label_20'
    change_label_25='change_label_25'
    change_label_30='change_label_30'
    change_label_35='change_label_35'
    change_label_40='change_label_40'
    change_label_45='change_label_45'
    change_label_50='change_label_50'
    change_label_55='change_label_55'

    change_label_ones_5='change_label_ones_5'
    change_label_ones_10='change_label_ones_10'
    change_label_ones_15='change_label_ones_15'
    change_label_ones_20='change_label_ones_20'
    change_label_ones_25='change_label_ones_25'
    change_label_ones_30='change_label_ones_30'
    change_label_ones_35='change_label_ones_35'
    change_label_ones_40='change_label_ones_40'
    change_label_ones_45='change_label_ones_45'
    change_label_ones_50='change_label_ones_50'
    change_label_ones_55='change_label_ones_55'

    change_label_zeros_5='change_label_zeros_5'
    change_label_zeros_10='change_label_zeros_10'
    change_label_zeros_15='change_label_zeros_15'
    change_label_zeros_20='change_label_zeros_20'
    change_label_zeros_25='change_label_zeros_25'
    change_label_zeros_30='change_label_zeros_30'
    change_label_zeros_35='change_label_zeros_35'
    change_label_zeros_40='change_label_zeros_40'
    change_label_zeros_45='change_label_zeros_45'
    change_label_zeros_50='change_label_zeros_50'
    change_label_zeros_55='change_label_zeros_55'

    change_label_diff_5='change_label_diff_5'
    change_label_diff_10='change_label_diff_10'
    change_label_diff_15='change_label_diff_15'
    change_label_diff_20='change_label_diff_20'
    change_label_diff_25='change_label_diff_25'
    change_label_diff_30='change_label_diff_30'
    change_label_diff_35='change_label_diff_35'
    change_label_diff_40='change_label_diff_40'
    change_label_diff_45='change_label_diff_45'
    change_label_diff_50='change_label_diff_50'
    change_label_diff_55='change_label_diff_55'

    cycle_edge_5='cycle_edge_5'
    cycle_edge_10='cycle_edge_10'
    cycle_edge_15='cycle_edge_15'
    cycle_edge_20='cycle_edge_20'
    cycle_edge_25='cycle_edge_25'
    cycle_edge_30='cycle_edge_30'
    cycle_edge_35='cycle_edge_35'
    cycle_edge_40='cycle_edge_40'
    cycle_edge_45='cycle_edge_45'
    cycle_edge_50='cycle_edge_50'
    cycle_edge_55='cycle_edge_55'



    path_edge_5='path_edge_5'
    path_edge_10='path_edge_10'
    path_edge_15='path_edge_15'
    path_edge_20='path_edge_20'
    path_edge_25='path_edge_25'
    path_edge_30='path_edge_30'
    path_edge_35='path_edge_35'
    path_edge_40='path_edge_40'
    path_edge_45='path_edge_45'
    path_edge_50='path_edge_50'
    path_edge_55='path_edge_55'

    dense_edge_5='dense_edge_5'
    dense_edge_10='dense_edge_10'
    dense_edge_15='dense_edge_15'
    dense_edge_20='dense_edge_20'
    dense_edge_25='dense_edge_25'
    dense_edge_30='dense_edge_30'
    dense_edge_35='dense_edge_35'
    dense_edge_40='dense_edge_40'
    dense_edge_45='dense_edge_45'
    dense_edge_50='dense_edge_50'
    dense_edge_55='dense_edge_55'


    dense_edge='dense_edge'
    batch_edge_del = 'batch_edge_delete'

    none = 'none'


def get_mutation_prob(mut_id: MutationStrengthEnum, node: GraphNode,
                      default_mutation_prob: float = 0.7) -> float:
    """ Function returns mutation probability for certain node in the graph

    :param mut_id: MutationStrengthEnum mean weak or strong mutation
    :param node: root node of the graph
    :param default_mutation_prob: mutation probability used when mutation_id is invalid
    :return mutation_prob: mutation probability
    """
    if mut_id in list(MutationStrengthEnum):
        mutation_strength = mut_id.value
        mutation_prob = mutation_strength / (distance_to_primary_level(node) + 1)
    else:
        mutation_prob = default_mutation_prob
    return mutation_prob


@register_native
def simple_mutation(graph: OptGraph,
                    requirements: GraphRequirements,
                    graph_gen_params: GraphGenerationParams,
                    parameters: 'GPAlgorithmParameters',
                    ) -> OptGraph:
    """
    This type of mutation is passed over all nodes of the tree started from the root node and changes
    nodes’ operations with probability - 'node mutation probability'
    which is initialised inside the function

    :param graph: graph to mutate
    """

    exchange_node = graph_gen_params.node_factory.exchange_node
    visited_nodes = set()

    def replace_node_to_random_recursive(node: OptNode) -> OptGraph:
        if node not in visited_nodes and random() < node_mutation_probability:
            new_node = exchange_node(node)
            if new_node:
                graph.update_node(node, new_node)
            # removed node must not be visited because it's outdated
            visited_nodes.add(node)
            # new node must not mutated if encountered further during traverse
            visited_nodes.add(new_node)
            for parent in node.nodes_from:
                replace_node_to_random_recursive(parent)

    node_mutation_probability = get_mutation_prob(mut_id=parameters.mutation_strength,
                                                  node=graph.root_node)

    replace_node_to_random_recursive(graph.root_node)

    return graph


@register_native
def single_edge_mutation(graph: OptGraph,
                         requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: 'GPAlgorithmParameters',
                         ) -> OptGraph:
    """
    This mutation adds new edge between two random nodes in graph.

    :param graph: graph to mutate
    """
    old_graph = deepcopy(graph)
    for _ in range(parameters.max_num_of_operator_attempts):
        if len(graph.nodes) < 2:# or graph.depth > requirements.max_depth:
                return graph


        source_node, target_node = sample(graph.nodes, 2)
        if (source_node not in target_node.nodes_from) and (target_node not in source_node.nodes_from):
            graph.connect_nodes(source_node, target_node)
            break
    for _ in range(parameters.max_num_of_operator_attempts):
        try:
            if len(graph.get_edges()) ==0:
                return graph
            source_node, target_node = sample(graph.get_edges(), 1)[0]

            graph.disconnect_nodes(source_node, target_node)
            break
        except:
            continue

#    if graph.depth > requirements.max_depth:
 #       return old_graph
    return graph


@register_native
def single_edge_add_mutation(graph: OptGraph,
                         requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: 'GPAlgorithmParameters',
                         ) -> OptGraph:
    """
    This mutation adds new edge between two random nodes in graph.

    :param graph: graph to mutate
    """
    old_graph = deepcopy(graph)
    for _ in range(parameters.max_num_of_operator_attempts):
        if len(graph.nodes) < 2:# or graph.depth > requirements.max_depth:
                return graph


        source_node, target_node = sample(graph.nodes, 2)
        if (source_node not in target_node.nodes_from) and (target_node not in source_node.nodes_from):
            graph.connect_nodes(source_node, target_node)
            break


#    if graph.depth > requirements.max_depth:
 #       return old_graph
    return graph

@register_native
def single_edge_del_mutation(graph: OptGraph,
                         requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: 'GPAlgorithmParameters',
                         ) -> OptGraph:
    """
    This mutation adds new edge between two random nodes in graph.

    :param graph: graph to mutate
    """
    old_graph = deepcopy(graph)

    for _ in range(parameters.max_num_of_operator_attempts):
        try:
            if len(graph.get_edges()) ==0:
                return graph
            source_node, target_node = sample(graph.get_edges(), 1)[0]

            graph.disconnect_nodes(source_node, target_node)
            break
        except:
            continue

#    if graph.depth > requirements.max_depth:
 #       return old_graph
    return graph

@register_native
def batch_edge_mutation(graph: OptGraph,
                         requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,

                         parameters: 'GPAlgorithmParameters',num_edges

                         ) -> OptGraph:
    """
    This mutation adds new edge between two random nodes in graph.

    :param graph: graph to mutate
    """


   # print('num nodes',num_edges)
    old_graph = deepcopy(graph)
    for _ in range(num_edges):
        for _ in range(parameters.max_num_of_operator_attempts):
            if len(graph.nodes) < 2:# or graph.depth > requirements.max_depth:
                    return graph


            source_node, target_node = sample(graph.nodes, 2)
            if (source_node not in target_node.nodes_from) and (target_node not in source_node.nodes_from):
                graph.connect_nodes(source_node, target_node)
                break
        for _ in range(parameters.max_num_of_operator_attempts):
            try:
                if len(graph.get_edges()) ==0:
                    return graph
                source_node, target_node = sample(graph.get_edges(), 1)[0]

                graph.disconnect_nodes(source_node, target_node)
                break
            except:
                continue

      #  if graph.depth > requirements.max_depth:
       #     return old_graph

    return graph

batch_edge_5_mutation = partial(batch_edge_mutation, num_edges=5)
batch_edge_10_mutation = partial(batch_edge_mutation, num_edges=10)
batch_edge_15_mutation = partial(batch_edge_mutation, num_edges=15)
batch_edge_20_mutation = partial(batch_edge_mutation, num_edges=20)
batch_edge_25_mutation = partial(batch_edge_mutation, num_edges=25)
batch_edge_30_mutation = partial(batch_edge_mutation, num_edges=30)
batch_edge_35_mutation = partial(batch_edge_mutation, num_edges=35)
batch_edge_40_mutation = partial(batch_edge_mutation, num_edges=40)
batch_edge_45_mutation = partial(batch_edge_mutation, num_edges=45)
batch_edge_50_mutation = partial(batch_edge_mutation, num_edges=50)
batch_edge_55_mutation = partial(batch_edge_mutation, num_edges=55)


@register_native
def star_edge_mutation(graph: OptGraph,
                         requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: 'GPAlgorithmParameters',num_edges
                         ) -> OptGraph:
    """
    This mutation adds new edge between two random nodes in graph.

    :param graph: graph to mutate
    """
    #num_edges = requirements.num_edges
    num_nodes = num_edges + 1
    old_graph = deepcopy(graph)

    for _ in range(parameters.max_num_of_operator_attempts):
        try:
            if len(graph.nodes) < 2: # or graph.depth > requirements.max_depth:
                    return graph

            nodes = sample(graph.nodes, num_nodes)

            source_node = nodes[0]
            for target_node in nodes[1:]:
                if (source_node not in target_node.nodes_from) and (target_node not in source_node.nodes_from):
                    graph.connect_nodes(source_node, target_node)
            break
        except:
            continue

    for _ in range(int(num_edges)):
        for _ in range(parameters.max_num_of_operator_attempts):
            try:
                if len(graph.get_edges()) == 0:
                    return graph
                source_node, target_node = sample(graph.get_edges(), 1)[0]

                graph.disconnect_nodes(source_node, target_node)
                break
            except:
                continue

    #if graph.depth > requirements.max_depth:
     #   return old_graph
    return graph
star_edge_5_mutation = partial(star_edge_mutation, num_edges=5)
star_edge_10_mutation = partial(star_edge_mutation, num_edges=10)
star_edge_15_mutation = partial(star_edge_mutation, num_edges=15)
star_edge_20_mutation = partial(star_edge_mutation, num_edges=20)
star_edge_25_mutation = partial(star_edge_mutation, num_edges=25)
star_edge_30_mutation = partial(star_edge_mutation, num_edges=30)
star_edge_35_mutation = partial(star_edge_mutation, num_edges=35)
star_edge_40_mutation = partial(star_edge_mutation, num_edges=40)
star_edge_45_mutation = partial(star_edge_mutation, num_edges=45)
star_edge_50_mutation = partial(star_edge_mutation, num_edges=50)
star_edge_55_mutation = partial(star_edge_mutation, num_edges=55)


@register_native
def cycle_edge_mutation(graph: OptGraph,
                       requirements: GraphRequirements,
                       graph_gen_params: GraphGenerationParams,
                       parameters: 'GPAlgorithmParameters', num_edges
                       ) -> OptGraph:
    """
    This mutation adds new edge between two random nodes in graph.

    :param graph: graph to mutate
    """
    num_edges = num_edges#requirements.num_edges
    num_nodes = num_edges
    old_graph = deepcopy(graph)

    for _ in range(parameters.max_num_of_operator_attempts):
        try:
            if len(graph.nodes) < 2: # or graph.depth > requirements.max_depth:
                return graph

            nodes = sample(graph.nodes, num_nodes)

            first_node=nodes[0]
            source_node = first_node
            for target_node in nodes[1:]:
                if (source_node not in target_node.nodes_from) and (target_node not in source_node .nodes_from):
                    graph.connect_nodes(source_node, target_node)
                source_node = target_node
            if (first_node not in target_node.nodes_from) and (target_node not in source_node .nodes_from):
                graph.connect_nodes(source_node, first_node)
            break
        except:
            continue
    for _ in range(int(num_edges)):
        for _ in range(parameters.max_num_of_operator_attempts):
            try:
                if len(graph.get_edges()) == 0:
                    return graph
                source_node, target_node = sample(graph.get_edges(), 1)[0]

                graph.disconnect_nodes(source_node, target_node)
                break
            except:
                continue

#    if graph.depth > requirements.max_depth:
 #       return old_graph
    return graph
cycle_edge_5_mutation = partial(cycle_edge_mutation, num_edges=5)
cycle_edge_10_mutation = partial(cycle_edge_mutation, num_edges=10)
cycle_edge_15_mutation = partial(cycle_edge_mutation, num_edges=15)
cycle_edge_20_mutation = partial(cycle_edge_mutation, num_edges=20)
cycle_edge_25_mutation = partial(cycle_edge_mutation, num_edges=25)
cycle_edge_30_mutation = partial(cycle_edge_mutation, num_edges=30)
cycle_edge_35_mutation = partial(cycle_edge_mutation, num_edges=35)
cycle_edge_40_mutation = partial(cycle_edge_mutation, num_edges=40)
cycle_edge_45_mutation = partial(cycle_edge_mutation, num_edges=45)
cycle_edge_50_mutation = partial(cycle_edge_mutation, num_edges=50)
cycle_edge_55_mutation = partial(cycle_edge_mutation, num_edges=55)
@register_native
def path_edge_mutation(graph: OptGraph,
                       requirements: GraphRequirements,
                       graph_gen_params: GraphGenerationParams,
                       parameters: 'GPAlgorithmParameters',num_edges
                       ) -> OptGraph:
    """
    This mutation adds new edge between two random nodes in graph.

    :param graph: graph to mutate
    """
    num_edges = num_edges#requirements.num_edges
    num_nodes = num_edges + 1

    old_graph = deepcopy(graph)

    for _ in range(parameters.max_num_of_operator_attempts):
        try:
            if len(graph.nodes) < 2: # or graph.depth > requirements.max_depth:
                return graph

            nodes = sample(graph.nodes, num_nodes)

            source_node = nodes[0]
            for target_node in nodes[1:]:
                if (source_node not in target_node.nodes_from) and (target_node not in source_node.nodes_from):
                    graph.connect_nodes(source_node, target_node)
                source_node = target_node
            break
        except:
            continue
    for _ in range(int(num_edges)):
        for _ in range(parameters.max_num_of_operator_attempts):
            try:
                if len(graph.get_edges()) == 0:
                    return graph
                source_node, target_node = sample(graph.get_edges(), 1)[0]

                graph.disconnect_nodes(source_node, target_node)
                break
            except:
                continue

 #   if graph.depth > requirements.max_depth:
  #      return old_graph
    return graph

path_edge_5_mutation = partial(path_edge_mutation, num_edges=5)
path_edge_10_mutation = partial(path_edge_mutation, num_edges=10)
path_edge_15_mutation = partial(path_edge_mutation, num_edges=15)
path_edge_20_mutation = partial(path_edge_mutation, num_edges=20)
path_edge_25_mutation = partial(path_edge_mutation, num_edges=25)
path_edge_30_mutation = partial(path_edge_mutation, num_edges=30)
path_edge_35_mutation = partial(path_edge_mutation, num_edges=35)
path_edge_40_mutation = partial(path_edge_mutation, num_edges=40)
path_edge_45_mutation = partial(path_edge_mutation, num_edges=45)
path_edge_50_mutation = partial(path_edge_mutation, num_edges=50)
path_edge_55_mutation = partial(path_edge_mutation, num_edges=55)

@register_native
def dense_edge_mutation(graph: OptGraph,
                       requirements: GraphRequirements,
                       graph_gen_params: GraphGenerationParams,
                       parameters: 'GPAlgorithmParameters', num_edges
                       ) -> OptGraph:
    """
    This mutation adds new edge between two random nodes in graph.

    :param graph: graph to mutate
    """
    num_edges = num_edges# requirements.num_edges#int((num_nodes*(num_nodes-1))/4)
    num_nodes = int((1 + math.sqrt(1+8*num_edges))/2)#num_nodes*(num_nodes-1)/2
    old_graph = deepcopy(graph)
    for o in range(parameters.max_num_of_operator_attempts):
        try:
            if len(graph.nodes) < 2:# or graph.depth > requirements.max_depth:
                return graph

            nodes = sample(graph.nodes, num_nodes)
            for k,source_node in enumerate(nodes):
                for target_node in nodes[k:]:
                    if (source_node not in target_node.nodes_from) and (target_node not in source_node.nodes_from):
                        graph.connect_nodes(source_node, target_node)
                    source_node = target_node
            break
        except:
            continue
    for _ in range(int(num_edges)):
        for _ in range(parameters.max_num_of_operator_attempts):
            try:
                if len(graph.get_edges()) == 0:
                    return graph
                source_node, target_node = sample(graph.get_edges(), 1)[0]

                graph.disconnect_nodes(source_node, target_node)
                break
            except:
                continue
   # if graph.depth > requirements.max_depth:
    #    return old_graph
    return graph

dense_edge_5_mutation = partial(dense_edge_mutation, num_edges=5)
dense_edge_10_mutation = partial(dense_edge_mutation, num_edges=10)
dense_edge_15_mutation = partial(dense_edge_mutation, num_edges=15)
dense_edge_20_mutation = partial(dense_edge_mutation, num_edges=20)
dense_edge_25_mutation = partial(dense_edge_mutation, num_edges=25)
dense_edge_30_mutation = partial(dense_edge_mutation, num_edges=30)
dense_edge_35_mutation = partial(dense_edge_mutation, num_edges=35)
dense_edge_40_mutation = partial(dense_edge_mutation, num_edges=40)
dense_edge_45_mutation = partial(dense_edge_mutation, num_edges=45)
dense_edge_50_mutation = partial(dense_edge_mutation, num_edges=50)
dense_edge_55_mutation = partial(dense_edge_mutation, num_edges=55)

@register_native
def add_intermediate_node(graph: OptGraph,
                          node_to_mutate: OptNode,
                          node_factory: OptNodeFactory) -> OptGraph:
    # add between node and parent
    new_node = node_factory.get_parent_node(node_to_mutate, is_primary=False)
    if not new_node:
        return graph

    # rewire old children to new parent
    new_node.nodes_from = node_to_mutate.nodes_from
    node_to_mutate.nodes_from = [new_node]

    # add new node to graph
    graph.add_node(new_node)
    return graph

@register_native
def change_label(graph: OptGraph,requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: 'GPAlgorithmParameters') -> OptGraph:
    node = choice(graph.nodes)
    print('before change label')
    new_label = randint(0,1)
    ind = randint(0,len(graph.nodes)-1)
    graph.nodes[ind].content['label']=new_label
    print('after change label')
    return graph

@register_native
def change_label_to_1(graph: OptGraph,requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: 'GPAlgorithmParameters') -> OptGraph:
    num_nodes = 9
    num_edges = int((num_nodes*(num_nodes-1))/4)

    for _ in range(num_edges):
        for _ in range(parameters.max_num_of_operator_attempts):
            if len(graph.get_edges()) ==0:
                return graph
            source_node, target_node = sample(graph.get_edges(), 1)[0]
            graph.nodes[int(source_node.descriptive_id.split('_')[-1])].content['label'] = 1
            graph.nodes[int(target_node.descriptive_id.split('_')[-1])].content['label'] = 1
            break
    return graph

@register_native
def change_label_to_0(graph: OptGraph,requirements: GraphRequirements,
                      graph_gen_params: GraphGenerationParams,
                      parameters: 'GPAlgorithmParameters') -> OptGraph:
    num_nodes = 9
    num_edges = int((num_nodes * (num_nodes - 1)) / 4)

    for _ in range(num_edges):
        for _ in range(parameters.max_num_of_operator_attempts):
            if len(graph.get_edges()) == 0:
                return graph
            source_node, target_node = sample(graph.get_edges(), 1)[0]
            #print(source_node.descriptive_id, (source_node.descriptive_id.split('_')[-1]))
            graph.nodes[int(source_node.descriptive_id.split('_')[-1])].content['label'] = 0
            graph.nodes[int(target_node.descriptive_id.split('_')[-1])].content['label'] = 0
            break
    return graph

@register_native
def change_label_to_diff(graph: OptGraph,requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: 'GPAlgorithmParameters') -> OptGraph:
    num_nodes = 9
    num_edges = int((num_nodes * (num_nodes - 1)) / 4)

    for _ in range(num_edges):
        for _ in range(parameters.max_num_of_operator_attempts):
            if len(graph.get_edges()) == 0:
                return graph
            source_node, target_node = sample(graph.get_edges(), 1)[0]
            graph.nodes[int(source_node.descriptive_id.split('_')[-1])].content['label'] = 0
            graph.nodes[int(target_node.descriptive_id.split('_')[-1])].content['label'] = 1
            break
    return graph

def node_parents(graph, node: GraphNode):
    nodes = []
    for other_node in graph.nodes:
        if other_node in node.nodes_from:
            nodes.append(other_node)
    return nodes



@register_native
def change_label(graph: OptGraph,requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: 'GPAlgorithmParameters', num_edges) -> OptGraph:
    node = choice(graph.nodes)
    for i in range(num_edges):
        new_label = randint(0,1)
        ind = randint(0,len(graph.nodes)-1)
        graph.nodes[ind].content['label']=new_label
    return graph

change_label_5_mutation = partial(change_label, num_edges=5)
change_label_10_mutation = partial(change_label, num_edges=10)
change_label_15_mutation = partial(change_label, num_edges=15)
change_label_20_mutation = partial(change_label, num_edges=20)
change_label_25_mutation = partial(change_label, num_edges=25)
change_label_30_mutation = partial(change_label, num_edges=30)
change_label_35_mutation = partial(change_label, num_edges=35)
change_label_40_mutation = partial(change_label, num_edges=40)
change_label_45_mutation = partial(change_label, num_edges=45)
change_label_50_mutation = partial(change_label, num_edges=50)
change_label_55_mutation = partial(change_label, num_edges=55)

@register_native
def change_label_ones(graph: OptGraph,requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: 'GPAlgorithmParameters', num_edges) -> OptGraph:
    for _ in range(num_edges):
        for _ in range(parameters.max_num_of_operator_attempts):
            if len(graph.get_edges()) ==0:
                return graph
            source_node, target_node = sample(graph.get_edges(), 1)[0]

            graph.nodes[int(str(source_node))].content['label'] = 1
            graph.nodes[int(str(target_node))].content['label'] = 1
            break
    return graph
change_label_ones_5_mutation = partial(change_label_ones, num_edges=5)
change_label_ones_10_mutation = partial(change_label_ones, num_edges=10)
change_label_ones_15_mutation = partial(change_label_ones, num_edges=15)
change_label_ones_20_mutation = partial(change_label_ones, num_edges=20)
change_label_ones_25_mutation = partial(change_label_ones, num_edges=25)
change_label_ones_30_mutation = partial(change_label_ones, num_edges=30)
change_label_ones_35_mutation = partial(change_label_ones, num_edges=35)
change_label_ones_40_mutation = partial(change_label_ones, num_edges=40)
change_label_ones_45_mutation = partial(change_label_ones, num_edges=45)
change_label_ones_50_mutation = partial(change_label_ones, num_edges=50)
change_label_ones_55_mutation = partial(change_label_ones, num_edges=55)
@register_native
def change_label_zeros(graph: OptGraph,requirements: GraphRequirements,
                      graph_gen_params: GraphGenerationParams,
                      parameters: 'GPAlgorithmParameters', num_edges) -> OptGraph:
    for _ in range(num_edges):
        for _ in range(parameters.max_num_of_operator_attempts):
            if len(graph.get_edges()) == 0:
                return graph
            source_node, target_node = sample(graph.get_edges(), 1)[0]

            #print(source_node.descriptive_id, (source_node.descriptive_id.split('_')[-1]))

            graph.nodes[int(str(source_node))].content['label'] = 0
            graph.nodes[int(str(target_node))].content['label'] = 0
            break
    return graph
change_label_zeros_5_mutation = partial(change_label_zeros, num_edges=5)
change_label_zeros_10_mutation = partial(change_label_zeros, num_edges=10)
change_label_zeros_15_mutation = partial(change_label_zeros, num_edges=15)
change_label_zeros_20_mutation = partial(change_label_zeros, num_edges=20)
change_label_zeros_25_mutation = partial(change_label_zeros, num_edges=25)
change_label_zeros_30_mutation = partial(change_label_zeros, num_edges=30)
change_label_zeros_35_mutation = partial(change_label_zeros, num_edges=35)
change_label_zeros_40_mutation = partial(change_label_zeros, num_edges=40)
change_label_zeros_45_mutation = partial(change_label_zeros, num_edges=45)
change_label_zeros_50_mutation = partial(change_label_zeros, num_edges=50)
change_label_zeros_55_mutation = partial(change_label_zeros, num_edges=55)

@register_native
def change_label_diff(graph: OptGraph,requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: 'GPAlgorithmParameters',num_edges) -> OptGraph:
    for _ in range(num_edges):
        for _ in range(parameters.max_num_of_operator_attempts):
            if len(graph.get_edges()) == 0:
                return graph
            source_node, target_node = sample(graph.get_edges(), 1)[0]

            graph.nodes[int(str(source_node))].content['label'] = 0
            graph.nodes[int(str(target_node))].content['label'] = 1
            break
    return graph
change_label_diff_5_mutation = partial(change_label_diff, num_edges=5)
change_label_diff_10_mutation = partial(change_label_diff, num_edges=10)
change_label_diff_15_mutation = partial(change_label_diff, num_edges=15)
change_label_diff_20_mutation = partial(change_label_diff, num_edges=20)
change_label_diff_25_mutation = partial(change_label_diff, num_edges=25)
change_label_diff_30_mutation = partial(change_label_diff, num_edges=30)
change_label_diff_35_mutation = partial(change_label_diff, num_edges=35)
change_label_diff_40_mutation = partial(change_label_diff, num_edges=40)
change_label_diff_45_mutation = partial(change_label_diff, num_edges=45)
change_label_diff_50_mutation = partial(change_label_diff, num_edges=50)
change_label_diff_55_mutation = partial(change_label_diff, num_edges=55)
def node_parents(graph, node: GraphNode):
    nodes = []
    for other_node in graph.nodes:
        if other_node in node.nodes_from:
            nodes.append(other_node)
    return nodes



@register_native
def add_separate_parent_node(graph: OptGraph,
                             node_to_mutate: OptNode,
                             node_factory: OptNodeFactory) -> OptGraph:
    # add as separate parent
    new_node = node_factory.get_parent_node(node_to_mutate, is_primary=True)
    if not new_node:
        # there is no possible operators
        return graph
    if node_to_mutate.nodes_from:
        node_to_mutate.nodes_from.append(new_node)
    else:
        node_to_mutate.nodes_from = [new_node]
    graph.nodes.append(new_node)
    return graph


@register_native
def add_as_child(graph: OptGraph,
                 node_to_mutate: OptNode,
                 node_factory: OptNodeFactory) -> OptGraph:
    # add as child
    old_node_children = graph.node_children(node_to_mutate)
    new_node_child = choice(old_node_children) if old_node_children else None
    new_node = node_factory.get_node(is_primary=False)
    if not new_node:
        return graph
    graph.add_node(new_node)
    graph.connect_nodes(node_parent=node_to_mutate, node_child=new_node)
    if new_node_child:
        graph.connect_nodes(node_parent=new_node, node_child=new_node_child)
        graph.disconnect_nodes(node_parent=node_to_mutate, node_child=new_node_child,
                               clean_up_leftovers=True)

    return graph


@register_native
def single_add_mutation(graph: OptGraph,
                        requirements: GraphRequirements,
                        graph_gen_params: GraphGenerationParams,
                        parameters: AlgorithmParameters,
                        ) -> OptGraph:
    """
    Add new node between two sequential existing modes

    :param graph: graph to mutate
    """

   # if graph.depth >= requirements.max_depth:
        # add mutation is not possible
    #    return graph

    node_to_mutate = choice(graph.nodes)

    single_add_strategies = [add_as_child, add_separate_parent_node]
    if node_to_mutate.nodes_from:
        single_add_strategies.append(add_intermediate_node)
    strategy = choice(single_add_strategies)

    result = strategy(graph, node_to_mutate, graph_gen_params.node_factory)
    return result


@register_native
def single_change_mutation(graph: OptGraph,
                           requirements: GraphRequirements,
                           graph_gen_params: GraphGenerationParams,
                           parameters: AlgorithmParameters,
                           ) -> OptGraph:
    """
    Change node between two sequential existing modes.

    :param graph: graph to mutate
    """
    node = choice(graph.nodes)
    new_node = graph_gen_params.node_factory.exchange_node(node)
    if not new_node:
        return graph
    graph.update_node(node, new_node)
    return graph


@register_native
def single_drop_mutation(graph: OptGraph,
                         requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: AlgorithmParameters,
                         ) -> OptGraph:
    """
    Drop single node from graph.

    :param graph: graph to mutate
    """
    if len(graph.nodes) < 2:
        return graph
    node_to_del = choice(graph.nodes)
    node_name = node_to_del.name
    removal_type = graph_gen_params.advisor.can_be_removed(node_to_del)
    if removal_type == RemoveType.with_direct_children:
        # TODO refactor workaround with data_source
        graph.delete_node(node_to_del)
        nodes_to_delete = \
            [n for n in graph.nodes
             if n.descriptive_id.count('data_source') == 1
             and node_name in n.descriptive_id]
        for child_node in nodes_to_delete:
            graph.delete_node(child_node, reconnect=ReconnectType.all)
    elif removal_type == RemoveType.with_parents:
        graph.delete_subtree(node_to_del)
    elif removal_type == RemoveType.node_rewire:
        graph.delete_node(node_to_del, reconnect=ReconnectType.all)
    elif removal_type == RemoveType.node_only:
        graph.delete_node(node_to_del, reconnect=ReconnectType.none)
    elif removal_type == RemoveType.forbidden:
        pass
    else:
        raise ValueError("Unknown advice (RemoveType) returned by Advisor ")
    return graph


@register_native
def tree_growth(graph: OptGraph,
                requirements: GraphRequirements,
                graph_gen_params: GraphGenerationParams,
                parameters: AlgorithmParameters,
                local_growth: bool = True) -> OptGraph:
    """
    This mutation selects a random node in a tree, generates new subtree,
    and replaces the selected node's subtree.

    :param graph: graph to mutate
    :param local_growth: if true then maximal depth of new subtree equals depth of tree located in
    selected random node, if false then previous depth of selected node doesn't affect to
    new subtree depth, maximal depth of new subtree just should satisfy depth constraint in parent tree
    """
    node_from_graph = choice(graph.nodes)
    if local_growth:
        max_depth = distance_to_primary_level(node_from_graph)
        is_primary_node_selected = (not node_from_graph.nodes_from) or (node_from_graph != graph.root_node and
                                                                        randint(0, 1))
    else:
        max_depth = requirements.max_depth - distance_to_root_level(graph, node_from_graph)
        is_primary_node_selected = \
            distance_to_root_level(graph, node_from_graph) >= requirements.max_depth and randint(0, 1)
    if is_primary_node_selected:
        new_subtree = graph_gen_params.node_factory.get_node(is_primary=True)
        if not new_subtree:
            return graph
    else:
        new_subtree = graph_gen_params.random_graph_factory(requirements, max_depth).root_node
    graph.update_subtree(node_from_graph, new_subtree)
    return graph


@register_native
def growth_mutation(graph: OptGraph,
                    requirements: GraphRequirements,
                    graph_gen_params: GraphGenerationParams,
                    parameters: AlgorithmParameters,
                    local_growth: bool = True
                    ) -> OptGraph:
    """
    This mutation adds new nodes to the graph (just single node between existing nodes or new subtree).

    :param graph: graph to mutate
    :param local_growth: if true then maximal depth of new subtree equals depth of tree located in
    selected random node, if false then previous depth of selected node doesn't affect to
    new subtree depth, maximal depth of new subtree just should satisfy depth constraint in parent tree
    """

    if random() > 0.5:
        # simple growth (one node can be added)
        return single_add_mutation(graph, requirements, graph_gen_params, parameters)
    else:
        # advanced growth (several nodes can be added)
        return tree_growth(graph, requirements, graph_gen_params, parameters, local_growth)


@register_native
def reduce_mutation(graph: OptGraph,
                    requirements: GraphRequirements,
                    graph_gen_params: GraphGenerationParams,
                    parameters: AlgorithmParameters,
                    ) -> OptGraph:
    """
    Selects a random node in a tree, then removes its subtree. If the current arity of the node's
    parent is more than the specified minimal arity, then the selected node is also removed.
    Otherwise, it is replaced by a random primary node.

    :param graph: graph to mutate
    """
    if len(graph.nodes) == 1:
        return graph

    nodes = [node for node in graph.nodes if node is not graph.root_node]
    node_to_del = choice(nodes)
    children = graph.node_children(node_to_del)
    is_possible_to_delete = all([len(child.nodes_from) - 1 >= requirements.min_arity for child in children])
    if is_possible_to_delete:
        graph.delete_subtree(node_to_del)
    else:
        primary_node = graph_gen_params.node_factory.get_node(is_primary=True)
        if not primary_node:
            return graph
        graph.update_subtree(node_to_del, primary_node)
    return graph


@register_native
def no_mutation(graph: OptGraph, *args, **kwargs) -> OptGraph:
    return graph


base_mutations_repo = {

    MutationTypesEnum.none: no_mutation,
    MutationTypesEnum.simple: simple_mutation,
    MutationTypesEnum.growth: partial(growth_mutation, local_growth=False),
    MutationTypesEnum.local_growth: partial(growth_mutation, local_growth=True),
    MutationTypesEnum.tree_growth: tree_growth,
    MutationTypesEnum.reduce: reduce_mutation,
    MutationTypesEnum.single_add: single_add_mutation,
    MutationTypesEnum.single_edge: single_edge_mutation,
    MutationTypesEnum.single_edge_add: single_edge_add_mutation,
    MutationTypesEnum.single_edge_del: single_edge_del_mutation,


    MutationTypesEnum.star_edge_5: star_edge_5_mutation,
    MutationTypesEnum.star_edge_10: star_edge_10_mutation,
    MutationTypesEnum.star_edge_15: star_edge_15_mutation,
    MutationTypesEnum.star_edge_20: star_edge_20_mutation,
    MutationTypesEnum.star_edge_25: star_edge_25_mutation,
    MutationTypesEnum.star_edge_30: star_edge_30_mutation,
    MutationTypesEnum.star_edge_35: star_edge_35_mutation,
    MutationTypesEnum.star_edge_40: star_edge_40_mutation,
    MutationTypesEnum.star_edge_45: star_edge_45_mutation,
    MutationTypesEnum.star_edge_50: star_edge_50_mutation,
    MutationTypesEnum.star_edge_55: star_edge_55_mutation,

    MutationTypesEnum.change_label_5: change_label_5_mutation,
    MutationTypesEnum.change_label_10: change_label_10_mutation,
    MutationTypesEnum.change_label_15: change_label_15_mutation,
    MutationTypesEnum.change_label_20: change_label_20_mutation,
    MutationTypesEnum.change_label_25: change_label_25_mutation,
    MutationTypesEnum.change_label_30: change_label_30_mutation,
    MutationTypesEnum.change_label_35: change_label_35_mutation,
    MutationTypesEnum.change_label_40: change_label_40_mutation,
    MutationTypesEnum.change_label_45: change_label_45_mutation,
    MutationTypesEnum.change_label_50: change_label_50_mutation,
    MutationTypesEnum.change_label_55: change_label_55_mutation,

    MutationTypesEnum.change_label_ones_5: change_label_ones_5_mutation,
    MutationTypesEnum.change_label_ones_10: change_label_ones_10_mutation,
    MutationTypesEnum.change_label_ones_15: change_label_ones_15_mutation,
    MutationTypesEnum.change_label_ones_20: change_label_ones_20_mutation,
    MutationTypesEnum.change_label_ones_25: change_label_ones_25_mutation,
    MutationTypesEnum.change_label_ones_30: change_label_ones_30_mutation,
    MutationTypesEnum.change_label_ones_35: change_label_ones_35_mutation,
    MutationTypesEnum.change_label_ones_40: change_label_ones_40_mutation,
    MutationTypesEnum.change_label_ones_45: change_label_ones_45_mutation,
    MutationTypesEnum.change_label_ones_50: change_label_ones_50_mutation,
    MutationTypesEnum.change_label_ones_55: change_label_ones_55_mutation,

    MutationTypesEnum.change_label_zeros_5: change_label_zeros_5_mutation,
    MutationTypesEnum.change_label_zeros_10: change_label_zeros_10_mutation,
    MutationTypesEnum.change_label_zeros_15: change_label_zeros_15_mutation,
    MutationTypesEnum.change_label_zeros_20: change_label_zeros_20_mutation,
    MutationTypesEnum.change_label_zeros_25: change_label_zeros_25_mutation,
    MutationTypesEnum.change_label_zeros_30: change_label_zeros_30_mutation,
    MutationTypesEnum.change_label_zeros_35: change_label_zeros_35_mutation,
    MutationTypesEnum.change_label_zeros_40: change_label_zeros_40_mutation,
    MutationTypesEnum.change_label_zeros_45: change_label_zeros_45_mutation,
    MutationTypesEnum.change_label_zeros_50: change_label_zeros_50_mutation,
    MutationTypesEnum.change_label_zeros_55: change_label_zeros_55_mutation,

    MutationTypesEnum.change_label_diff_5: change_label_diff_5_mutation,
    MutationTypesEnum.change_label_diff_10: change_label_diff_10_mutation,
    MutationTypesEnum.change_label_diff_15: change_label_diff_15_mutation,
    MutationTypesEnum.change_label_diff_20: change_label_diff_20_mutation,
    MutationTypesEnum.change_label_diff_25: change_label_diff_25_mutation,
    MutationTypesEnum.change_label_diff_30: change_label_diff_30_mutation,
    MutationTypesEnum.change_label_diff_35: change_label_diff_35_mutation,
    MutationTypesEnum.change_label_diff_40: change_label_diff_40_mutation,
    MutationTypesEnum.change_label_diff_45: change_label_diff_45_mutation,
    MutationTypesEnum.change_label_diff_50: change_label_diff_50_mutation,
    MutationTypesEnum.change_label_diff_55: change_label_diff_55_mutation,

    MutationTypesEnum.dense_edge_5: dense_edge_5_mutation,
MutationTypesEnum.dense_edge_10: dense_edge_10_mutation,
MutationTypesEnum.dense_edge_15: dense_edge_15_mutation,
MutationTypesEnum.dense_edge_20: dense_edge_20_mutation,
MutationTypesEnum.dense_edge_25: dense_edge_25_mutation,
MutationTypesEnum.dense_edge_30: dense_edge_30_mutation,
MutationTypesEnum.dense_edge_35: dense_edge_35_mutation,
MutationTypesEnum.dense_edge_40: dense_edge_40_mutation,
MutationTypesEnum.dense_edge_45: dense_edge_45_mutation,
MutationTypesEnum.dense_edge_50: dense_edge_50_mutation,
MutationTypesEnum.dense_edge_55: dense_edge_55_mutation,

    MutationTypesEnum.path_edge_5: path_edge_5_mutation,
MutationTypesEnum.path_edge_10: path_edge_10_mutation,
MutationTypesEnum.path_edge_15: path_edge_15_mutation,
MutationTypesEnum.path_edge_20: path_edge_20_mutation,
MutationTypesEnum.path_edge_25: path_edge_25_mutation,
MutationTypesEnum.path_edge_30: path_edge_30_mutation,
MutationTypesEnum.path_edge_35: path_edge_35_mutation,
MutationTypesEnum.path_edge_40: path_edge_40_mutation,
MutationTypesEnum.path_edge_45: path_edge_45_mutation,
MutationTypesEnum.path_edge_50: path_edge_50_mutation,
MutationTypesEnum.path_edge_55: path_edge_55_mutation,

    MutationTypesEnum.cycle_edge_5: cycle_edge_5_mutation,
MutationTypesEnum.cycle_edge_10: cycle_edge_10_mutation,
MutationTypesEnum.cycle_edge_15: cycle_edge_15_mutation,
MutationTypesEnum.cycle_edge_20: cycle_edge_20_mutation,
MutationTypesEnum.cycle_edge_25: cycle_edge_25_mutation,
MutationTypesEnum.cycle_edge_30: cycle_edge_30_mutation,
MutationTypesEnum.cycle_edge_35: cycle_edge_35_mutation,
MutationTypesEnum.cycle_edge_40: cycle_edge_40_mutation,
MutationTypesEnum.cycle_edge_45: cycle_edge_45_mutation,
MutationTypesEnum.cycle_edge_50: cycle_edge_50_mutation,
MutationTypesEnum.cycle_edge_55: cycle_edge_55_mutation,




    MutationTypesEnum.batch_edge_5: batch_edge_5_mutation,
    MutationTypesEnum.batch_edge_10: batch_edge_10_mutation,
    MutationTypesEnum.batch_edge_15: batch_edge_15_mutation,
    MutationTypesEnum.batch_edge_20: batch_edge_20_mutation,
    MutationTypesEnum.batch_edge_25: batch_edge_25_mutation,
    MutationTypesEnum.batch_edge_30: batch_edge_30_mutation,
    MutationTypesEnum.batch_edge_35: batch_edge_35_mutation,
    MutationTypesEnum.batch_edge_40: batch_edge_40_mutation,
    MutationTypesEnum.batch_edge_45: batch_edge_45_mutation,
    MutationTypesEnum.batch_edge_50: batch_edge_50_mutation,
    MutationTypesEnum.batch_edge_55: batch_edge_55_mutation,

    MutationTypesEnum.single_drop: single_drop_mutation,

    MutationTypesEnum.single_change: single_change_mutation,
    MutationTypesEnum.change_label: change_label,

#MutationTypesEnum.change_label_to_1: change_label_to_1,
#MutationTypesEnum.change_label_to_0: change_label_to_0,
#MutationTypesEnum.change_label_to_diff: change_label_to_diff,

}


simple_mutation_set = (
    MutationTypesEnum.tree_growth,
    MutationTypesEnum.single_add,
    MutationTypesEnum.single_change,
    MutationTypesEnum.single_drop,
    MutationTypesEnum.single_edge,
)


rich_mutation_set = (
    MutationTypesEnum.simple,
    MutationTypesEnum.reduce,
    MutationTypesEnum.growth,
    MutationTypesEnum.local_growth
)
