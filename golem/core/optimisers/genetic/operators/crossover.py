from copy import deepcopy
from random import choice, random, sample
from math import ceil
from itertools import chain
from typing import Callable, Union, Iterable, Tuple, TYPE_CHECKING

from golem.core.adapter import register_native
from golem.core.dag.graph_utils import nodes_from_layer, node_depth
from golem.core.optimisers.genetic.gp_operators import equivalent_subtree, replace_subtrees
from golem.core.optimisers.genetic.operators.operator import PopulationT, Operator
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters


class CrossoverTypesEnum(Enum):
    subtree = 'subtree'
    one_point = "one_point"
    none = 'none'
    exchange_edges = 'exchange_edges'
    exchange_parents_one = 'exchange_parents_one'
    exchange_parents_both = 'exchange_parents_both'


CrossoverCallable = Callable[[OptGraph, OptGraph, int], Tuple[OptGraph, OptGraph]]


class Crossover(Operator):
    def __init__(self,
                 parameters: 'GPAlgorithmParameters',
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams):
        super().__init__(parameters, requirements)
        self.graph_generation_params = graph_generation_params

    def __call__(self, population: PopulationT) -> PopulationT:
        if len(population) == 1:
            new_population = population
        else:
            new_population = []
            for ind_1, ind_2 in Crossover.crossover_parents_selection(population):
                new_population += self._crossover(ind_1, ind_2)
        return new_population

    @staticmethod
    def crossover_parents_selection(population: PopulationT) -> Iterable[Tuple[Individual, Individual]]:
        return zip(population[::2], population[1::2])

    def _crossover(self, ind_first: Individual, ind_second: Individual) -> Tuple[Individual, Individual]:
        crossover_type = choice(self.parameters.crossover_types)

        if self._will_crossover_be_applied(ind_first.graph, ind_second.graph, crossover_type):
            crossover_func = self._get_crossover_function(crossover_type)
            for _ in range(self.parameters.max_num_of_operator_attempts):
                first_object = deepcopy(ind_first.graph)
                second_object = deepcopy(ind_second.graph)
                new_graphs = crossover_func(first_object, second_object, max_depth=self.requirements.max_depth)
                are_correct = all(self.graph_generation_params.verifier(new_graph) for new_graph in new_graphs)
                if are_correct:
                    parent_individuals = (ind_first, ind_second)
                    new_individuals = self._get_individuals(new_graphs, parent_individuals, crossover_type)
                    return new_individuals

            self.log.debug('Number of crossover attempts exceeded. '
                           'Please check optimization parameters for correctness.')

        return ind_first, ind_second

    def _get_crossover_function(self, crossover_type: Union[CrossoverTypesEnum, Callable]) -> Callable:
        if isinstance(crossover_type, Callable):
            crossover_func = crossover_type
        else:
            crossover_func = self._crossover_by_type(crossover_type)
        return self.graph_generation_params.adapter.adapt_func(crossover_func)

    def _crossover_by_type(self, crossover_type: CrossoverTypesEnum) -> CrossoverCallable:
        crossovers = {
            CrossoverTypesEnum.subtree: subtree_crossover,
            CrossoverTypesEnum.one_point: one_point_crossover,
            CrossoverTypesEnum.exchange_edges: exchange_edges_crossover,
            CrossoverTypesEnum.exchange_parents_one: exchange_parents_one_crossover,
            CrossoverTypesEnum.exchange_parents_both: exchange_parents_both_crossover            
        }
        if crossover_type in crossovers:
            return crossovers[crossover_type]
        else:
            raise ValueError(f'Required crossover type is not found: {crossover_type}')

    def _get_individuals(self, new_graphs: Tuple[OptGraph, OptGraph], parent_individuals: Tuple[Individual, Individual],
                         crossover_type: Union[CrossoverTypesEnum, Callable]) -> Tuple[Individual, Individual]:
        operator = ParentOperator(type_='crossover',
                                  operators=str(crossover_type),
                                  parent_individuals=parent_individuals)
        metadata = self.requirements.static_individual_metadata
        return tuple(Individual(graph, operator, metadata=metadata) for graph in new_graphs)

    def _will_crossover_be_applied(self, graph_first, graph_second, crossover_type) -> bool:
        return not (graph_first is graph_second or
                    random() > self.parameters.crossover_prob or
                    crossover_type is CrossoverTypesEnum.none)


@register_native
def subtree_crossover(graph_first: OptGraph, graph_second: OptGraph, max_depth: int) -> Tuple[OptGraph, OptGraph]:
    """Performed by the replacement of random subtree
    in first selected parent to random subtree from the second parent"""
    random_layer_in_graph_first = choice(range(graph_first.depth))
    min_second_layer = 1 if random_layer_in_graph_first == 0 and graph_second.depth > 1 else 0
    random_layer_in_graph_second = choice(range(min_second_layer, graph_second.depth))

    node_from_graph_first = choice(nodes_from_layer(graph_first, random_layer_in_graph_first))
    node_from_graph_second = choice(nodes_from_layer(graph_second, random_layer_in_graph_second))

    replace_subtrees(graph_first, graph_second, node_from_graph_first, node_from_graph_second,
                     random_layer_in_graph_first, random_layer_in_graph_second, max_depth)

    return graph_first, graph_second


@register_native
def one_point_crossover(graph_first: OptGraph, graph_second: OptGraph, max_depth: int) -> Tuple[OptGraph, OptGraph]:
    """Finds common structural parts between two trees, and after that randomly
    chooses the location of nodes, subtrees of which will be swapped"""
    pairs_of_nodes = equivalent_subtree(graph_first, graph_second)
    if pairs_of_nodes:
        node_from_graph_first, node_from_graph_second = choice(pairs_of_nodes)

        layer_in_graph_first = graph_first.depth - node_depth(node_from_graph_first)
        layer_in_graph_second = graph_second.depth - node_depth(node_from_graph_second)

        replace_subtrees(graph_first, graph_second, node_from_graph_first, node_from_graph_second,
                         layer_in_graph_first, layer_in_graph_second, max_depth)
    return graph_first, graph_second


@register_native
def exchange_edges_crossover(graph_first: OptGraph, graph_second: OptGraph, max_depth):
    
    old_edges1 = []
    old_edges2 = []

    edges_1 = graph_first.operator.get_edges()
    edges_2 = graph_second.operator.get_edges()
    count = ceil(min(len(edges_1), len(edges_2))/2)
    choice_edges_1 = sample(edges_1, count)
    choice_edges_2 = sample(edges_2, count)
    
    for pair in choice_edges_1:
        pair[1].nodes_from.remove(pair[0])
    for pair in choice_edges_2:
        pair[1].nodes_from.remove(pair[0])
    
    old_edges1 = graph_first.operator.get_edges()
    old_edges2 = graph_second.operator.get_edges()

    new_edges_2 = [(graph_second.get_nodes_by_name(str(i[0]))[0], graph_second.get_nodes_by_name(str(i[1]))[0]) for i in choice_edges_1]
    new_edges_1 = [(graph_first.get_nodes_by_name(str(i[0]))[0], graph_first.get_nodes_by_name(str(i[1]))[0]) for i in choice_edges_2]

    for pair in new_edges_1:
        if pair not in old_edges1:
            pair[1].nodes_from.append(pair[0])
    for pair in new_edges_2:
        if pair not in old_edges2:
            pair[1].nodes_from.append(pair[0])                                          
        
    return graph_first, graph_second


@register_native
def exchange_parents_one_crossover(graph_first: OptGraph, graph_second: OptGraph, max_depth):

    edges = graph_second.operator.get_edges()
    flatten_edges = list(chain(*edges))
    nodes_with_parent_or_child=list(set(flatten_edges))
    if nodes_with_parent_or_child!=[]:
        
        selected_node=choice(nodes_with_parent_or_child)
        parents=selected_node.nodes_from
        
        node_from_first_graph=graph_first.get_nodes_by_name(str(selected_node))[0]
        
        node_from_first_graph.nodes_from=[]
        old_edges1 = graph_first.operator.get_edges()
        
        if parents!=[] and parents!=None:
            parents_in_first_graph=[graph_first.get_nodes_by_name(str(i))[0] for i in parents]
            for parent in parents_in_first_graph:
                if (parent, node_from_first_graph) not in old_edges1:
                    node_from_first_graph.nodes_from.append(parent)

    return graph_first, graph_second


@register_native
def exchange_parents_both_crossover(graph_first: OptGraph, graph_second: OptGraph, max_depth):

    parents1 = []
    parents2 = []
    parents_in_first_graph = []
    parents_in_second_graph = []


    edges = graph_second.operator.get_edges()
    flatten_edges = list(chain(*edges))
    nodes_with_parent_or_child=list(set(flatten_edges))
    if nodes_with_parent_or_child!=[]:
        
        selected_node2=choice(nodes_with_parent_or_child)
        parents2=selected_node2.nodes_from
        if parents2!=[] and parents2!=None:
            parents_in_first_graph=[graph_first.get_nodes_by_name(str(i))[0] for i in parents2]

        selected_node1 = graph_first.get_nodes_by_name(str(selected_node2))[0]
        parents1=selected_node1.nodes_from
        if parents1!=[] and parents1!=None:
            parents_in_second_graph=[graph_second.get_nodes_by_name(str(i))[0] for i in parents1]

        if parents1:
            for p in parents1:
                selected_node1.nodes_from.remove(p)
        if parents2:
            for p in parents2:
                selected_node2.nodes_from.remove(p)

        old_edges1 = graph_first.operator.get_edges()
        old_edges2 = graph_second.operator.get_edges()

        if parents_in_first_graph!=[]:
            for parent in parents_in_first_graph:
                if (parent, selected_node1) not in old_edges1:
                    selected_node1.nodes_from.append(parent)

        if parents_in_second_graph!=[]:
            for parent in parents_in_second_graph:
                if (parent, selected_node2) not in old_edges2:
                    selected_node2.nodes_from.append(parent)

    return graph_first, graph_second
