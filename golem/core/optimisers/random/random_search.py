from random import randint
from typing import Optional

from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS
from golem.core.dag.graph import Graph
from golem.core.dag.graph_utils import distance_to_root_level
from golem.core.optimisers.fitness import null_fitness
from golem.core.optimisers.genetic.evaluation import SimpleDispatcher
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphOptimizer, GraphGenerationParams
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.utilities.grouped_condition import GroupedCondition


class RandomGraphFactory:
    def __init__(self,
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams):
        self.requirements = requirements
        self.graph_generation_params = graph_generation_params

    def __call__(self):
        return self.random_graph()

    def random_graph(self, max_depth: Optional[int] = None) -> Graph:
        return random_graph(self.graph_generation_params, self.requirements, max_depth)


class RandomSearchOptimizer(GraphOptimizer):

    def __init__(self, objective: Objective,
                 requirements: GraphRequirements,
                 graph_generation_params: Optional[GraphGenerationParams] = None):
        super().__init__(objective, requirements=requirements, graph_generation_params=graph_generation_params)
        self._factory = RandomGraphFactory(self.requirements, self.graph_generation_params)
        self.timer = OptimisationTimer(timeout=self.requirements.timeout)
        self.current_iteration_num = 0
        self.stop_optimization = \
            GroupedCondition(results_as_message=True).add_condition(
                lambda: self.timer.is_time_limit_reached(self.current_iteration_num),
                'Optimisation stopped: Time limit is reached'
            ).add_condition(
                lambda: requirements.num_of_generations is not None and
                        self.current_iteration_num >= requirements.num_of_generations,
                'Optimisation stopped: Max number of iterations reached')

    def optimise(self, objective: ObjectiveFunction) -> Graph:

        best_fitness = null_fitness()
        dispatcher = SimpleDispatcher(self.graph_generation_params.adapter)
        evaluator = dispatcher.dispatch(objective, self.timer)
        self.current_iteration_num = 0

        with self.timer as t:
            while not self.stop_optimization():
                new_graph = self._factory()
                new_ind = Individual(new_graph)
                evaluator([new_ind])
                if new_ind.fitness > best_fitness:
                    best_fitness = new_ind.fitness
                    best_ind = new_ind

                self.history.add_to_history([best_ind])
                self.log.info(f'Spent time: {round(self.timer.minutes_from_start, 1)} min')
                self.log.info(f'Iter {self.current_iteration_num}: '
                              f'best fitness {self._objective.format_fitness(best_fitness)},'
                              f'try {self._objective.format_fitness(new_ind.fitness)} with num nodes {new_graph.length}')
                self.current_iteration_num += 1
        return best_ind.graph


def random_graph(graph_generation_params: GraphGenerationParams,
                 requirements: GraphRequirements,
                 max_depth: Optional[int] = None) -> OptGraph:
    max_depth = max_depth if max_depth else requirements.max_depth
    is_correct_graph = False
    graph = None
    n_iter = 0
    node_factory = graph_generation_params.node_factory

    while not is_correct_graph:
        graph = OptGraph()
        graph_root = node_factory.get_node()
        graph.add_node(graph_root)
        if requirements.max_depth > 1:
            graph_growth(graph, graph_root, node_factory, requirements, max_depth)

        is_correct_graph = graph_generation_params.verifier(graph)
        n_iter += 1
        if n_iter > MAX_GRAPH_GEN_ATTEMPTS:
            raise ValueError(f'Could not generate random graph for {n_iter} '
                             f'iterations with requirements {requirements}')
    return graph


def graph_growth(graph: OptGraph,
                 node_parent: OptNode,
                 node_factory: OptNodeFactory,
                 requirements: GraphRequirements,
                 max_depth: int):
    """Function create a graph and links between nodes"""
    offspring_size = randint(requirements.min_arity, requirements.max_arity)

    for offspring_node in range(offspring_size):
        node = node_factory.get_node()
        node_parent.nodes_from.append(node)
        graph.add_node(node)

        height = distance_to_root_level(graph, node_parent)
        is_max_depth_exceeded = height >= max_depth - 1
        if not is_max_depth_exceeded:
            if randint(0, 1):
                graph_growth(graph, node, node_factory, requirements, max_depth)
