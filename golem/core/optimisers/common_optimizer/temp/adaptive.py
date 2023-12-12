from typing import Any, Dict, List

from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import Task
from golem.core.optimisers.optimizer import OptimizationParameters, GraphGenerationParams, AlgorithmParameters
from golem.core.optimisers.genetic.parameters.population_size import AdaptivePopulationSize as OldAdaptivePopulationSize
from golem.core.optimisers.genetic.parameters.population_size import init_adaptive_pop_size
from golem.core.optimisers.genetic.parameters.graph_depth import AdaptiveGraphDepth as OldAdaptiveGraphDepth
from golem.core.optimisers.genetic.parameters.operators_prob import AdaptiveVariationProb as OldAdaptiveVariationProb
from golem.core.optimisers.genetic.parameters.operators_prob import init_adaptive_operators_prob
from golem.core.optimisers.genetic.operators.mutation import Mutation as OldMutation


class AdaptiveParametersTask(Task):
    """
    This class is for storing a state of OptimizationParameters,
    GraphGenerationParams and AlgorithmParameters in CommonOptimizerParameters.
    :param parameters: instance of CommonOptimizerParameters containing the initial parameters
    """

    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__()
        self.requirements = parameters.requirements
        self.graph_optimizer_params = parameters.graph_optimizer_params
        self.graph_generation_params = parameters.graph_generation_params
        self.population = parameters.population
        self.generations = parameters.generations
        self.stages = parameters.stages

    def update_parameters(self, parameters: 'CommonOptimizerParameters') -> 'CommonOptimizerParameters':
        """
        Update the parameters in CommonOptimizerParameters with stored
        OptimizationParameters, GraphGenerationParams and AlgorithmParameters values.
        :param parameters: instance of CommonOptimizerParameters to update
        :return: updated parameters object
        """
        parameters.population = self.population
        parameters.requirements = self.requirements
        parameters.graph_optimizer_params = self.graph_optimizer_params
        parameters.graph_generation_params = self.graph_generation_params
        return parameters


class AdaptivePopulationSize(Node):
    def __init__(self, name: str = 'pop_size'):
        self.name = name
        self._pop_size = None

    def __call__(self, task: AdaptiveParametersTask) -> List[AdaptiveParametersTask]:
        if not isinstance(task, AdaptiveParametersTask):
            raise TypeError(f"task should be `AdaptiveParametersTask`, got {type(task)} instead")
        if self._pop_size is None:
            self._pop_size: OldAdaptivePopulationSize = init_adaptive_pop_size(
                    task.graph_optimizer_params,
                    task.generations
                )
        task.graph_optimizer_params.pop_size = self._pop_size.next(task.population)
        return [task]


class AdaptiveGraphDepth(Node):
    def __init__(self, name: str = 'max_depth'):
        self.name = name
        self._max_depth = None

    def __call__(self, task: AdaptiveParametersTask) -> List[AdaptiveParametersTask]:
        if not isinstance(task, AdaptiveParametersTask):
            raise TypeError(f"task should be `AdaptiveParametersTask`, got {type(task)} instead")
        if self._max_depth is None:
            self._max_depth: OldAdaptiveGraphDepth = OldAdaptiveGraphDepth(
                    task.population,
                    start_depth=task.requirements.start_depth,
                    max_depth=task.requirements.max_depth,
                    max_stagnation_gens=task.graph_optimizer_params.adaptive_depth_max_stagnation,
                    adaptive=task.graph_optimizer_params.adaptive_depth
                )

        task.requirements.max_depth = self._max_depth.next()
        return [task]


class AdaptiveOperatorsProb(Node):
    # TODO add reinitialization of probs in nodes
    def __init__(self, name: str = 'operators_prob'):
        self.name = name
        self._operators_prob = None

    def __call__(self, task: AdaptiveParametersTask) -> List[AdaptiveParametersTask]:
        if not isinstance(task, AdaptiveParametersTask):
            raise TypeError(f"task should be `AdaptiveParametersTask`, got {type(task)} instead")
        if self._operators_prob is None:
            self._operators_prob: OldAdaptiveVariationProb = init_adaptive_operators_prob(task.graph_optimizer_params)
        probs = self._operators_prob.next(task.population)
        task.graph_optimizer_params.mutation_prob, task.graph_optimizer_params.crossover_prob = probs

        for stage in task.stages:
            for node in stage.nodes:
                if node.__class__.__name__ == 'Mutation':
                    if hasattr(node, '_mutation'):
                        node._mutation = None
        return [task]
