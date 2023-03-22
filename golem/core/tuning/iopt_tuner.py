from dataclasses import dataclass, field
from typing import List, Dict, Generic, Tuple, Any

import numpy as np
from iOpt.method.listener import ConsoleFullOutputListener
from iOpt.problem import Problem
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import Point, FunctionValue

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveEvaluate
from golem.core.tuning.tuner_interface import BaseTuner, DomainGraphForTune


@dataclass
class IOptProblemParameters:
    float_params_names: List[str] = field(default_factory=list)
    discrete_params_names: List[str] = field(default_factory=list)
    lower_bounds_of_float_params: List[float] = field(default_factory=list)
    upper_bounds_of_float_params: List[float] = field(default_factory=list)
    discrete_params_vals: List[List[Any]] = field(default_factory=list)

    def append_float_parameter(self, name: str, bounds: List):
        self.float_params_names.append(name)
        self.lower_bounds_of_float_params.append(bounds[0])
        self.upper_bounds_of_float_params.append(bounds[1])

    def append_discrete_parameters(self, name: str, values: List):
        self.discrete_params_names.append(name)
        self.discrete_params_vals.append(values)


class GolemProblem(Problem, Generic[DomainGraphForTune]):
    def __init__(self, graph: DomainGraphForTune, objective_evaluate: ObjectiveEvaluate,
                 problem_parameters: IOptProblemParameters,
                 n_jobs: int = -1):
        super().__init__()
        self.n_jobs = n_jobs
        objective_evaluate.eval_n_jobs = self.n_jobs
        self.objective_evaluate = objective_evaluate.evaluate
        self.graph = graph

        self.numberOfObjectives = 1
        self.numberOfConstraints = 0

        self.discreteVariableNames = problem_parameters.discrete_params_names
        self.discreteVariableValues = problem_parameters.discrete_params_vals
        self.numberOfDiscreteVariables = len(self.discreteVariableNames)

        self.floatVariableNames = problem_parameters.float_params_names
        self.lowerBoundOfFloatVariables = problem_parameters.lower_bounds_of_float_params
        self.upperBoundOfFloatVariables = problem_parameters.upper_bounds_of_float_params
        self.numberOfFloatVariables = len(self.floatVariableNames)

        self._default_metric_value = np.inf

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        new_params = get_parameters_dict_from_iopt_point(point, self.floatVariableNames, self.discreteVariableNames)
        BaseTuner.set_arg_graph(self.graph, new_params)
        graph_fitness = self.objective_evaluate(self.graph)
        metric_value = graph_fitness.value if graph_fitness.valid else self._default_metric_value
        functionValue.value = metric_value
        return functionValue


class IOptTuner(BaseTuner):
    def __init__(self, objective_evaluate: ObjectiveEvaluate,
                 adapter: BaseOptimizationAdapter = None,
                 iterations=100,
                 n_jobs: int = -1):
        super().__init__(objective_evaluate, adapter, iterations, n_jobs)

    def tune(self, graph: DomainGraphForTune) -> DomainGraphForTune:
        graph = self.adapter.adapt(graph)

        problem_parameters, initial_parameters = self._get_parameters_for_tune(graph)

        # initial_point = Point(floatVariables=initial_parameters['float'],
        #                       discreteVariables=initial_parameters['discrete'])

        # float_params_names = parameters_dict['float'].keys()
        # discrete_params_names = parameters_dict['discrete'].keys()
        #
        # lower_bounds = [bounds[0] for bounds in parameters_dict['float'].values()]
        # upper_bounds = [bounds[1] for bounds in parameters_dict['float'].values()]
        #
        # bounds_of_float_params = {'low': lower_bounds, 'upper': upper_bounds}
        # discrete_params_vals = list(parameters_dict['discrete'].values())

        # problem = GolemProblem(graph, self.objective_evaluate, float_params_names,
        #                        discrete_params_names, bounds_of_float_params, discrete_params_vals)

        initial_point = Point(**initial_parameters) if initial_parameters else None

        problem = GolemProblem(graph, self.objective_evaluate, problem_parameters)

        method_params = SolverParameters(r=np.double(3.0), itersLimit=self.iterations, startPoint=initial_point)
        solver = Solver(problem, parameters=method_params)

        cfol = ConsoleFullOutputListener(mode='full')
        solver.AddListener(cfol)

        solution = solver.Solve()
        best_point = solution.bestTrials[0].point
        best_params = get_parameters_dict_from_iopt_point(best_point, problem_parameters.float_params_names,
                                                          problem_parameters.discrete_params_names)
        tuned_graph = self.set_arg_graph(graph, best_params)
        tuned_graph = self.adapter.restore(tuned_graph)
        return tuned_graph

    def _get_parameters_for_tune(self, graph: OptGraph) -> Tuple[IOptProblemParameters, dict]:
        """ Method for defining the search space

        Args:
            graph: graph to be tuned

        Returns:
            parameters_dict: dict with operation names and parameters
            initial_parameters: dict with initial parameters of the graph
        """
        parameters_dict = IOptProblemParameters()
        # must be full
        initial_parameters = {}

        return parameters_dict, initial_parameters


def get_parameters_dict_from_iopt_point(point: Point, float_params_names: List[str], discrete_params_names: List[str]) \
        -> Dict[str, Any]:
    float_params = dict(zip(float_params_names, point.floatVariables)) \
        if point.floatVariables is not None else {}
    discrete_params = dict(zip(discrete_params_names, point.discreteVariables)) \
        if point.discreteVariables is not None else {}
    params_dict = {**float_params, **discrete_params}
    return params_dict
