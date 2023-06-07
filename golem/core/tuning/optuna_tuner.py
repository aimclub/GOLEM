from datetime import timedelta
from functools import partial
from typing import Optional, Tuple, Union, Sequence

import optuna
from optuna import Trial, Study
from optuna.trial import FrozenTrial

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.optimisers.fitness import MultiObjFitness
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.tuning.search_space import SearchSpace, get_node_operation_parameter_label
from golem.core.tuning.tuner_interface import BaseTuner, DomainGraphForTune


class OptunaTuner(BaseTuner):
    def __init__(self, objective_evaluate: ObjectiveFunction,
                 search_space: SearchSpace,
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 iterations: int = 100,
                 early_stopping_rounds: Optional[int] = None,
                 timeout: timedelta = timedelta(minutes=5),
                 n_jobs: int = -1,
                 deviation: float = 0.05,
                 objectives_number: int = 1):
        super().__init__(objective_evaluate,
                         search_space,
                         adapter,
                         iterations,
                         early_stopping_rounds,
                         timeout,
                         n_jobs,
                         deviation)
        self.objectives_number = objectives_number

    def tune(self, graph: DomainGraphForTune, show_progress: bool = True) -> \
            Union[DomainGraphForTune, Sequence[DomainGraphForTune]]:
        graph = self.adapter.adapt(graph)
        predefined_objective = partial(self.objective, graph=graph)

        self.init_check(graph)

        study = optuna.create_study(directions=['minimize'] * self.objectives_number)

        # Enqueue initial point to try
        init_parameters = self._get_initial_point(graph)
        if init_parameters:
            study.enqueue_trial(init_parameters)

        study.optimize(predefined_objective,
                       n_trials=self.iterations,
                       n_jobs=self.n_jobs,
                       timeout=self.timeout.seconds,
                       callbacks=[self.early_stopping_callback] if self.early_stopping_rounds else None,
                       show_progress_bar=show_progress)

        best_parameters = study.best_trials[0].params
        tuned_graph = self.set_arg_graph(graph, best_parameters)
        graph = self.final_check(tuned_graph)
        self.was_tuned = True

        tuned_graph = self.adapter.restore(graph)
        return tuned_graph

    def objective(self, trial: Trial, graph: OptGraph) -> Union[float, Tuple[float, ]]:
        new_parameters = self._get_parameters_from_trial(graph, trial)
        new_graph = BaseTuner.set_arg_graph(graph, new_parameters)
        metric_value = self.get_metric_value(new_graph)
        return metric_value

    def _get_parameters_from_trial(self, graph: OptGraph, trial: Trial) -> dict:
        new_parameters = {}
        for node_id, node in enumerate(graph.nodes):
            operation_name = node.name

            # Get available parameters for operation
            tunable_node_params = self.search_space.parameters_per_operation.get(operation_name)

            if tunable_node_params is not None:

                for parameter_name, parameter_properties in tunable_node_params.items():
                    node_op_parameter_name = get_node_operation_parameter_label(node_id, operation_name, parameter_name)

                    parameter_type = parameter_properties.get('type')
                    sampling_scope = parameter_properties.get('sampling-scope')
                    if parameter_type == 'discrete':
                        new_parameters.update({node_op_parameter_name:
                                               trial.suggest_int(node_op_parameter_name, *sampling_scope)})
                    elif parameter_type == 'continuous':
                        new_parameters.update({node_op_parameter_name:
                                               trial.suggest_float(node_op_parameter_name, *sampling_scope)})
                    elif parameter_type == 'categorical':
                        new_parameters.update({node_op_parameter_name:
                                               trial.suggest_categorical(node_op_parameter_name, *sampling_scope)})
        return new_parameters

    def _get_initial_point(self, graph: OptGraph) -> dict:
        initial_parameters = {}
        for node_id, node in enumerate(graph.nodes):
            operation_name = node.name

            # Get available parameters for operation
            tunable_node_params = self.search_space.parameters_per_operation.get(operation_name)

            if tunable_node_params:
                tunable_initial_params = {get_node_operation_parameter_label(node_id, operation_name, p):
                                          node.parameters[p] for p in node.parameters if p in tunable_node_params}
                if tunable_initial_params:
                    initial_parameters.update(tunable_initial_params)
        return initial_parameters

    def early_stopping_callback(self, study: Study, trial: FrozenTrial):
        if self.early_stopping_rounds is not None:
            current_trial_number = trial.number
            best_trial_number = study.best_trial.number
            should_stop = (current_trial_number - best_trial_number) >= self.early_stopping_rounds
            if should_stop:
                self.log.debug(f'Early stopping rounds criteria was reached')
                study.stop()

    def no_parameters_to_optimize_callback(self, study: Study, trial: FrozenTrial, graph: OptGraph):
        parameters = study.trials[-1].params
        if not parameters:
            self._stop_tuning_with_message(f'Graph {graph.graph_description} has no parameters to optimize')
            study.stop()
