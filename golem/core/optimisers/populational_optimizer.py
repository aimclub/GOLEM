import os
import time
from abc import abstractmethod
from datetime import timedelta, datetime
from random import choice
from typing import Any, Optional, Sequence, Dict

from tqdm import tqdm

from golem.core.constants import MIN_POP_SIZE
from golem.core.dag.graph import Graph
from golem.core.optimisers.archive import GenerationKeeper
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher, SequentialDispatcher
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.objective import GraphFunction, ObjectiveFunction
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer, AlgorithmParameters
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.utilities.grouped_condition import GroupedCondition
from golem.core.paths import default_data_dir


class PopulationalOptimizer(GraphOptimizer):
    """
    Base class of populational optimizer.
    PopulationalOptimizer implements all basic methods for optimization not related to evolution process
    to experiment with other kinds of evolution optimization methods
    It allows to find the optimal solution using specified metric (one or several).
    To implement the specific evolution strategy, implement `_evolution_process`.

    Args:
         objective: objective for optimization
         initial_graphs: graphs which were initialized outside the optimizer
         requirements: implementation-independent requirements for graph optimizer
         graph_generation_params: parameters for new graph generation
         graph_optimizer_params: parameters for specific implementation of graph optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Graph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: Optional['AlgorithmParameters'] = None,
                 use_saved_state: bool = False,
                 saved_state_path: str = 'saved_optimisation_state/main/populational_optimiser',
                 saved_state_file: str = None):

        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params,
                         saved_state_path)

        if use_saved_state:
            # add logging!!!!
            print('USING SAVED STATE')
            if saved_state_file:
                if os.path.isfile(saved_state_file):
                    current_saved_state_path = saved_state_file
                else:
                    # add logging
                    raise SystemExit('ERROR: Could not restore saved optimisation state: '
                                     f'given file with saved state {saved_state_file} not found.')
            else:
                try:
                    full_state_path = os.path.join(default_data_dir(), self._saved_state_path)
                    current_saved_state_path = self._find_latest_file_in_dir(self._find_latest_dir(full_state_path))
                except (ValueError, FileNotFoundError):
                    # add logging
                    raise SystemExit('ERROR: Could not restore saved optimisation state: '
                                     f'path with saved state {full_state_path} not found.')
            try:
                self.load(current_saved_state_path)
            except Exception as e:
                # add logging
                raise SystemExit('ERROR: Could not restore saved optimisation state from {full_state_path}.'
                                 f'If saved state file is broken remove it manually from the saved state dir or'
                                 f'pass a valid saved state filepath.'
                                 f'Full error message: {e}')

            saved_state_timestamp = datetime.fromtimestamp(os.path.getmtime(current_saved_state_path))
            downtime = datetime.now() - saved_state_timestamp

            self.requirements.early_stopping_iterations = requirements.early_stopping_iterations
            self.requirements.early_stopping_timeout = requirements.early_stopping_timeout
            self.requirements.num_of_generations = requirements.num_of_generations
            self.requirements.timeout = requirements.timeout
            self.requirements.static_individual_metadata['evaluation_time_iso'] = datetime.isoformat(
                datetime.fromisoformat(self.requirements.static_individual_metadata['evaluation_time_iso']) + downtime)

            elapsed_time: timedelta = saved_state_timestamp - self.timer.start_time
            timeout = self.requirements.timeout - elapsed_time
            self.timer = OptimisationTimer(timeout=timeout)

            self.requirements.early_stopping_timeout = self.requirements.early_stopping_timeout - \
                                                       elapsed_time.total_seconds() / 60
            self.requirements.timeout = self.requirements.timeout - timedelta(seconds=elapsed_time.total_seconds())

            self._edit_time_vars(self.__dict__, saved_state_timestamp, downtime, elapsed_time)
            # stag_time_delta = saved_state_timestamp - self.generations._stagnation_start_time
            # self.generations._stagnation_start_time = datetime.now() - stag_time_delta
        else:
            self.population = None
            self.generations = GenerationKeeper(self.objective, keep_n_best=requirements.keep_n_best)
            self.timer = OptimisationTimer(timeout=self.requirements.timeout)

            dispatcher_type = MultiprocessingDispatcher if self.requirements.parallelization_mode == 'populational' \
                else SequentialDispatcher

            self.eval_dispatcher = dispatcher_type(adapter=graph_generation_params.adapter,
                                                   n_jobs=requirements.n_jobs,
                                                   graph_cleanup_fn=_try_unfit_graph,
                                                   delegate_evaluator=graph_generation_params.remote_evaluator)

            # in how many generations structural diversity check should be performed
            self.gen_structural_diversity_check = self.graph_optimizer_params.structural_diversity_frequency_check

        self.use_saved_state = use_saved_state

        # early_stopping_iterations and early_stopping_timeout may be None, so use some obvious max number
        max_stagnation_length = requirements.early_stopping_iterations or requirements.num_of_generations
        max_stagnation_time = requirements.early_stopping_timeout or self.timer.timeout

        self.stop_optimization = \
            GroupedCondition(results_as_message=True).add_condition(
                lambda: self.timer.is_time_limit_reached(self.current_generation_num),
                'Optimisation stopped: Time limit is reached'
            ).add_condition(
                lambda: (requirements.num_of_generations is not None and
                         self.current_generation_num >= requirements.num_of_generations + 1),
                'Optimisation stopped: Max number of generations reached'
            ).add_condition(
                lambda: (max_stagnation_length is not None and
                         self.generations.stagnation_iter_count >= max_stagnation_length),
                'Optimisation finished: Early stopping iterations criteria was satisfied (stagnation_iter_count)'
            ).add_condition(
                lambda: self.generations.stagnation_time_duration >= max_stagnation_time,
                'Optimisation finished: Early stopping timeout criteria was satisfied (stagnation_time_duration)'
            )

    @property
    def current_generation_num(self) -> int:
        return self.generations.generation_num

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        # Redirect callback to evaluation dispatcher
        self.eval_dispatcher.set_graph_evaluation_callback(callback)

    def optimise(self, objective: ObjectiveFunction, save_state_delta: int = 60) -> Sequence[Graph]:

        saved_state_path = os.path.join(default_data_dir(), self._saved_state_path, self._run_id)

        # eval_dispatcher defines how to evaluate objective on the whole population
        evaluator = self.eval_dispatcher.dispatch(objective, self.timer)

        last_write_time = datetime.now()

        with self.timer, self._progressbar:
            if not self.use_saved_state:
                self._initial_population(evaluator)  # !!!!!

            while not self.stop_optimization():
                try:
                    new_population = self._evolve_population(evaluator)
                    if self.gen_structural_diversity_check != -1 \
                            and self.generations.generation_num % self.gen_structural_diversity_check == 0 \
                            and self.generations.generation_num != 0:
                        new_population = self.get_structure_unique_population(new_population, evaluator)
                except EvaluationAttemptsError as ex:
                    self.log.warning(f'Composition process was stopped due to: {ex}')
                    return [ind.graph for ind in self.best_individuals]
                # Adding of new population to history
                self._update_population(new_population)
                delta = datetime.now() - last_write_time
                if delta.seconds >= save_state_delta:
                    self.save(os.path.join(saved_state_path, f'{str(round(time.time()))}.pkl'))
                    print(os.path.join(saved_state_path, f'{str(round(time.time()))}.pkl'))  # log!!!!!!!!!
                    last_write_time = datetime.now()
        self._update_population(self.best_individuals, 'final_choices')
        self.save(os.path.join(saved_state_path, f'{str(round(time.time()))}.pkl'))
        print(os.path.join(saved_state_path, f'{str(round(time.time()))}.pkl'))  # log!!!!!!!!!
        return [ind.graph for ind in self.best_individuals]

    @property
    def best_individuals(self):
        return self.generations.best_individuals

    @abstractmethod
    def _initial_population(self, evaluator: EvaluationOperator):
        """ Initializes the initial population """
        raise NotImplementedError()

    @abstractmethod
    def _evolve_population(self, evaluator: EvaluationOperator) -> PopulationT:
        """ Method realizing full evolution cycle """
        raise NotImplementedError()

    def _extend_population(self, pop: PopulationT, target_pop_size: int) -> PopulationT:
        """ Extends population to specified `target_pop_size`. """
        n = target_pop_size - len(pop)
        extended_population = list(pop)
        extended_population.extend([Individual(graph=choice(pop).graph) for _ in range(n)])
        return extended_population

    def _update_population(self, next_population: PopulationT, label: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        self.generations.append(next_population)
        self._log_to_history(next_population, label, metadata)
        self._iteration_callback(next_population, self)
        self.population = next_population

        self.log.info(f'Generation num: {self.current_generation_num} size: {len(next_population)}')
        self.log.info(f'Best individuals: {str(self.generations)}')
        if self.generations.stagnation_iter_count > 0:
            self.log.info(f'no improvements for {self.generations.stagnation_iter_count} iterations')
            self.log.info(f'spent time: {round(self.timer.minutes_from_start, 1)} min')

    def _log_to_history(self, population: PopulationT, label: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        self.history.add_to_history(population, label, metadata)
        self.history.add_to_archive_history(self.generations.best_individuals)
        if self.requirements.history_dir:
            self.history.save_current_results(self.requirements.history_dir)

    def get_structure_unique_population(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        """ Increases structurally uniqueness of population to prevent stagnation in optimization process.
        Returned population may be not entirely unique, if the size of unique population is lower than MIN_POP_SIZE. """
        unique_population_with_ids = {ind.graph.descriptive_id: ind for ind in population}
        unique_population = list(unique_population_with_ids.values())

        # if size of unique population is too small, then extend it to MIN_POP_SIZE by repeating individuals
        if len(unique_population) < MIN_POP_SIZE:
            unique_population = self._extend_population(pop=unique_population, target_pop_size=MIN_POP_SIZE)
        return evaluator(unique_population)

    @property
    def _progressbar(self):
        if self.requirements.show_progress:
            bar = tqdm(total=self.requirements.num_of_generations,
                       desc='Generations', unit='gen', initial=1)
        else:
            # disable call to tqdm.__init__ to avoid stdout/stderr access inside it
            # part of a workaround for https://github.com/nccr-itmo/FEDOT/issues/765
            bar = EmptyProgressBar()
        return bar

    def _edit_time_vars(self, var, saved_state_timestamp, downtime, prev_run_len, varname=None, parent=None):
        '''
        @param var: initial object    can be any type
        @param saved_state_timestamp: the timecode of the saved state file     datetime.datetime
        @param downtime: the amount of time between runs   downtime datetime.timedelta
        @param prev_run_len: the amount of time the previous run took    downtime datetime.timedelta
        @param varname: name of the dict or object element that is passed as var   str
        @param parent: the dict or object that contains var
        @return: nothing, modifies the var object in place
        '''
        if isinstance(var, dict):
            for key, item in var.items():
                if isinstance(item, GraphRequirements):
                    var[key] = self.requirements
                elif isinstance(item, OptimisationTimer):
                    var[key] = self.timer
                else:
                    self._edit_time_vars(item, saved_state_timestamp, downtime, prev_run_len, key, var)

        elif isinstance(var, list) or isinstance(var, tuple) or isinstance(var, set):
            for el in var:
                self._edit_time_vars(el, saved_state_timestamp, downtime, prev_run_len)

        else:
            try:
                var_dict = var.__dict__
            except Exception:
                var_dict = None
            if var_dict:
                self._edit_time_vars(var_dict, saved_state_timestamp, downtime, prev_run_len)
            elif varname:
                # try:
                #     if ('time' in varname) and (varname != 'computation_time_in_seconds') and (varname != 'evaluation_time_iso'):
                #         print(varname, type(varname), var, type(var))
                # except Exception:
                #     pass
                if isinstance(varname, str) and 'stagnation_start_time' in varname:
                    stag_time_delta = saved_state_timestamp - var
                    parent[varname] = datetime.now() - stag_time_delta
                elif isinstance(varname, str) and 'evaluation_time_iso' in varname:
                    parent[varname] = datetime.isoformat(datetime.fromisoformat(var) + downtime)
                # elif isinstance(varname, str) and 'init_time' in varname:
                #     parent[varname] = var + downtime
                # elif isinstance(varname, str) and 'timeout' in varname:
                #     parent[varname] = var - prev_run_len


# TODO: remove this hack (e.g. provide smth like FitGraph with fit/unfit interface)
def _try_unfit_graph(graph: Any):
    if hasattr(graph, 'unfit'):
        graph.unfit()


class EmptyProgressBar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class EvaluationAttemptsError(Exception):
    """ Number of evaluation attempts exceeded """

    def __init__(self, *args):
        self.message = args[0] or None

    def __str__(self):
        return self.message or 'Too many fitness evaluation errors.'
