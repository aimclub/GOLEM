import glob
import os
import uuid
import dill as pickle
# from deepdiff import DeepDiff

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

from golem.core.adapter import BaseOptimizationAdapter, IdentityAdapter
from golem.core.dag.graph import Graph
from golem.core.dag.graph_verifier import GraphVerifier, VerifierRuleType
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.log import default_log
from golem.core.optimisers.advisor import DefaultChangeAdvisor
from golem.core.optimisers.optimization_parameters import OptimizationParameters
from golem.core.optimisers.genetic.evaluation import DelegateEvaluator
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import GraphFunction, Objective, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory, OptNodeFactory
from golem.core.optimisers.random_graph_factory import RandomGraphFactory, RandomGrowthGraphFactory


STRUCTURAL_DIVERSITY_FREQUENCY_CHECK = 5


def do_nothing_callback(*args, **kwargs):
    pass


@dataclass
class AlgorithmParameters:
    """Base class for definition of optimizers-specific parameters.
    Can be extended for custom optimizers.

    :param multi_objective: defines if the optimizer must be multi-criterial
    :param offspring_rate: offspring rate used on next population
    :param pop_size: initial population size
    :param max_pop_size: maximum population size; optional, if unspecified, then population size is unbound
    :param adaptive_depth: flag to enable adaptive configuration of graph depth
    :param adaptive_depth_max_stagnation: max number of stagnating populations before adaptive depth increment
    """

    multi_objective: bool = False
    offspring_rate: float = 0.5
    pop_size: int = 20
    max_pop_size: Optional[int] = 55
    adaptive_depth: bool = False
    adaptive_depth_max_stagnation: int = 3
    structural_diversity_frequency_check: int = STRUCTURAL_DIVERSITY_FREQUENCY_CHECK


@dataclass
class GraphGenerationParams:
    """
    This dataclass is for defining the parameters using in graph generation process

    :param adapter: instance of domain graph adapter for adaptation
     between domain and optimization graphs
    :param rules_for_constraint: collection of constraints for graph verification
    :param advisor: instance providing task and context-specific advices for graph changes
    :param node_factory: instance for generating new nodes in the process of graph search
    :param remote_evaluator: instance of delegate evaluator for evaluation of graphs
    """
    adapter: BaseOptimizationAdapter
    verifier: GraphVerifier
    advisor: DefaultChangeAdvisor
    node_factory: OptNodeFactory
    random_graph_factory: RandomGraphFactory
    remote_evaluator: Optional[DelegateEvaluator] = None

    def __init__(self, adapter: Optional[BaseOptimizationAdapter] = None,
                 rules_for_constraint: Sequence[VerifierRuleType] = tuple(DEFAULT_DAG_RULES),
                 advisor: Optional[DefaultChangeAdvisor] = None,
                 node_factory: Optional[OptNodeFactory] = None,
                 random_graph_factory: Optional[RandomGraphFactory] = None,
                 available_node_types: Optional[Sequence[Any]] = None,
                 remote_evaluator: Optional[DelegateEvaluator] = None,
                 ):
        self.adapter = adapter or IdentityAdapter()
        self.verifier = GraphVerifier(rules_for_constraint, self.adapter)
        self.advisor = advisor or DefaultChangeAdvisor()
        self.remote_evaluator = remote_evaluator
        if node_factory:
            self.node_factory = node_factory
        elif available_node_types:
            self.node_factory = DefaultOptNodeFactory(available_node_types)
        else:
            self.node_factory = DefaultOptNodeFactory()
        self.random_graph_factory = random_graph_factory or RandomGrowthGraphFactory(self.verifier,
                                                                                     self.node_factory)


class GraphOptimizer:
    """
    Base class of graph optimizer. It allows to find the optimal solution using specified metric (one or several).
    To implement the specific optimisation method,
    the abstract method 'optimize' should be re-defined in the ancestor class
    (e.g.  PopulationalOptimizer, RandomSearchGraphOptimiser, etc).

    :param objective: objective for optimisation
    :param initial_graphs: graphs which were initialized outside the optimizer
    :param requirements: implementation-independent requirements for graph optimizer
    :param graph_generation_params: parameters for new graph generation
    :param graph_optimizer_params: parameters for specific implementation of graph optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Optional[Sequence[Union[Graph, Any]]] = None,
                 # TODO: rename params to avoid confusion
                 requirements: Optional[OptimizationParameters] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 graph_optimizer_params: Optional[AlgorithmParameters] = None,
                 saved_state_path='saved_optimisation_state/main'):
        self.log = default_log(self)
        self._objective = objective
        self.initial_graphs = graph_generation_params.adapter.adapt(initial_graphs) if initial_graphs else None
        self.requirements = requirements or OptimizationParameters()
        self.graph_generation_params = graph_generation_params or GraphGenerationParams()
        self.graph_optimizer_params = graph_optimizer_params or AlgorithmParameters()
        self._iteration_callback: IterationCallback = do_nothing_callback
        self._history = OptHistory(objective.get_info(), requirements.history_dir) \
            if requirements and requirements.keep_history else None
        self._saved_state_path = saved_state_path
        self._run_id = str(uuid.uuid1())
    @property
    def objective(self) -> Objective:
        """Returns Objective of this optimizer with information about metrics used."""
        return self._objective

    @property
    def history(self) -> Optional[OptHistory]:
        """Returns optimization history"""
        return self._history

    @abstractmethod
    def optimise(self, objective: ObjectiveFunction) -> Sequence[OptGraph]:
        """
        Method for running of optimization using specified algorithm.
        :param objective: objective function that specifies optimization target
        :return: sequence of the best graphs
        """
        pass

    def set_iteration_callback(self, callback: Optional[Callable]):
        """Set optimisation callback that is called at the end of
        each iteration, with the next generation passed as argument.
        Resets the callback if None is passed."""
        self._iteration_callback = callback or do_nothing_callback

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        """Set or reset (with None) post-evaluation callback
        that's called on each graph after its evaluation."""
        pass

    # def save(self, saved_state_path):
    #     folder_path = os.path.dirname(os.path.abspath(saved_state_path))
    #     if not os.path.isdir(folder_path):
    #         os.makedirs(folder_path)
    #         self.log.info(f'Created directory for saving optimization state: {folder_path}')
    #     # pickle.settings['recurse'] = True
    #     with open(saved_state_path, 'wb') as f:
    #         pickle.dump(self.__dict__, f, 2)
    #         pickle.dump(self.__class__, f, 2)
    #         # pickle.dump(self, f, 2)
    #
    # def load(self, saved_state_path):
    #     with open(saved_state_path, 'rb') as f:
    #         self.__dict__.update(pickle.load(f))
    #         self.__class__ = pickle.load(f)

    def save(self, saved_state_path):
        folder_path = os.path.dirname(os.path.abspath(saved_state_path))
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
            self.log.info(f'Created directory for saving optimization state: {folder_path}')
        # pickle.settings['recurse'] = True
        with open(saved_state_path, 'wb') as f:
            # pickle.dump(self, f, 2)
            pickle.dump(self.__dict__, f, 2)
            pickle.dump(self.__class__, f, 2)

    def load(self, saved_state_path):
        with open(saved_state_path, 'rb') as f:
            dict_state = pickle.load(f)
            class_state = pickle.load(f)
            self.__dict__.update(dict_state)
            self.__class__ = class_state


    # def __getstate__(self):
    #     dict_state = self.__dict__.copy()
    #     class_state = self.__class__
    #     del class_state['_progressbar']
    #     return (dict_state, class_state)
    #
    # def __setstate__(self, state):
    #     dict_state, class_state = state
    #     self.__dict__.update(dict_state)
    #     self.__class__ = class_state

    def _find_latest_dir(self, directory: str) -> str:
        return max([os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(
            os.path.join(directory, d))], key=os.path.getmtime)

    def _find_latest_file_in_dir(self, directory: str) -> str:
        return max(glob.glob(os.path.join(directory, '*')), key=os.path.getmtime)

    # def compobj(self, pickleobj): #?? probably remove
    #     dif_vars = []
    #     for element in self.__dict__:
    #         if self.__dict__[element] != pickleobj[element]:
    #             dif_vars.append(element)
    #     print(DeepDiff(self.__dict__, pickleobj))
    #     print(dif_vars)


IterationCallback = Callable[[PopulationT, GraphOptimizer], Any]
