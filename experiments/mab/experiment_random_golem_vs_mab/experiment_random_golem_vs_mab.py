import os.path
from datetime import timedelta
from functools import partial
from typing import Optional, List, Sequence

import pandas as pd

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.utils import draw_graphs_subplots
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.graph import Graph
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.core.optimisers.populational_optimizer import PopulationalOptimizer
from golem.core.paths import project_root
from golem.metrics.edit_distance import tree_edit_dist
from golem.metrics.graph_metrics import spectral_dist, degree_distance, size_diff


class ExperimentLauncher:
    def __init__(self, setup_name: str, launch_num: int, path_to_save: str,
                 datasets: Optional[List[str]] = None):
        # some experiments can be conducted without exact datasets, however, the structure of
        # result saving for ExperimentAnalyzer considers dataset in result saving directory tree
        if datasets is None:
            datasets = ['default']

        self.path_to_save = path_to_save
        os.makedirs(self.path_to_save, exist_ok=True)

        self.launch_num = launch_num
        self.setup_name = setup_name
        self.datasets = datasets

    def launch(self, ):
        for dataset in self.datasets:
            for i in range(self.launch_num):
                path_to_save_launch_data = os.path.join(self.path_to_save, self.setup_name, dataset, str(i))
                found_graphs, optimizer, target_graph = self._launch(dataset=dataset, size=10, timeout=2, node_types=['x'],
                                                                     launch_num=i, path_to_save=path_to_save_launch_data)
                self.save_experiment_results(path_to_save=path_to_save_launch_data,
                                             found_graphs=found_graphs, target_graph=target_graph,
                                             optimizer=optimizer)

    def _launch(self, dataset: str, size: int, timeout: int, node_types: List[str], launch_num: int, path_to_save: str):
        target_graph = generate_labeled_graph('tree', size, node_labels=node_types)

        # Generate initial population with small tree graphs
        initial_graphs = [generate_labeled_graph('tree', 5, node_types) for _ in range(10)]
        # Setup objective: edit distance to target graph
        objective = Objective(
            quality_metrics={
                'sp_adj': partial(spectral_dist, target_graph, kind='adjacency'),
                'sp_lapl': partial(spectral_dist, target_graph, kind='laplacian'),
            },
            complexity_metrics={
                'graph_size': partial(size_diff, target_graph),
                'degree': partial(degree_distance, target_graph),
            },
            is_multi_objective=True
        )

        # Setup optimization parameters
        requirements = GraphRequirements(
            early_stopping_iterations=10,
            timeout=timedelta(minutes=timeout),
            n_jobs=1,
            # history_dir=path_to_save_launch_data,
            agent_dir=path_to_save
        )
        gp_params = GPAlgorithmParameters(
            adaptive_mutation_type=MutationAgentTypeEnum.bandit,
            genetic_scheme_type=GeneticSchemeTypesEnum.generational,
            max_pop_size=50,
            mutation_types=[MutationTypesEnum.single_add,
                            MutationTypesEnum.single_drop,
                            MutationTypesEnum.single_change],
            multi_objective=objective.is_multi_objective,
            crossover_types=[CrossoverTypesEnum.none]
        )
        graph_gen_params = GraphGenerationParams(
            adapter=BaseNetworkxAdapter(),  # Example works with NetworkX graphs
            rules_for_constraint=DEFAULT_DAG_RULES,  # We don't want cycles in the graph
            available_node_types=node_types  # Node types that can appear in graphs
        )
        all_parameters = (requirements, graph_gen_params, gp_params)

        # Build and run the optimizer
        optimiser = EvoGraphOptimizer(objective, initial_graphs, *all_parameters)
        found_graphs = optimiser.optimise(objective)
        return found_graphs, optimiser, target_graph

    def save_experiment_results(self, path_to_save: str, found_graphs: Sequence[Graph], target_graph: Graph,
                                optimizer: PopulationalOptimizer, is_visualize: bool = True):

        os.makedirs(path_to_save, exist_ok=True)
        path_for_pics = os.path.join(path_to_save, 'pics')

        # save final graphs
        for i, ind in enumerate(optimizer.best_individuals):
            ind.save(os.path.join(path_to_save, f'{i}_ind.json'))
            ind.graph.show(os.path.join(path_for_pics, f'{i}_ind.png'))

        # save metrics
        obj_names = optimizer.objective.metric_names
        fitness = dict.fromkeys(obj_names)
        for ind in optimizer.best_individuals:
            for j, metric in enumerate(obj_names):
                if not fitness[metric]:
                    fitness[metric] = []
                fitness[metric].append(ind.fitness.values[j])
        df_metrics = pd.DataFrame(fitness)
        df_metrics.to_csv(os.path.join(path_to_save, 'metrics.csv'))

        # save history
        history = optimizer.history
        os.makedirs(path_to_save, exist_ok=True)
        history.save(os.path.join(path_to_save, 'history.json'))
        if is_visualize:
            found_graph = found_graphs[0] if isinstance(found_graphs, Sequence) else found_graphs
            found_nx_graph = BaseNetworkxAdapter().restore(found_graph)
            draw_graphs_subplots(target_graph, found_nx_graph,
                                 titles=['Target Graph', 'Found Graph'], show=False)
            diversity_path = os.path.join(path_to_save, 'diversity')
            os.makedirs(diversity_path, exist_ok=True)
            diversity_filename = (f'./results/diversity_hist_{graph_name}_n{num_nodes}.gif')
            history.show.diversity_population(save_path=diversity_filename)
            history.show.diversity_line(show=False)
            history.show.fitness_line()


if __name__ == '__main__':
    path_to_save = os.path.join(project_root(), 'experiments', 'mab',
                                'experiment_random_golem_vs_mab', 'results')
    launcher = ExperimentLauncher(setup_name='random', launch_num=1, path_to_save=path_to_save)
    launcher.launch()
