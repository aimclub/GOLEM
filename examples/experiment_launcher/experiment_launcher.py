import os

from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from experiments.experiment_launcher import GraphStructureExperimentLauncher
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.paths import project_root

if __name__ == '__main__':
    path_to_save = os.path.join(project_root(), 'examples', 'experiment_launcher', 'experiment_results')

    launcher = GraphStructureExperimentLauncher(optimizer_cls=EvoGraphOptimizer,
                                                graph_names=['gnp', 'tree'],
                                                graph_sizes=[5, 10],
                                                num_trials=2,
                                                trial_iterations=2,
                                                is_save_visualizations=True,
                                                path_to_save=path_to_save)

    launcher.launch(optimizer_setup=graph_search_setup)
