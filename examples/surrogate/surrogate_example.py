from examples.synthetic_graph_evolution.experiment import run_experiments
from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from golem.core.optimisers.meta.surrogate_optimizer import SurrogateOptimizer

if __name__ == '__main__':
    results_log = run_experiments(optimizer_setup=graph_search_setup,
                                  optimizer_cls=SurrogateOptimizer,
                                  graph_names=['2ring', 'gnp'],
                                  graph_sizes=[30, 100],
                                  num_trials=1,
                                  trial_timeout=5,
                                  trial_iterations=2000,
                                  visualize=True)
    print(results_log)