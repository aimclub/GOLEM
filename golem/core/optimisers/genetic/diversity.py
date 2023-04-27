from typing import Union, Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import ArrayLike

from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimizer import GraphOptimizer, IterationCallback


def compute_fitness_diversity(population: PopulationT) -> np.ndarray:
    """Returns numpy array of standard deviations of fitness values."""
    # substitutes None values
    fitness_values = np.array([ind.fitness.values for ind in population], dtype=float)
    # compute std along each axis while ignoring nan-s
    diversity = np.nanstd(fitness_values, axis=0)
    return diversity


def plot_diversity_dynamic(history: OptHistory, show=True):
    np_history = np.array([compute_fitness_diversity(pop)
                           for pop in history.individuals])
    labels = history.objective.metric_names
    xs = np.arange(len(np_history))
    ys = {label: np_history[:, i] for i, label in enumerate(labels)}

    fig, ax = plt.subplots()
    for label, metric_std in ys.items():
        ax.plot(xs, metric_std, label=label)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Std')
    ax.grid()
    ax.legend()
    fig.suptitle('Population diversity (Fitness std)')

    if show:
        plt.show()
