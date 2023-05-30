from typing import Union, Callable, Sequence, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
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


class PopulationStatsLogger(IterationCallback):
    """Stores numpy array of standard deviations of fitness values.
    Can be used as an ``IterationCallback`` in optimizer."""
    def __init__(self, stats_fn: Callable[[PopulationT], ArrayLike]):
        self._stats_fn = stats_fn
        self._history = []

    def __call__(self, population: PopulationT, optimizer: GraphOptimizer) -> np.ndarray:
        diversity = self._stats_fn(population)
        self._history.append(diversity)

    def get_history(self) -> np.ndarray:
        np_hist = np.array(self._history)
        return np_hist


def plot_diversity_dynamic_gif(history: OptHistory,
                               filename: Optional[str] = None,
                               fig_size: int = 5,
                               fps: int = 4,
                               ) -> FuncAnimation:
    metric_names = history.objective.metric_names
    # dtype=float removes None, puts np.nan
    # indexed by [population, metric, individual] after transpose (.T)
    pops = history.individuals[1:-1]  # ignore initial pop and final choices
    fitness_distrib = [np.array([ind.fitness.values for ind in pop], dtype=float).T
                       for pop in pops]

    # Define bounds on metrics: find min & max on a flattened view of array
    q = 0.025
    lims_max = np.max([np.quantile(pop, 1 - q, axis=1) for pop in fitness_distrib], axis=0)
    lims_min = np.min([np.quantile(pop, q, axis=1) for pop in fitness_distrib], axis=0)

    # Setup the plot
    fig, axs = plt.subplots(ncols=len(metric_names))
    fig.set_size_inches(fig_size * len(metric_names), fig_size)
    np.ravel(axs)

    # Set update function for updating data on the axes
    def update_axes(iframe: int):
        for i, (ax, metric_distrib) in enumerate(zip(axs, fitness_distrib[iframe])):
            # Clear & Prepare axes
            ax: plt.Axes
            ax.clear()
            ax.set_xlim(0.5, 1.5)
            ax.set_ylim(lims_min[i], lims_max[i])
            ax.set_ylabel('Metric value')
            ax.grid()
            # Plot information
            fig.suptitle(f'Population {iframe+1} diversity by metric')
            ax.set_title(f'{metric_names[i]}, '
                         f'mean={np.mean(metric_distrib).round(3)}, '
                         f'std={np.nanstd(metric_distrib).round(3)}')
            ax.violinplot(metric_distrib,
                          quantiles=[0.25, 0.5, 0.75])

    # Run this function in FuncAnimation
    num_frames = len(fitness_distrib)
    animate = FuncAnimation(
        fig=fig,
        func=update_axes,
        save_count=num_frames,
        interval=200,
    )
    # Save the GIF from animation
    if filename:
        animate.save(filename, fps=fps, dpi=150)
    return animate


def plot_diversity_dynamic(history: OptHistory, show=True):
    labels = history.objective.metric_names
    h = history.individuals[:-1]  # don't consider final choices
    xs = np.arange(len(h))

    # Compute diversity by metrics
    np_history = np.array([compute_fitness_diversity(pop) for pop in h])
    ys = {label: np_history[:, i] for i, label in enumerate(labels)}
    # Compute number of unique individuals, plot
    ratio_unique = [len(set(ind.graph.descriptive_id for ind in pop)) / len(pop) for pop in h]

    fig, ax = plt.subplots()
    for label, metric_std in ys.items():
        ax.plot(xs, metric_std, label=label)

    ax2 = ax.twinx()
    ax2.set_ylabel('Num unique')
    ax2.plot(xs, ratio_unique, label='Num unique', color='tab:gray')

    fig.suptitle('Population diversity')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Std')
    ax.grid()
    ax.legend(loc='upper left')

    if show:
        plt.show()
