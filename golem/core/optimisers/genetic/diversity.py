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
                               nbins: int = 8) -> FuncAnimation:
    metric_names = history.objective.metric_names
    # dtype=float removes None, puts np.nan
    # indexed by [population, metric, individual] after transpose (.T)
    fitness_distrib = [np.array([ind.fitness.values for ind in pop], dtype=float).T
                       for pop in history.individuals]

    # Setup the plot
    # TODO: setup figure size even
    # Define bounds on metrics: find min & max on a flattened view of array
    maxs = np.max([np.max(pop, axis=1) for pop in fitness_distrib], axis=0)
    mins = np.min([np.min(pop, axis=1) for pop in fitness_distrib], axis=0)

    fig, axs = plt.subplots(ncols=len(metric_names))
    fig.suptitle('Population diversity by metric')
    np.ravel(axs)
    for ax, metric_name, min_lim, max_lim in zip(axs, metric_names, mins, maxs):
        ax: plt.Axes
        ax.set_title(metric_name)
        ax.set_xlabel('Metric value')
        ax.set_ylabel('Individuals')
        ax.set_xlim(min_lim, max_lim)
        ax.grid()
        ax.legend()

    # Create artists for each axis
    artists = []
    initial_pop = fitness_distrib[0]
    for ax, pop_metrics in zip(axs, initial_pop):
        _, _, bar_container = ax.hist(pop_metrics, bins=nbins, linewidth=0.5, edgecolor="white")
        artists.append(bar_container)

    # Set update function for updating data on artists of the axes
    def update_axes(iframe: int):
        next_pop_metrics = fitness_distrib[iframe]
        for ax, metric_name, artist, metric_distrib in zip(axs, metric_names, artists, next_pop_metrics):
            ax.set_title(f'{metric_name}, '
                         f'mean={np.mean(metric_distrib).round(3)}, '
                         f'std={np.nanstd(metric_distrib).round(3)}')
            n, _ = np.histogram(metric_distrib, nbins)
            for count, rect in zip(n, artist.patches):
                rect.set_height(count)

    # Run this function in FuncAnimation
    num_frames = len(fitness_distrib)
    animate = FuncAnimation(
        fig=fig,
        func=update_axes,
        save_count=num_frames,
        interval=40,
    )
    # Save the GIF from animation
    if filename:
        animate.save(filename, fps=10, dpi=100)
    return animate


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
