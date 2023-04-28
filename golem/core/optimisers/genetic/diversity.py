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


def plot_diversity_dynamic_gif(history: OptHistory,
                               filename: Optional[str] = None,
                               nbins: int = 8) -> FuncAnimation:
    metric_names = history.objective.metric_names
    # dtype=float removes None, puts np.nan
    # indexed by [population, individual, metric]
    fitness_values = np.array([[ind.fitness.values for ind in pop]
                               for pop in history.individuals], dtype=float)
    # indexed by [population, metric, individual]
    fitness_distrib = fitness_values.swapaxes(1, 2)

    # Setup the plot
    fig, axs = plt.subplots(ncols=len(metric_names))
    fig.suptitle('Population diversity by metric')
    np.ravel(axs)
    for ax, metric_name in zip(axs, metric_names):
        ax: plt.Axes
        ax.set_title(metric_name)
        ax.set_xlabel('Metric value')
        ax.set_ylabel('Individuals')
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
        for artist, metric_distrib in zip(artists, next_pop_metrics):
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
