from typing import Sequence, Optional, Any, List

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax


def plot_action_values(stats: Sequence[Sequence[float]],
                       action_tags: Optional[Sequence[Any]] = None,
                       size: float = 5.,
                       titles: List[str] = None):
    # Plot stackplot of how action expectations and probabilities changed
    x = np.arange(len(stats))
    y = np.array(stats).T
    y_prob = softmax(y, axis=0)

    labels = [str(action) for action in action_tags]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(size * 2, size))
    ax0, ax1 = ax

    for ys, label in zip(y, labels):
        ax0.plot(x, ys, label=label)
        ax0.grid()
    ax1.stackplot(x, y_prob, labels=labels)

    if not titles:
        expectation_values_title = 'Action Expectation Values'
        probabilities_title = 'Action Probabilities'
    else:
        expectation_values_title, probabilities_title = titles
    ax0.set_title(expectation_values_title, size=10)
    ax0.set_xlabel('Generation')
    ax0.set_ylabel('Reward Expectation')
    ax1.set_title(probabilities_title, size=10)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Probability')
    ax1.set(ylim=(0, 1.0), yticks=np.linspace(0., 1., 21))

    if action_tags:
        ax[0].legend(loc='upper right')
