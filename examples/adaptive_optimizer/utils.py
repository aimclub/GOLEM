from typing import Sequence, Optional, Any

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax


def plot_action_values(stats: Sequence[Sequence[float]],
                       action_tags: Optional[Sequence[Any]] = None,
                       size: float = 5.,
                       ylim: float = 5.):
    # Plot stackplot of how action expectations and probabilities changed
    x = np.arange(len(stats))
    y = np.array(stats).T
    y_prob = softmax(y, axis=0)

    labels = [str(action) for action in action_tags]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(size * 2, size))
    ax[0].stackplot(x, y, labels=labels)
    ax[1].stackplot(x, y_prob, labels=labels)

    ax[0].set(ylim=(0, ylim), yticks=np.linspace(0., ylim, 21))
    ax[1].set(ylim=(0, 1.0), yticks=np.linspace(0., 1., 21))

    if action_tags:
        ax[0].legend(loc='upper right')
