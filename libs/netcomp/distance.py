import numpy as np


def edit_distance(A1, A2):
    """The edit distance between graphs, defined as the number of changes one
    needs to make to put the edge lists in correspondence.

    Parameters
    ----------
    A1, A2 : NumPy matrices
        Adjacency matrices of graphs to be compared

    Returns
    -------
    dist : float
        The edit distance between the two graphs
    """
    dist = np.abs((A1 - A2)).sum() / 2
    return dist
