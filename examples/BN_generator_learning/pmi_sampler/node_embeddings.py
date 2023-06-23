import numpy as np
from numpy.linalg import svd as np_svd
from sklearn.decomposition import TruncatedSVD as sklearn_svd
from scipy.linalg import svd as scipy_svd
from tabulate import tabulate


def get_svd(pmi_matrix):
    # u, s, v = np_svd(pmi_matrix, compute_uv=True)
    # print(u)
    # print(v)
    # svd = sklearn_svd(n_components=5, n_iter=7, random_state=42)
    # svd.fit(pmi_matrix)
    # print(svd.__dict__)

    u, s, v = scipy_svd(pmi_matrix, full_matrices=False)

    # print(s)
    # print(u)
    # print(v)

    return u, s


def get_embedding_matrix(pmi_matrix):
    u, s = get_svd(pmi_matrix)

    s_square_root = np.sqrt(s)
    embedding_matrix = u @ np.diag(s_square_root)

    # table = tabulate(embedding_matrix)
    # print(table)

    return embedding_matrix
