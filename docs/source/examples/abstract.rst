Abstract Graph Search
=====================

This example is about the search for graphs that are closest to a target graph by some distance metric.

Target graphs are generated using `networkx`. Both ordered and random graphs are used.

For objective are used graph theoretic metrics defined on features of the graph and graph matrices. In particular, distance between spectres of graph matrices are used. Spectre of the graph defines its characteristic features (like number of components, clustering, and more). For details see the `this publication <https://api.semanticscholar.org/CorpusID:118711105>`_ about graph similarity.

Full code of the example can be found `in this python file <https://github.com/aimclub/GOLEM/blob/main/examples/synthetic_graph_evolution/abstract_graph_search.py>`_.
