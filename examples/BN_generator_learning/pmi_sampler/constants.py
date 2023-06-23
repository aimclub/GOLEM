from bamt.networks.discrete_bn import DiscreteBN


class Config:
    n_initial = None
    bn_path = None
    bn = None
    evidences = {}

    @classmethod
    def set_initial_config(cls, n_initial=1000, bn_path=""):
        cls.n_initial = n_initial
        cls.bn_path = bn_path

        bn = DiscreteBN()
        bn.load(cls.bn_path)
        cls.bn = bn

    @classmethod
    def set_evidences(cls, evidences):
        cls.evidences = evidences


class Storage:
    variable_value_index = None
    embeddings_matrix = None
    node_neighbours = None
    node_index = None
    non_parents_nodes = None

    sampled_nodes = None
    similarity_matrix = None

