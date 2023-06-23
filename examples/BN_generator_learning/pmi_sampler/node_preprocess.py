class NodeProcessor:
    def __init__(self, bn, evidence):
        self.bn = bn
        self.evidence = evidence

        self.node_parents = None
        self.non_parents_nodes = None
        self.non_parents_node_probs = None
        self.sampled_nodes = None

        self.node_value_index = None
        self.similarity_matrix = None

    def initial_setting(self):
        self.set_node_parents()
        self.set_sampled_nodes()
        self.set_non_parents_nodes()

    def set_node_parents(self):
        nodes = [node.name for node in self.bn.nodes]
        node_parents = [node.disc_parents + node.cont_parents for node in self.bn.nodes]

        self.node_parents = dict(zip(nodes, node_parents))

    @staticmethod
    def get_probs_from_data(node, data):
        values = data[node].unique()
        probs = dict([(value, 0) for value in values])

        for item_value in data[node]:
            probs[item_value] += 1

        for value in values:
            probs[value] /= len(data)

        return probs

    def set_non_parents_node_probs_from_data(self, data):
        if self.node_parents is None:
            raise Exception("Non-parents nodes must be set firstly")

        node_probs = dict()

        for node in data.columns:
            node_probs[node] = self.get_probs_from_data(node, data)

        self.non_parents_node_probs = node_probs

    def set_non_parents_node_probs_from_distribution(self):
        node_probs = dict()

        for node in self.bn.nodes:
            probs = self.bn.distributions[node.name]["cprob"]
            node_probs[node] = probs

        self.non_parents_node_probs = node_probs

    def set_sampled_nodes(self):
        sampled_nodes = [node.name for node in self.bn.nodes if node.name not in self.evidence.keys()]

        self.sampled_nodes = sampled_nodes

    def set_non_parents_nodes(self):
        non_parents_nodes = [node.name for node in self.bn.nodes if not node.disc_parents and not node.cont_parents]

        self.non_parents_nodes = non_parents_nodes

    def set_similarity_matrix(self, similarity_matrix):
        self.similarity_matrix = similarity_matrix

    def set_node_value_index(self, node_value_index):
        self.node_value_index = node_value_index
