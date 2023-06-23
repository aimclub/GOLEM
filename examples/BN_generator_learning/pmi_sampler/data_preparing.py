from bamt.preprocess.discretization import code_categories
from pmi_sampler.constants import Storage, Config


def get_index_generator():
    cur = 0
    while True:
        yield cur
        cur += 1


def encode_data(row_data):
    labels = list(row_data.columns)
    discrete_data, node_value = code_categories(data=row_data,
                                                method="label",
                                                columns=labels)

    node_value = [(node, set(node_value[node].values())) for node in node_value]
    node_value = dict(node_value)

    return discrete_data, node_value


def get_node_value_index(node_value):
    node_value_index = dict()
    index = get_index_generator()

    for node in node_value:
        values = node_value[node]
        value_index = [(value, next(index)) for value in values]
        node_value_index[node] = dict(value_index)

    Storage.variable_value_index = node_value_index

    return node_value_index


def get_node_value_list(node_value):
    node_value_list = []

    for node in node_value:
        node_value_list += [(node, value) for value in node_value[node]]

    return node_value_list


def get_node_parents(bn):
    nodes = [node.name for node in bn.nodes]
    node_parents = [node.disc_parents for node in bn.nodes]

    node_neighbours = dict(zip(nodes, node_parents))
    return node_neighbours


def create_node_index():
    nodes = [node.name for node in Config.bn.nodes]
    indexes = range(len(nodes))

    node_index = dict(zip(nodes, indexes))
    Storage.node_index = node_index


def get_sampled_nodes(bn, evidence):
    sampled_nodes = [node.name for node in bn.nodes if node.name not in evidence.keys()]
    return sampled_nodes


def get_non_parents_nodes(bn):
    non_parents_nodes = [node.name for node in bn.nodes if not node.disc_parents]
    return non_parents_nodes
