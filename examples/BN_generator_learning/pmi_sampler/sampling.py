from numpy import sum, random, zeros
from pandas import DataFrame
from tqdm.auto import tqdm
from pmi_sampler.similarity_evaluation import get_similarity_for_value


def get_value_probabilities(node, cur_state, node_processor):
    value_indexes = node_processor.node_value_index[node].values()
    values = node_processor.node_value_index[node].keys()

    parents = node_processor.node_parents[node]
    parents_state_indexes = [node_processor.node_value_index[parent][cur_state[parent]] for parent in parents]
    value_similarities = [get_similarity_for_value(value_index, parents_state_indexes, node_processor) for value_index in value_indexes]
    total_sum = sum(value_similarities)
    value_probabilities = [cur_sim / total_sum for cur_sim in value_similarities]
    value_probabilities = dict(zip(values, value_probabilities))

    return value_probabilities


def sample_node(node, cur_state, node_processor):
    if node in node_processor.non_parents_nodes:
        probs = node_processor.non_parents_node_probs[node]
    else:
        probs = get_value_probabilities(node, cur_state, node_processor)

    values = list(probs.keys())
    probs = list(probs.values())

    new_value = random.choice(values, p=probs)
    return new_value


def get_initial_state(node_processor):
    values = [0] * len(node_processor.sampled_nodes)
    sampled_nodes_state = dict(zip(node_processor.sampled_nodes, values))

    state = sampled_nodes_state | node_processor.evidence

    return state


def add_state(i, new_state, all_states):
    for element in all_states:
        all_states[element][i] = new_state[element]


def sample(n, node_processor):
    cur_state = get_initial_state(node_processor)

    states = [(node.name, zeros(n)) for node in node_processor.bn.nodes]
    states = dict(states)

    for i in tqdm(range(n)):
        cur_node = random.choice(node_processor.sampled_nodes)
        new_value = sample_node(cur_node, cur_state, node_processor)
        cur_state[cur_node] = new_value
        add_state(i, cur_state, states)

    result = DataFrame(states)
    return result
