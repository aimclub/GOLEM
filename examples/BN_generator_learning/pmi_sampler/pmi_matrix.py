import numpy as np
from itertools import combinations
from tabulate import tabulate
from tqdm.auto import tqdm


def get_empty_pmi_matrix(last_index):
    return np.zeros((last_index, last_index))


def count_pmi(value_pair, encoded_data):
    variable_1, value_1 = value_pair[0][0], value_pair[0][1]
    variable_2, value_2 = value_pair[1][0], value_pair[1][1]

    joint_num = len(encoded_data[(encoded_data[variable_1] == value_1) & (encoded_data[variable_2] == value_2)])
    value_1_num = len(encoded_data[encoded_data[variable_1] == value_1])
    value_2_num = len(encoded_data[encoded_data[variable_2] == value_2])

    if value_1_num * value_2_num != 0:
        return len(encoded_data) * joint_num / (value_1_num * value_2_num)
    else:
        return np.nan

    
def get_pmi_matrix(fs_data, node_value_index, all_values):
    pmi_matrix = get_empty_pmi_matrix(len(all_values))

    all_value_pairs = list(combinations(all_values, 2))

    value_set = dict()
    for value in all_values:
        cur_set = fs_data[fs_data[value[0]] == value[1]].itertuples(index=False)
        value_set[value] = set(cur_set)

    print("\nPMI matrix in process...")
    for values_pair in tqdm(all_value_pairs):
        node_1, value_1 = values_pair[0][0], values_pair[0][1]
        node_2, value_2 = values_pair[1][0], values_pair[1][1]

        set1 = value_set[(node_1, value_1)]
        set2 = value_set[(node_2, value_2)]
        joint_set = set1.intersection(set2)

        pmi = len(fs_data)*len(joint_set) / (len(set1) * len(set2))

        column = node_value_index[node_1][value_1]
        row = node_value_index[node_2][value_2]

        pmi_matrix[column, row] = pmi

    pmi_matrix = np.maximum(pmi_matrix, pmi_matrix.transpose())
    # table = tabulate(pmi_matrix)
    # print(table)

    return pmi_matrix
