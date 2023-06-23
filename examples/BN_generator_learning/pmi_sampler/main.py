from bamt.networks.discrete_bn import DiscreteBN
from bamt.networks.continuous_bn import ContinuousBN
from data_preparing import encode_data, get_node_value_index, get_node_value_list
from node_preprocess import NodeProcessor
from pmi_matrix import get_pmi_matrix
from node_embeddings import get_embedding_matrix
from sampling import sample
from similarity_evaluation import get_similarity_matrix
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from pandas import DataFrame


def get_similarities(fs_data):
    discrete_data, node_value = encode_data(fs_data)
    node_value_index = get_node_value_index(node_value)
    all_values = get_node_value_list(node_value)
    pmi_matrix = get_pmi_matrix(discrete_data, node_value_index, all_values)
    embedding_matrix = get_embedding_matrix(pmi_matrix)
    similarity_matrix = get_similarity_matrix(embedding_matrix)

    return similarity_matrix, node_value_index, discrete_data


def set_node_processor(node_processor, similarity_matrix, node_value_index, discrete_data):
    node_processor.initial_setting()
    node_processor.set_non_parents_node_probs_from_data(discrete_data)
    node_processor.set_similarity_matrix(similarity_matrix)
    node_processor.set_node_value_index(node_value_index)

    return node_processor


n_samples = 100000
evidence = {}
bins = 100

bn_path = "structures/arth150_10.json"
data_path = "data/arth150_10.csv"
sampled_data_path = "sampled_data/sampled_data.csv"

bn = ContinuousBN()  # use_mixture=True
bn.load(bn_path)

data = read_csv(data_path, index_col=0)
bn.fit_parameters(data)

discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
discretizer.fit(data)
disc_data = discretizer.transform(data)
disc_data = DataFrame(columns=data.columns, data=disc_data, dtype=int)

similarity_matrix, node_value_index, encoded_data = get_similarities(disc_data)
node_processor = NodeProcessor(bn, evidence)
node_processor = set_node_processor(node_processor, similarity_matrix, node_value_index, encoded_data)

df = sample(n_samples, node_processor)
df.to_csv(sampled_data_path)
