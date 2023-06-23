
import os
import sys

parentdir = os.getcwd()
sys.path.insert(0, parentdir)

from typing import Optional, Union, List
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.dag.graph_delegate import GraphDelegate
from golem.core.dag.linked_graph_node import LinkedGraphNode
from golem.core.dag.linked_graph import LinkedGraph
import numpy as np
import pandas as pd
from gmr import GMM
from bamt.networks.continuous_bn import ContinuousBN
from random import choice, random,randint, sample
import math
from datetime import timedelta
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.objective.objective_eval import ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from itertools import repeat
import networkx as nx
from pmi_sampler.data_preparing import encode_data, get_node_value_index, get_node_value_list
from pmi_sampler.node_preprocess import NodeProcessor
from pmi_sampler.pmi_matrix import get_pmi_matrix
from pmi_sampler.node_embeddings import get_embedding_matrix
from pmi_sampler.sampling import sample
from pmi_sampler.similarity_evaluation import get_similarity_matrix
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer


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

class GeneratorModel(GraphDelegate):
    def __init__(self, nodes: Optional[Union[LinkedGraphNode, List[LinkedGraphNode]]] = None):
        super().__init__(nodes)
        self.unique_pipeline_id = 1

class GeneratorNode(LinkedGraphNode):
    def __str__(self):
        return self.content["name"]

def custom_mutation_change_mean(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        node.content['mean'] = [[randint(-1000,1000)] for _ in range(len(node.content['w']))]
    except Exception as ex:
        graph.log.warn(f'Incorrect mutation: {ex}')
    return graph

def custom_mutation_change_mean_i(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        n = len(node.content['w'])
        random_index = randint(0, n)
        mean = node.content['mean']
        new_mean = []
        for i, m in enumerate(mean):
            if i == random_index:
                new_mean.append([randint(-1000,1000)])
            else:
                new_mean.append(m)
        node.content['mean'] = new_mean
    except Exception as ex:
        graph.log.warn(f'Incorrect mutation: {ex}')
    return graph

def custom_mutation_change_var(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        node.content['var'] = [[[randint(1,50)]] for _ in range(len(node.content['w']))]
    except Exception as ex:
        graph.log.warn(f'Incorrect var: {ex}')
    return graph

def custom_mutation_change_var_i(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        n = len(node.content['w'])
        random_index = randint(0, n)
        var = node.content['var']
        new_var = []
        for i, m in enumerate(var):
            if i == random_index:
                new_var.append([[randint(1,50)]])
            else:
                new_var.append(m)
        node.content['var'] = new_var
    except Exception as ex:
        graph.log.warn(f'Incorrect mutation: {ex}')
    return graph


def custom_crossover_exchange_mean(graph1: GeneratorModel,graph2: GeneratorModel, **kwargs):
    node1 = choice(graph1.nodes)
    node2 = choice(graph2.nodes)
    mean1 = node1.content['mean']
    mean2 = node2.content['mean']
    node1.content['mean'] = mean2
    node2.content['mean'] = mean1
    return graph1, graph2


def custom_crossover_exchange_mean_i(graph1: GeneratorModel,graph2: GeneratorModel, **kwargs):
    node1 = choice(graph1.nodes)
    node2 = choice(graph2.nodes)
    mean1 = node1.content['mean']
    mean2 = node2.content['mean']
    n = len(node1.content['w'])
    random_index = randint(0, n)
    new_mean1 = []
    new_mean2 = []
    for i, m in enumerate(mean1):
        if i == random_index:
            new_mean1.append(mean2[i])
            new_mean2.append(mean1[i])
        else:
            new_mean1.append(m)
            new_mean2.append(mean2[i])
    node1.content['mean'] = new_mean1
    node2.content['mean'] = new_mean2
    return graph1, graph2


def custom_crossover_exchange_var_i(graph1: GeneratorModel,graph2: GeneratorModel, **kwargs):
    node1 = choice(graph1.nodes)
    node2 = choice(graph2.nodes)
    var1 = node1.content['var']
    var2 = node2.content['var']
    n = len(node1.content['w'])
    random_index = randint(0, n)
    new_var1 = []
    new_var2 = []
    for i, m in enumerate(var1):
        if i == random_index:
            new_var1.append(var2[i])
            new_var2.append(var1[i])
        else:
            new_var1.append(m)
            new_var2.append(var2[i])
    node1.content['var'] = new_var1
    node2.content['var'] = new_var2
    return graph1, graph2

def custom_crossover_exchange_var(graph1: GeneratorModel, graph2: GeneratorModel, **kwargs):
    node1 = choice(graph1.nodes)
    node2 = choice(graph2.nodes)
    var1 = node1.content['var']
    var2 = node2.content['var']
    node1.content['var'] = var2
    node2.content['var'] = var1
    return graph1, graph2
    
# models = [np.cos, np.sin, lambda x: x*x, lambda x: 5*x, np.log]
    
# def model_assortativity(graph:GeneratorModel):
#     sample = pd.DataFrame()
#     params_dict = dict()
#     for node in graph.nodes:
#         if node.nodes_from == None or node.nodes_from == []:
#             mu = node.content['mean']
#             var = node.content['var']
#             params_dict[node.content['name']] = {'mean': mu,
#                     'regressor_obj': None,
#                     'regressor': None,
#                     'variance': var,
#                     'serialization': None}
#             sample[node.content['name']] = np.random.normal(mu,var, 500)
#         else:
#             V = np.zeros((500,1))
#             x_column = []
#             for parent in node.nodes_from:
#                 x_column.append(parent.content['name'])
#                 model = node.content['parent_model'][parent.content['name']]
#                 V = V + model(sample[parent.content['name']].values)
#             sample[node.content['name']] = V
#             model = xgboost.XGBRegressor()
#             model.fit(sample[x_column].values, sample[node.content['name']].values)
#             ex_b = pickle.dumps(model, protocol=4)
#             model_ser = ex_b.decode('latin1')
#             params_dict[node.content['name']] = {'mean': None,
#                     'regressor_obj': None,
#                     'regressor': None,
#                     'variance': var,
#                     'serialization': 'pickle'}

def model_assortativity(graph:GeneratorModel):
    sample_data = pd.DataFrame()
    sample_data.index = [i for i in range(5000)]
    structure = []
    info = {'types':{}, 'signs':{}}
    evidence = {}
    for node in graph.nodes:
        info['types'][node.content['name']] = 'cont'
        info['signs'][node.content['name']] = 'neg'
        w = node.content['w']
        mean = node.content['mean']
        var = node.content['var']
        gmm = GMM(n_components=len(w), priors=w, means=mean, covariances=var)
        sample_data[node.content['name']] = gmm.sample(5000)
        for parent in node.nodes_from:
            structure.append((parent.content['name'], node.content['name']))
    discretizer = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='quantile')
    discretizer.fit(sample_data)
    disc_data = discretizer.transform(sample_data)
    disc_data = pd.DataFrame(columns=sample_data.columns, data=disc_data, dtype=int)
    bn = ContinuousBN(use_mixture=False)
    bn.add_nodes(info)
    bn.set_structure(edges=structure)
    bn.fit_parameters(sample_data)
    data = bn.sample(500)
    # similarity_matrix, node_value_index, encoded_data = get_similarities(disc_data)
    # node_processor = NodeProcessor(bn, evidence)
    # node_processor = set_node_processor(node_processor, similarity_matrix, node_value_index, encoded_data)
    # data = sample(500, node_processor)
    # data = pd.DataFrame(columns=data.columns, data=discretizer.inverse_transform(data), dtype=float)
    data['norm'] = data.apply(np.linalg.norm, axis=1)
    for c in data.columns[0:-1]:
        data[c] = data[c] / data['norm']
    data = data.drop(columns=['norm'])
    mean_assort = []
    data = data.values
    for attr_i in data:
        for attr_j in data:
            mean_assort.append((np.dot(attr_i,attr_j)))
    return round(np.mean(mean_assort), 1)

def save_in_bn(graph:GeneratorModel, name):
    sample_data = pd.DataFrame()
    sample_data.index = [i for i in range(5000)]
    structure = []
    info = {'types':{}, 'signs':{}}
    for node in graph.nodes:
        info['types'][node.content['name']] = 'cont'
        info['signs'][node.content['name']] = 'neg'
        w = node.content['w']
        mean = node.content['mean']
        var = node.content['var']
        gmm = GMM(n_components=len(w), priors=w, means=mean, covariances=var)
        sample_data[node.content['name']] = gmm.sample(5000)
        for parent in node.nodes_from:
            structure.append((parent.content['name'], node.content['name']))
    discretizer = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='quantile')
    discretizer.fit(sample_data)
    disc_data = discretizer.transform(sample_data)
    disc_data = pd.DataFrame(columns=sample_data.columns, data=disc_data, dtype=int)
    bn = ContinuousBN(use_mixture=False)
    bn.add_nodes(info)
    bn.set_structure(edges=structure)
    bn.fit_parameters(sample_data)
    bn.save(name)



    

    
def optimisation_metric(generator:GeneratorModel, target_assortativity):
    generator_assort = model_assortativity(generator)
    return math.fabs(generator_assort - target_assortativity)

