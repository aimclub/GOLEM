
import os
import sys

parentdir = os.getcwd()
sys.path.insert(0, parentdir)
from scipy.stats import pearsonr
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
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from scipy.spatial import distance


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
        node.content['mean'] = randint(-1000,1000)
    except Exception as ex:
        graph.log.warn(f'Incorrect mutation: {ex}')
    return graph


def custom_mutation_change_var(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        node.content['var'] = randint(1,100)
    except Exception as ex:
        graph.log.warn(f'Incorrect var: {ex}')
    return graph


def custom_mutation_change_cov(graph: GeneratorModel, **kwargs):
   
    flag = True
    while flag:
        node = choice(graph.nodes)
        if node.nodes_from:
            flag = False
            n = len(node.nodes_from)
            rand_index = randint(0,n-1)
            new_cov = []
            for i, c in enumerate(node.content['cov']):
                if i == rand_index:
                    new_cov.append(randint(-500,500))
                else:
                    new_cov.append(c)
            node.content['cov'] = new_cov
    
    return graph





def custom_crossover_exchange_mean(graph1: GeneratorModel,graph2: GeneratorModel, **kwargs):
    node1 = choice(graph1.nodes)
    node2 = choice(graph2.nodes)
    mean1 = node1.content['mean']
    mean2 = node2.content['mean']
    node1.content['mean'] = mean2
    node2.content['mean'] = mean1
    return graph1, graph2



def custom_crossover_exchange_var(graph1: GeneratorModel, graph2: GeneratorModel, **kwargs):
    node1 = choice(graph1.nodes)
    node2 = choice(graph2.nodes)
    var1 = node1.content['var']
    var2 = node2.content['var']
    node1.content['var'] = var2
    node2.content['var'] = var1
    return graph1, graph2

    

def model_assortativity(graph:GeneratorModel, graph_index: dict, synthetic_graph=None):
    sample_data = pd.DataFrame()
    structure = []
    info = {'types':{}, 'signs':{}}
    cov_matrix = np.zeros((len(graph.nodes),len(graph.nodes)))
    means = []

    for i, node in enumerate(graph.nodes):
        info['types'][node.content['name']] = 'cont'
        info['signs'][node.content['name']] = 'neg'
        mean = node.content['mean']
        var = node.content['var']
        cov = node.content['cov']
        cov_matrix[i,i] = var
        means.append(mean)
        if cov:
            for j, parent in enumerate(node.nodes_from):
                cov_matrix[i, graph_index[parent.content['name']]] = cov[j]
                cov_matrix[graph_index[parent.content['name']], i] = cov[j]
                structure.append((parent.content['name'], node.content['name']))
    cov_matrix = np.dot(cov_matrix, cov_matrix.transpose())
    sample_data = pd.DataFrame(np.random.multivariate_normal(means, cov_matrix, 5000))
    sample_data.columns = list(info['signs'].keys())
    bn = ContinuousBN()
    bn.add_nodes(info)
    bn.set_structure(edges=structure)
    bn.fit_parameters(sample_data)
    if synthetic_graph is None:
        data = bn.sample(100)
        data['norm'] = data.apply(np.linalg.norm, axis=1)
        for c in data.columns[0:-1]:
            data[c] = data[c] / data['norm']
        data = data.drop(columns=['norm'])
        mean_assort = []
        data = data.values
        for attr_i in data:
            for attr_j in data:
                mean_assort.append(np.dot(attr_i,attr_j))
    else:
        data = bn.sample(50)
        data['norm'] = data.apply(np.linalg.norm, axis=1)
        for c in data.columns[0:-1]:
            data[c] = data[c] / data['norm']
        data = data.drop(columns=['norm'])
        mean_assort = []
        data = data.values
        for edge in synthetic_graph:
            i = edge[0]
            j = edge[1]
            mean_assort.append(np.dot(data[i], data[j]))
    return round(np.mean(mean_assort), 1)

def attr_correlation(graph:GeneratorModel, graph_index: dict):
    sample_data = pd.DataFrame()
    structure = []
    info = {'types':{}, 'signs':{}}
    cov_matrix = np.zeros((len(graph.nodes),len(graph.nodes)))
    means = []
    for i, node in enumerate(graph.nodes):
        info['types'][node.content['name']] = 'cont'
        info['signs'][node.content['name']] = 'neg'
        mean = node.content['mean']
        var = node.content['var']
        cov = node.content['cov']
        cov_matrix[i,i] = var
        means.append(mean)
        if node.nodes_from:
            for j, parent in enumerate(node.nodes_from):
                cov_matrix[i, graph_index[parent.content['name']]] = cov[j]
                cov_matrix[graph_index[parent.content['name']], i] = cov[j]
                structure.append((parent.content['name'], node.content['name']))

    cov_matrix = np.dot(cov_matrix, cov_matrix.transpose())
    sample_data = pd.DataFrame(np.random.multivariate_normal(means, cov_matrix, 5000))
    sample_data.columns = list(info['signs'].keys())
    bn = ContinuousBN()
    bn.add_nodes(info)
    bn.set_structure(edges=structure)
    bn.fit_parameters(sample_data)
    correlation_vector = []
    data = bn.sample(100)
    for node in graph.nodes:
        if node.nodes_from:
            for parent in node.nodes_from:
                correlation_vector.append(pearsonr(data[node.content['name']].values, data[parent.content['name']].values)[0])
    return correlation_vector
    


def save_in_bn(graph:GeneratorModel, name, graph_index):
    sample_data = pd.DataFrame()
    structure = []
    info = {'types':{}, 'signs':{}}
    cov_matrix = np.zeros((len(graph.nodes),len(graph.nodes)))
    means = []

    for i, node in enumerate(graph.nodes):
        info['types'][node.content['name']] = 'cont'
        info['signs'][node.content['name']] = 'neg'
        mean = node.content['mean']
        var = node.content['var']
        cov = node.content['cov']
        cov_matrix[i,i] = var
        means.append(mean)
        if cov:
            for j, parent in enumerate(node.nodes_from):
                cov_matrix[i, graph_index[parent.content['name']]] = cov[j]
                cov_matrix[graph_index[parent.content['name']], i] = cov[j]
                structure.append((parent.content['name'], node.content['name']))
    cov_matrix = np.dot(cov_matrix, cov_matrix.transpose())
    sample_data = pd.DataFrame(np.random.multivariate_normal(means, cov_matrix, 5000))
    sample_data.columns = list(info['signs'].keys())
    bn = ContinuousBN()
    bn.add_nodes(info)
    bn.set_structure(edges=structure)
    bn.fit_parameters(sample_data)
    bn.save(name)



    

    
def optimisation_metric_assort(generator:GeneratorModel, target_assortativity, target_correlation, graph_index, synthetic_graph=None):
    generator_assort = model_assortativity(generator, graph_index, synthetic_graph)
    return math.fabs(generator_assort - target_assortativity)


def optimisation_metric_correlaion(generator:GeneratorModel, target_assortativity, target_correlation, graph_index, synthetic_graph=None):
    generator_correlation = attr_correlation(generator, graph_index)
    return distance.euclidean(target_correlation, generator_correlation)

# def optimisation_metric_multi(generator:GeneratorModel, target_assortativity, target_correlation, synthetic_graph=None):
#     assort_metric = optimisation_metric_assort(generator, target_assortativity, synthetic_graph)
#     corr_metric = optimisation_metric_correlaion(generator, target_correlation, synthetic_graph)
#     return (assort_metric+corr_metric) / 2