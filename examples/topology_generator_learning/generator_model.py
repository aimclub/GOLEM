

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

def change_mean(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        node.content['mean'] = randint(-1000,1000)
    except Exception as ex:
        graph.log.warn(f'Incorrect mutation: {ex}')
    return graph


def change_var(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        node.content['var'] = randint(1,100)
    except Exception as ex:
        graph.log.warn(f'Incorrect var: {ex}')
    return graph


def change_cov(graph: GeneratorModel, **kwargs):
   
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

    

def model_topology(graph:GeneratorModel):
    sample_data = pd.DataFrame()
    structure = []
    info = {'types':{}, 'signs':{}}
    for i, node in enumerate(graph.nodes):
        info['types'][node.content['name']] = 'cont'
        info['signs'][node.content['name']] = 'neg'
        mean = node.content['mean']
        var = node.content['var']
        sample_data[node.content['name']] = np.random.normal(loc=mean, scale=math.sqrt(var), size=100)
        if node.nodes_from:
            for parent in node.nodes_from:
                structure.append((parent.content['name'], node.content['name']))
    bn = ContinuousBN()
    bn.add_nodes(info)
    bn.set_structure(edges=structure)
    bn.fit_parameters(sample_data)
    sample = bn.sample(100)
    new_sample = pd.DataFrame()
    final_columns = ['X1', 'X2']
    for t in range(5):
        columns = []
        for c in range(2):
            columns.append('X'+str(c)+'_'+'t'*t)
        df = sample[columns]
        df.columns = final_columns
        new_sample = pd.concat([new_sample, df])
    
    new_sample = new_sample
    
    #тут надо посчитать топлогию на new_sample и вернуть её

   
    return ()


def save_in_bn(graph:GeneratorModel, name):
    sample_data = pd.DataFrame()
    structure = []
    info = {'types':{}, 'signs':{}}
    for i, node in enumerate(graph.nodes):
        info['types'][node.content['name']] = 'cont'
        info['signs'][node.content['name']] = 'neg'
        mean = node.content['mean']
        var = node.content['var']
        sample_data[node.content['name']] = np.random.normal(loc=mean, scale=math.sqrt(var), size=100)
        if node.nodes_from:
            for parent in node.nodes_from:
                structure.append((parent.content['name'], node.content['name']))
    bn = ContinuousBN()
    bn.add_nodes(info)
    bn.set_structure(edges=structure)
    bn.fit_parameters(sample_data)
    bn.save(name)



    

    
def optimisation_metric_topology(generator:GeneratorModel, target_topology):
    generator_topology = model_topology(generator)
    return 0#math.fabs(generator_topology - target_topology)

