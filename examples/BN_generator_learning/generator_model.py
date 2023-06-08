
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
import xgboost
import pickle
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
        node.content['mean'] = [[randint(-100,100)] for _ in range(len(node.content['w']))]
    except Exception as ex:
        graph.log.warn(f'Incorrect mutation: {ex}')
    return graph

def custom_mutation_change_var(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        node.content['var'] = [[[randint(1,20)]] for _ in range(len(node.content['w']))]
    except Exception as ex:
        graph.log.warn(f'Incorrect var: {ex}')
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
    sample = pd.DataFrame()
    sample.index = [i for i in range(200)]
    structure = []
    info = {'types':{}, 'signs':{}}
    for node in graph.nodes:
        info['types'][node.content['name']] = 'cont'
        info['signs'][node.content['name']] = 'neg'
        w = node.content['w']
        mean = node.content['mean']
        var = node.content['var']
        gmm = GMM(n_components=len(w), priors=w, means=mean, covariances=var)
        sample[node.content['name']] = gmm.sample(200)
        for parent in node.nodes_from:
            structure.append((parent.content['name'], node.content['name']))
    bn = ContinuousBN(use_mixture=True)
    bn.add_nodes(info)
    bn.set_structure(edges=structure)
    bn.fit_parameters(sample)
    data = bn.sample(100)
    data['norm'] = data.apply(np.linalg.norm, axis=1)
    for c in data.columns[0:-1]:
        data[c] = data[c] / data['norm']
    data = data.drop(columns=['norm'])
    mean_assort = []
    data = data.values
    for attr_i in data:
        for attr_j in data:
            mean_assort.append((np.dot(attr_i,attr_j)))
    return np.mean(mean_assort)

    

    
def optimisation_metric(generator:GeneratorModel, target_assortativity):
    generator_assort = model_assortativity(generator)
    return math.fabs(generator_assort - target_assortativity)

def run_example():

    number_of_components = 2
    number_of_atr = 3
    target_assort = 0.95

    vertices = []
    for i in range(number_of_atr):
        vertices.append('A'+str(i+1))

    structure_parents = {'A1':[], 'A2':['A1'], 'A3':['A1']}

   
    initial = [GeneratorModel(nodes=[GeneratorNode(nodes_from=[],
                                                    content={'name': vertex,
                                                             'w':[1/number_of_components]*number_of_components,
                                                             'mean':[[randint(-30,30)] for _ in range(number_of_components)],
                                                             'var':[[[randint(1,20)]] for _ in range(number_of_components)]
                                                            }) 
                                                    for vertex in vertices])]
    for node in initial[0].nodes:
        parents_names = structure_parents[node.content['name']]
        for name_p in parents_names:
            for node_p in initial[0].nodes:
                if node_p.content['name'] == name_p:
                    node.nodes_from.append(node_p)
                    break
  
    objective = Objective({'custom': optimisation_metric})
    objective_eval = ObjectiveEvaluate(objective, target_assortativity=target_assort)    

    

    requirements = GraphRequirements(
        max_arity=100,
        max_depth=100, 
        early_stopping_iterations=5,
        num_of_generations=n_generation,
        timeout=timedelta(minutes=time_m),
        history_dir = None
        )

    optimiser_parameters = GPAlgorithmParameters(
        max_pop_size=pop_size,
        pop_size=pop_size,
        crossover_prob=0.8, 
        mutation_prob=0.9,
        genetic_scheme_type = GeneticSchemeTypesEnum.steady_state,
        selection_types = [SelectionTypesEnum.tournament],
        mutation_types = [
        custom_mutation_change_mean, custom_mutation_change_var],
        crossover_types = [custom_crossover_exchange_mean, custom_crossover_exchange_var]
    )
    rules = []
    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=GeneratorModel, base_node_class=GeneratorNode),
        rules_for_constraint=rules,
        # node_factory=DefaultOptNodeFactory(available_node_types=nodes_types)
        )

    optimiser = EvoGraphOptimizer(
        graph_generation_params=graph_generation_params,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements,
        initial_graphs=initial,
        objective=objective)



    # запуск оптимизатора
    optimized_graph = optimiser.optimise(objective_eval)[0]
    print('=============')
    print(model_assortativity(optimized_graph))


if __name__ == '__main__':

    n_generation=200
    time_m=40
    pop_size = 10
    run_example()