
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
from generator_model import GeneratorModel, GeneratorNode, model_assortativity, optimisation_metric, custom_mutation_change_mean_i, custom_mutation_change_var_i, custom_mutation_change_mean, custom_mutation_change_var, custom_crossover_exchange_mean, custom_crossover_exchange_var, custom_crossover_exchange_mean_i, custom_crossover_exchange_var_i, save_in_bn

import time














def run_example():

    number_of_components = 3
    number_of_atr = [10, 50]
    p_edge = [0.05, 0.2]
    target_assort = [0.3, 0.9]
    df_result = pd.DataFrame(columns=['Number of atr', 'P_edge', 'Target assort', 'Time', 'Deviation', 'Cluster', 'Path length', 'Node degree'])
    synth_graphs = ['G_50_0.4_2_10', 'G_50_0.5_2.5_5', 'G_50_0.1_2.5_5', 'G_50_0.3_1.5_20']
    for file in synth_graphs:
        file1 = open('examples/nets/'+file+'.txt', 'r')
        Lines = file1.readlines()
        df = pd.DataFrame(columns=['v1', 'v2'])
        for i, l in enumerate(Lines):
            l = l.rstrip()
            df.loc[i,'v1'] = int(l.split(',')[0])
            df.loc[i,'v2'] = int(l.split(',')[1])
        synthetic_graph = df.values
        cluster = file.split('_')[2]
        path_length = file.split('_')[3]
        node_degree = file.split('_')[4]

    
        for _ in range(5):
            for n in number_of_atr:
                vertices = []
                for i in range(n):
                    vertices.append('A'+str(i))
                for p_i in p_edge:
                    initial = [GeneratorModel(nodes=[GeneratorNode(nodes_from=[],
                                                                            content={'name': vertex,
                                                                                    'w':[1/number_of_components]*number_of_components,
                                                                                    'mean':[[randint(0,10)] for _ in range(number_of_components)],
                                                                                    'var':[[[randint(1,50)]] for _ in range(number_of_components)]
                                                                                    }) 
                                                                            for vertex in vertices])]
                    is_all = True
                    DAG = []
                    while is_all == True:
                        G=nx.gnp_random_graph(n,p_i,directed=True)
                        DAG = nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])
                        if len(DAG.nodes) == n:
                            is_all =False
                    structure_parents = {}
                    for v in DAG:
                        structure_parents['A'+str(v)] = ['A'+str(i) for i in DAG.pred[v].keys()]

                    
                    for node in initial[0].nodes:
                        parents_names = structure_parents[node.content['name']]
                        for name_p in parents_names:
                            for node_p in initial[0].nodes:
                                if node_p.content['name'] == name_p:
                                    node.nodes_from.append(node_p)
                                    break
                    for target in target_assort:
                        objective = Objective({'custom': optimisation_metric})
                        objective_eval = ObjectiveEvaluate(objective, target_assortativity=target, synthetic_graph=synthetic_graph)    

                        

                        requirements = GraphRequirements(
                            max_arity=100,
                            max_depth=100, 
                            early_stopping_iterations=10,
                            num_of_generations=n_generation,
                            timeout=timedelta(minutes=time_m),
                            history_dir = None
                            )

                        optimiser_parameters = GPAlgorithmParameters(
                            max_pop_size=55,
                            pop_size=pop_size,
                            crossover_prob=0.8, 
                            mutation_prob=0.9,
                            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state,
                            selection_types = [SelectionTypesEnum.tournament],
                            mutation_types = [custom_mutation_change_mean_i, custom_mutation_change_var_i,
                            custom_mutation_change_mean, custom_mutation_change_var],
                            crossover_types = [custom_crossover_exchange_mean, custom_crossover_exchange_var, custom_crossover_exchange_mean_i, custom_crossover_exchange_var_i]
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


                        start = time.time()
                        # запуск оптимизатора
                        optimized_graph = optimiser.optimise(objective_eval)[0]
                        end = time.time()
                        calculated = model_assortativity(optimized_graph, synthetic_graph)
                        df_dict = pd.DataFrame({'Number of atr':n, 'P_edge':p_i, 'Target assort':target, 'Time':end-start, 'Deviation':abs(target - calculated), 'Cluster':cluster, 'Path length':path_length, 'Node degree':node_degree}, index=[0])
                        df_result = pd.concat([df_result, df_dict], ignore_index=True)
                        df_result.to_csv('examples/BN_generator_learning/results/paper/'+str(_)+' '+str(n)+' '+str(p_i)+' '+str(target)+' '+file+' exp3.csv', index=False)
                        save_in_bn(optimized_graph, 'examples/BN_generator_learning/results/paper/'+str(_)+' '+str(n)+' '+str(p_i)+' '+str(target)+' '+file+' exp3.json')

    


if __name__ == '__main__':


    n_generation=500
    time_m=60
    pop_size = 10
    run_example()