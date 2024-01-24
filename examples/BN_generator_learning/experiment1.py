
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
from random import choice, random,randint, sample, uniform
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
from generator_model_two_crit import GeneratorModel, GeneratorNode, optimisation_metric_assort, custom_mutation_change_cov, custom_mutation_change_mean, custom_mutation_change_var, \
    custom_crossover_exchange_mean, custom_crossover_exchange_var, \
    save_in_bn, optimisation_metric_correlaion, attr_correlation, model_assortativity, optimisation_metric_assort
from functools import partial
import time
from scipy.spatial import distance
















def run_example():

    number_of_atr = [5, 8, 10, 20]
    p_edge = [0.05]
    target_assort = [0.2, 0.9]
    corr = ['low', 'high']
    df_result = pd.DataFrame(columns=['Number of atr', 'P_edge', 'Target assort', 'Target correlation', 'Calculated_corr', 'Time', 'Deviation_assort'])
    for _ in range(5):
        for n in number_of_atr:
            vertices = []
            for i in range(n):
                vertices.append('A'+str(i))
            for p_i in p_edge:
                initial = [GeneratorModel(nodes=[GeneratorNode(nodes_from=[],
                                                                        content={'name': vertex,
                                                                                'mean':randint(0,10),
                                                                                'var':randint(1,50),
                                                                                'cov':[]
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
                           
                graph_index = dict()
                for cor in corr:
                    target_correlation = []
                    if cor == 'high':
                        for k, node in enumerate(initial[0].nodes):
                            graph_index[node.content['name']] = k
                            if node.nodes_from:
                                for __ in node.nodes_from:
                                    target_correlation.append(uniform(0.7, 0.9))
                                    node.content['cov'].append(randint(-100,100))
                    else:
                        for k, node in enumerate(initial[0].nodes):
                            graph_index[node.content['name']] = k
                            if node.nodes_from:
                                for __ in node.nodes_from:
                                    target_correlation.append(uniform(0.0, 0.3))
                                    node.content['cov'].append(randint(-100,100))

                    for target in target_assort:

                        objective = Objective(quality_metrics={'assort':optimisation_metric_assort}, complexity_metrics={'corr':optimisation_metric_correlaion}, is_multi_objective=True)
                        
                        objective_eval = ObjectiveEvaluate(objective, target_assortativity=target, target_correlation=target_correlation, graph_index=graph_index)    
                    
                        


                        

                        requirements = GraphRequirements(
                            max_arity=100,
                            max_depth=100, 
                            early_stopping_iterations=10,
                            num_of_generations=n_generation,
                            timeout=timedelta(minutes=time_m),
                            history_dir = None
                            )

                        optimiser_parameters = GPAlgorithmParameters(
                            multi_objective=objective.is_multi_objective,
                            max_pop_size=55,
                            pop_size=pop_size,
                            crossover_prob=0.8, 
                            mutation_prob=0.9,
                            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state,
                            selection_types = [SelectionTypesEnum.tournament],
                            mutation_types = [custom_mutation_change_cov,
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
                        

                        



                        start = time.time()
                        # # запуск оптимизатора
                        optimized_graph = optimiser.optimise(objective_eval)
                        optimiser.history.save('examples/BN_generator_learning/results/paper_cec/'+str(_)+' '+str(n)+' '+str(p_i)+' '+str(target)+' '+cor+' exp2_history.json')
                        optimiser.history.show.visuals_params(save_path='examples/BN_generator_learning/results/paper_cec/'+str(_)+' '+str(n)+' '+str(p_i)+' '+str(target)+' '+cor+' exp2_fig.png')
                        end = time.time()
                        for g_i, g in enumerate(optimized_graph):
                            df_dict = pd.DataFrame({'Number of atr':[n], 'P_edge':[p_i], 'Target assort':[target], 'Target correlation':[str(target_correlation)], 'Calculated_corr':[str(attr_correlation(g,graph_index))], 'Time':[round(end-start)], 'Deviation_assort':[abs(target-model_assortativity(g, graph_index))]})
                            df_result = pd.concat([df_result, df_dict], ignore_index=True)
                            df_result.to_csv('examples/BN_generator_learning/results/paper_cec/'+str(_)+' '+str(n)+' '+str(p_i)+' '+str(target)+' '+cor+' exp2.csv', index=False)
                            save_in_bn(g, 'examples/BN_generator_learning/results/paper_cec/'+str(_)+' '+str(n)+' '+str(p_i)+' '+str(target)+' '+cor+' '+str(g_i)+' exp2.json', graph_index)

    


if __name__ == '__main__':


    n_generation=500
    time_m=100
    pop_size = 10
    run_example()