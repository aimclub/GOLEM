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
from generator_model import GeneratorModel, GeneratorNode, change_mean, change_var, custom_crossover_exchange_mean, custom_crossover_exchange_var, optimisation_metric_topology, model_topology
from functools import partial
import time
from scipy.spatial import distance
from golem.visualisation.opt_history.multiple_fitness_line import MultipleFitnessLines
from golem.visualisation.opt_viz_extra import visualise_pareto
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.operator import PopulationT

from examples.adaptive_optimizer.utils import plot_action_values

from matplotlib import pyplot as plt









def run_example():

    

    number_of_atr = [2]
    number_of_times = [5]
    df_result = pd.DataFrame(columns=['Number of nodes', 'Deviation'])
    target_topology = []
    for _ in range(5):
        for n in number_of_atr:
            for t in number_of_times:
                one_time_structure = [('X0', 'X1')]
                different_time_structure = [('X0', 'X0'), ('X1', 'X1')]
                structure = []
                vertices = []
                for j in range(t): 
                    for i in range(n):
                        vertices.append('X'+str(i)+'_'+'t'*j)
                for t1 in range(0, t-1, 1):
                    for edge1 in one_time_structure:
                        edge_new = (edge1[0]+'_'+'t'*t1, edge1[1]+'_'+'t'*t1)
                        structure.append(edge_new)
                    for edge2 in different_time_structure:
                        edge_new = (edge2[0]+'_'+'t'*t1, edge2[0]+'_'+'t'*(t1+1))
                        structure.append(edge_new)
                    
                        
                initial = [GeneratorModel(nodes=[GeneratorNode(nodes_from=[],
                                                                        content={'name': vertex,
                                                                                'mean':randint(0,10),
                                                                                'var':randint(1,50),
                                                                                }) 
                                                                        for vertex in vertices])]
                DAG = nx.DiGraph(structure)
                structure_parents = {}
                for v in DAG:
                    structure_parents[v] = list(DAG.pred[v].keys())
                
                for node in initial[0].nodes:
                    parents_names = structure_parents[node.content['name']]
                    for name_p in parents_names:
                        for node_p in initial[0].nodes:
                            if node_p.content['name'] == name_p:
                                node.nodes_from.append(node_p)
                

                objective = Objective(quality_metrics={'topology':optimisation_metric_topology})
                
                objective_eval = ObjectiveEvaluate(objective, target_topology = target_topology)    
        
                requirements = GraphRequirements(
                    max_arity=100,
                    max_depth=100, 
                    early_stopping_iterations=10,
                    num_of_generations=n_generation,
                    timeout=timedelta(minutes=time_m),
                    history_dir = None
                    )

                optimiser_parameters = GPAlgorithmParameters(
                    multi_objective=False,
                    max_pop_size=60,
                    pop_size=pop_size,
                    crossover_prob=0.8, 
                    mutation_prob=0.9,
                    selection_types = [SelectionTypesEnum.tournament],
                    mutation_types = [change_mean, change_var],
                    crossover_types = [CrossoverTypesEnum.none]#[custom_crossover_exchange_mean, custom_crossover_exchange_var]
                )
                rules = []
                graph_generation_params = GraphGenerationParams(
                    adapter=DirectAdapter(base_graph_class=GeneratorModel, base_node_class=GeneratorNode),
                    rules_for_constraint=rules,
                    )

                optimiser = EvoGraphOptimizer(
                    graph_generation_params=graph_generation_params,
                    graph_optimizer_params=optimiser_parameters,
                    requirements=requirements,
                    initial_graphs=initial,
                    objective=objective)
                

                



                start = time.time()
                optimized_graph = optimiser.optimise(objective_eval)
                history = optimiser.history
                history.save('examples/topology_generator_learning/results'+str(_)+' '+str(n)+' exp1_history.json')
                end = time.time()
                for g_i, g in enumerate(optimized_graph):
                    df_dict = pd.DataFrame({'Number of atr':[n], 'Deviation':[optimisation_metric_topology(g, target_topology)]})
                    df_result = pd.concat([df_result, df_dict], ignore_index=True)
                    df_result.to_csv('examples/topology_generator_learning/results'+str(_)+' '+str(n)+' exp1.csv', index=False)
                    

            


if __name__ == '__main__':


    n_generation=500
    time_m=100
    pop_size = 20
    run_example()