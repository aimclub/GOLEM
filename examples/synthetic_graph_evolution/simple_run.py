from one_graph_search import run_graph_search
import pandas as pd
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.operators.elitism import ElitismTypesEnum
from golem.core.optimisers.genetic.operators.regularization import RegularizationTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
import numpy as np
import os
from golem.visualisation.opt_history.multiple_fitness_line import MultipleFitnessLines

if __name__ == '__main__':

    num_edges = 5

    l=0
    des_num_nodes = max_graph_size = 30
    for des_degree in [3,5,10,15]:
        for des_cluster in [0.1,0.15,0.2,0.25,0.3,0.35]:
            for adaptive_scheme in ['default', 'bandit', 'context']:
                if adaptive_scheme == 'default':
                    adaptive_mutation_type = MutationAgentTypeEnum.default
                elif adaptive_scheme == 'bandit':
                    adaptive_mutation_type = MutationAgentTypeEnum.bandit
                elif adaptive_scheme == 'context':
                    adaptive_mutation_type = MutationAgentTypeEnum.contextual_bandit
                for specific in [False,True]:
                    if specific:
                        dense = cycle = path = star = True
                    else:
                        dense = cycle = path = star = False

                    for genetic_scheme_type_str in ['generational','steady_state', 'parameter_free']:
                        if genetic_scheme_type_str == 'generational':
                            genetic_scheme_type = GeneticSchemeTypesEnum.generational
                        elif genetic_scheme_type_str == 'steady_state':
                            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state
                        else:
                            genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free
                        #name = str(specific) + '_' + 'none' + '_' + elitism_type_str + '_' + regularization_type_str + '_' + genetic_scheme_type_str + '.npy'
                        #if not os.path.exists(name):
                        df = pd.read_csv('CEC_2.csv')
                        df = df.drop(columns='Unnamed: 0')
                        #df = pd.DataFrame(columns=['adaptive_mutation_type', 'context_type', 'added_patterns_mutations','selection_type','elitism_type','regularization_type','genetic_scheme_type','num_nodes', 'desired_degree','desired_cl','actual_degree','actual_cl'])

                        if len(df[(df['num_nodes'] == des_num_nodes) & (df['desired_degree'] == des_degree) & (
                                df['desired_cl'] == des_cluster) & (df['adaptive_mutation_type'] == adaptive_scheme) & (
                                          df['added_patterns_mutations'] == specific) & (
                                          df['genetic_scheme_type'] == genetic_scheme_type_str)]) == 0:
                            l+=1
                            try:
                                time, act_cl, act_ad, history = run_graph_search(adaptive_mutation_type, SelectionTypesEnum.tournament, ElitismTypesEnum.none, RegularizationTypesEnum.none, genetic_scheme_type, dense=dense,cycle=cycle,path=path,star=star,  num_edges=num_edges, des_degree=des_degree,
                                                                                                  des_cluster=des_cluster, des_num_nodes=des_num_nodes)
                                name = str(des_degree) +'_' + str(des_cluster) +'_' + str(des_num_nodes) + '_'+ str(
                                    adaptive_scheme) +'_' + str(genetic_scheme_type_str) + str(specific)
                                fitn = MultipleFitnessLines.from_histories({'0': [history]})
                                fitn.visualize(metric_id=0, dpi=1000,
                                               save_path='graphics\\' + str(name) + '_1' + '.png')
                                fitn.visualize(metric_id=1, dpi=1000,
                                               save_path='graphics\\' + str(name) + '_2' + '.png')
                                if adaptive_scheme=='context':
                                    df.loc[len(df)] = [adaptive_scheme, 'nodes_num', specific, 'tournament', 'none', 'none', genetic_scheme_type_str, des_num_nodes, des_degree, des_cluster, act_ad , act_cl]
                                else:
                                    df.loc[len(df)] = [adaptive_scheme, None, specific, 'tournament', 'none',
                                                       'none', genetic_scheme_type_str, des_num_nodes, des_degree,
                                                       des_cluster, act_ad, act_cl]

                                df.to_csv('CEC_2.csv')
                            except:
                                pass

    print(l)
    k=0
    des_num_nodes = max_graph_size = 40
    for des_degree in [3,5,10,15,20,25]:
        for des_cluster in [0.1,0.15,0.2,0.25,0.3,0.35]:
            for adaptive_scheme in ['default', 'bandit', 'context']:
                if adaptive_scheme == 'default':
                    adaptive_mutation_type = MutationAgentTypeEnum.default
                elif adaptive_scheme == 'bandit':
                    adaptive_mutation_type = MutationAgentTypeEnum.bandit
                elif adaptive_scheme == 'context':
                    adaptive_mutation_type = MutationAgentTypeEnum.contextual_bandit
                for specific in [False,True]:
                    if specific:
                        dense = cycle = path = star = True
                    else:
                        dense = cycle = path = star = False

                    for genetic_scheme_type_str in ['generational','steady_state', 'parameter_free']:
                        if genetic_scheme_type_str == 'generational':
                            genetic_scheme_type = GeneticSchemeTypesEnum.generational
                        elif genetic_scheme_type_str == 'steady_state':
                            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state
                        else:
                            genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free
                        #name = str(specific) + '_' + 'none' + '_' + elitism_type_str + '_' + regularization_type_str + '_' + genetic_scheme_type_str + '.npy'
                        #if not os.path.exists(name):
                        df = pd.read_csv('CEC_2.csv')
                        df = df.drop(columns='Unnamed: 0')
                        #df = pd.DataFrame(columns=['adaptive_mutation_type', 'context_type', 'added_patterns_mutations','selection_type','elitism_type','regularization_type','genetic_scheme_type','num_nodes', 'desired_degree','desired_cl','actual_degree','actual_cl'])

                        if len(df[(df['num_nodes'] == des_num_nodes) & (df['desired_degree'] == des_degree) & (
                                df['desired_cl'] == des_cluster) & (df['adaptive_mutation_type'] == adaptive_scheme) & (
                                          df['added_patterns_mutations'] == specific) & (
                                          df['genetic_scheme_type'] == genetic_scheme_type_str)]) == 0:

                            try:
                                time, act_cl, act_ad = run_graph_search(adaptive_mutation_type, SelectionTypesEnum.tournament, ElitismTypesEnum.none, RegularizationTypesEnum.none, genetic_scheme_type, dense=dense,cycle=cycle,path=path,star=star,  num_edges=num_edges, des_degree=des_degree,
                                                                                                 des_cluster=des_cluster, des_num_nodes=des_num_nodes)
                                if adaptive_scheme=='context':
                                    df.loc[len(df)] = [adaptive_scheme, 'nodes_num', specific, 'tournament', 'none', 'none', genetic_scheme_type_str, des_num_nodes, des_degree, des_cluster, act_ad , act_cl]
                                else:
                                    df.loc[len(df)] = [adaptive_scheme, None, specific, 'tournament', 'none',
                                                       'none', genetic_scheme_type_str, des_num_nodes, des_degree,
                                                       des_cluster, act_ad, act_cl]

                                df.to_csv('CEC_2.csv')
                            except:
                                pass
    print(k)
