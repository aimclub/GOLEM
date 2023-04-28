from copy import deepcopy
from datetime import timedelta
import sys
import os

parentdir = os.getcwd()
sys.path.insert(0, parentdir)
import pandas as pd
from sklearn import preprocessing
import bamt.preprocessors as pp
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.dag.convert import graph_structure_as_nx_graph
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.objective.objective_eval import ObjectiveEvaluate
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import PopulationalOptimizer
from pgmpy.estimators import K2Score, BicScore
from bn_model import  BNModel
from bn_node import BNNode
from pgmpy.models import BayesianNetwork
import seaborn as sns
import matplotlib.pyplot as plt
import bamt.networks as Nets
from golem.core.dag.linked_graph import LinkedGraph
from golem.core.dag.graph_utils import ordered_subnodes_hierarchy
from numpy import std, mean, log
from random import randint
from math import log10
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import itertools
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork as BayesianNetwork_pgmpy
from pgmpy.metrics import structure_score, log_likelihood_score
import numpy as np

from sklearn.linear_model import (LinearRegression as SklearnLinReg, LogisticRegression as SklearnLogReg)
from bn_genetic_operators import (
    custom_crossover_exchange_edges, 
    custom_crossover_exchange_parents_both, 
    custom_crossover_exchange_parents_one,
    custom_mutation_add_structure, 
    custom_mutation_delete_structure, 
    custom_mutation_reverse_structure)


# задаем метрику
# def custom_metric(graph: BNModel, data: pd.DataFrame):
#     score = 0
#     graph_nx, labels = graph_structure_as_nx_graph(graph)
#     struct = []
#     for pair in graph_nx.edges():
#         l1 = str(labels[pair[0]])
#         l2 = str(labels[pair[1]])
#         struct.append([l1, l2])
    
#     bn_model = BayesianNetwork(struct)
#     bn_model.add_nodes_from(data.columns)    
    
#     score = K2Score(data).score(bn_model)

#     return -score

def custom_metric(graph: BNModel, data: pd.DataFrame):
    data_all = data
    data_train , data_test = train_test_split(data_all, train_size = 0.8, random_state=42, shuffle = False)
    score, len_data = 0, len(data_train)
    for node in graph.nodes:   
        data_of_node_train = data_train[node.content['name']]
        data_of_node_test = data_test[node.content['name']]
        if node.nodes_from == None or node.nodes_from == []:
            if node.content['type'] == 'cont':
                mu, sigma = mean(data_of_node_train), std(data_of_node_train)
                score += norm.logpdf(data_of_node_test.values, loc=mu, scale=sigma).sum()
            else:
                count = data_of_node_train.value_counts()
                frequency  = log(count / len_data)
                index = frequency.index.tolist()
                for value in data_of_node_test:
                    if value in index:
                        score += frequency[value]

        else:
            columns, target, idx =  [n.content['name'] for n in node.nodes_from], data_of_node_train.to_numpy(), data_train.index.to_numpy()
            if node.content['type'] == 'cont':
                model = SklearnLinReg()
            else:
                model = SklearnLogReg()
            features = data_train[columns].to_numpy()
            fitted_model = model.fit(features, target)
            
            idx=data_test.index.to_numpy()
            features=data_test[columns].to_numpy()
            target=data_of_node_test.to_numpy()            
            if node.content['type'] == 'cont':
                predict = fitted_model.predict(features)        
                mse =  mean_squared_error(target, predict, squared=False) + 0.0000001
                a = norm.logpdf(target, loc=predict, scale=mse)
                score += a.sum()                
            else:
                predict_proba = fitted_model.predict_proba(features)
                idx = pd.array(list(range(len(target))))
                li = []
                
                for i in idx:
                    a = predict_proba[i]
                    try:
                        b = a[target[i]]
                    except:
                        b = 0.0000001
                    li.append(log(b))
                score += sum(li)
    edges_count = len(graph.get_edges())
    score -= (edges_count*percent)*log10(len_data)*edges_count
    # score -= len(graph.nodes)*log10(len_data)*edges_count/4

    return -score

# def custom_metric(graph: BNModel, data: pd.DataFrame):
#     score = 0
#     nodes = data.columns.to_list()
#     graph_nx, labels = graph_structure_as_nx_graph(graph)
#     data_values=data.values
#     struct = []
#     for pair in graph_nx.edges():
#         l1 = str(labels[pair[0]])
#         l2 = str(labels[pair[1]])
#         struct.append([l1, l2])

#     new_struct=[ [] for _ in range(len(vertices))]
#     for pair in struct:
#         i=dir_of_vertices[pair[1]]
#         j=dir_of_vertices[pair[0]]
#         new_struct[i].append(j)
    
#     new_struct=tuple(map(lambda x: tuple(x), new_struct))   
    
#     bn_model = BayesianNetwork_pgmpy(struct)
#     bn_model.add_nodes_from(data.columns)
#     bn_model.fit(data, estimator=MaximumLikelihoodEstimator)
#     LL = log_likelihood_score(bn_model, data)

#     Dim = 0
#     for i in nodes:
#         unique = (unique_values[i])
#         for j in new_struct[dir_of_vertices[i]]:
#             unique = unique * unique_values[dir_of_vertices_rev[j]]
#         Dim += unique
#     score = LL - (percent*Dim)*log10(len(data))*Dim    

#     return -score

def connect_nodes(self, parent: BNNode, child: BNNode):
    if child.descriptive_id not in [p.descriptive_id for p in ordered_subnodes_hierarchy(parent)]:
        try:
            if child.nodes_from==None or child.nodes_from==[]:
                child.nodes_from = []
                child.nodes_from.append(parent)               
            else:                      
                child.nodes_from.append(parent)
        except Exception as ex:
            print(ex)

def disconnect_nodes(self, node_parent: BNNode, node_child: BNNode,
                    clean_up_leftovers: bool = True):
    if not node_child.nodes_from or node_parent not in node_child.nodes_from:
        return
    elif node_parent not in self._nodes or node_child not in self._nodes:
        return
    elif len(node_child.nodes_from) == 1:
        node_child.nodes_from = None
    else:
        node_child.nodes_from.remove(node_parent)

    if clean_up_leftovers:
        self._clean_up_leftovers(node_parent)

    self._postprocess_nodes(self, self._nodes)


def reverse_edge(self, node_parent: BNNode, node_child: BNNode):
    self.disconnect_nodes(node_parent, node_child, False)
    self.connect_nodes(node_child, node_parent)

LinkedGraph.reverse_edge = reverse_edge
LinkedGraph.connect_nodes = connect_nodes
LinkedGraph.disconnect_nodes = disconnect_nodes


# задаем правила на запрет дублирующих узлов
def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True
    

def run_example():
    
    data = pd.read_csv('C:/Users/anaxa/Documents/Projects/CompositeBayesianNetworks/FEDOT/examples/data/'+file+'.csv') 
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data.reset_index(inplace=True, drop=True)

    global vertices
    vertices = list(data.columns)
    # global dir_of_vertices
    # dir_of_vertices={vertices[i]:i for i in range(len(vertices))}    
    # global dir_of_vertices_rev
    # dir_of_vertices_rev={i:vertices[i] for i in range(len(vertices))}  

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    # p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    p = pp.Preprocessor([('encoder', encoder)])
    discretized_data, est = p.apply(data)
    global dir_of_vertices
    dir_of_vertices={vertices[i]:i for i in range(len(vertices))}    
    global dir_of_vertices_rev
    dir_of_vertices_rev={i:vertices[i] for i in range(len(vertices))}        
    global unique_values
    unique_values = {vertices[i]:len(pd.unique(discretized_data[vertices[i]])) for i in range(len(vertices))}
    # global node_type
    # node_type = p.info['types'] 
    # global types
    # types=list(node_type.values())



    # правила для байесовских сетей: нет петель, нет циклов, нет повторяющихся узлов
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates
     ]
    

    # задаем для оптимизатора fitness-функцию
    objective = Objective({'custom': custom_metric})
    objective_eval = ObjectiveEvaluate(objective, data = discretized_data)    
    

    initial = [BNModel(nodes=[BNNode(nodes_from=[],
                                                    content={'name': v, 
                                                            'type': p.nodes_types[v]}) for v in vertices])]
    
    # init = deepcopy(initial[0])
    
    # def create_population(pop_size, initial = []):
    # # генерация рандомных индивидов, соответствующих правилам
    #     for i in range(0, pop_size):
    #         rand = randint(1, 2*len(vertices))
    #         g=deepcopy(init)
    #         for _ in range(rand):
    #             g=deepcopy(custom_mutation_add_structure(g))
    #         initial.append(g)    

    #     return initial

    # initial = []
    # initial = create_population(pop_size, initial) 
    
    start_time = time.perf_counter()
    l_n = 0
    last = 0
    elapsed_time = 0
    it = 0
    nich_list = []
    nich_result = []
    
    while l_n <= sequential_count and elapsed_time < time_m and it < max_numb_nich:

        requirements = GraphRequirements(
            max_arity=100,
            max_depth=100, 
            num_of_generations=n_generation,
            timeout=timedelta(minutes=time_m),
            history_dir = None,
            n_jobs = -1
            )

        optimiser_parameters = GPAlgorithmParameters(
            pop_size=pop_size,
            crossover_prob=crossover_probability, 
            mutation_prob=mutation_probability,
            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state,
            selection_types = [SelectionTypesEnum.tournament],
            mutation_types = [
            custom_mutation_add_structure, 
            custom_mutation_delete_structure, 
            custom_mutation_reverse_structure
            ],
            
            crossover_types = crossover

        )
        optimiser_parameters.niching = nich_list

        graph_generation_params = GraphGenerationParams(
            adapter=DirectAdapter(base_graph_class=BNModel, base_node_class=BNNode),
            rules_for_constraint=rules)

        optimiser = EvoGraphOptimizer(
            graph_generation_params=graph_generation_params,
            graph_optimizer_params=optimiser_parameters,
            requirements=requirements,
            initial_graphs=initial,
            objective=objective)
        
        it+=1
        # запуск оптимизатора
        optimized_graph = optimiser.optimise(objective_eval)[0]
        score = round(custom_metric(optimized_graph, data=discretized_data), 2)
        nich_list = optimiser_parameters.niching + [score]
        print(nich_list)
        initial=[optimized_graph]

        structure = [(str(edge[0]), str(edge[1])) for edge in optimized_graph.get_edges()]
        nich_result = nich_result + [[optimized_graph, structure, score]]  

        del requirements
        del optimiser_parameters
        del graph_generation_params
        del optimiser


        if last == score:
            l_n += 1
        else:
            last = score
            l_n = 0     
        print(l_n)
        last = score

      
    
    elapsed_time =(time.perf_counter() - start_time)/60
    print(nich_result)
    index_min = np.argmin(nich_list)
    optimized_graph = nich_result[index_min][0]
    structure = nich_result[index_min][1]
    score=nich_result[index_min][2]

    print(optimized_graph.operator.get_edges())

    with open('examples/data/'+file+'.txt') as f:
        lines = f.readlines()
    true_net = []
    for l in lines:
        e0 = l.split()[0]
        e1 = l.split()[1].split('\n')[0]
        true_net.append((e0, e1))
    
    def child_dict(net: list):
        res_dict = dict()
        for e0, e1 in net:
            if e1 in res_dict:
                res_dict[e1].append(e0)
            else:
                res_dict[e1] = [e0]
        return res_dict

    def precision_recall(pred_net: list, true_net: list, decimal = 2):
        pred_dict = child_dict(pred_net)
        true_dict = child_dict(true_net)
        corr_undir = 0
        corr_dir = 0
        for e0, e1 in pred_net:
            flag = True
            if e1 in true_dict:
                if e0 in true_dict[e1]:
                    corr_undir += 1
                    corr_dir += 1
                    flag = False
            if (e0 in true_dict) and flag:
                if e1 in true_dict[e0]:
                    corr_undir += 1
        pred_len = len(pred_net)
        true_len = len(true_net)
        shd = pred_len + true_len - corr_undir - corr_dir
        return {
        'AP': round(corr_undir/pred_len, decimal), 
        'AR': round(corr_undir/true_len, decimal), 
        'AHP': round(corr_dir/pred_len, decimal), 
        'AHR': round(corr_dir/true_len, decimal), 
        'SHD': shd}
    
    def func(strt, vector, full_set_edges):
        for i in range(len(full_set_edges)):
            if full_set_edges[i] in strt:
                vector[i] = 1
        return vector

    def F1(true, ga):
        flatten_edges = list(itertools.chain(*true))
        nodes = list(set(flatten_edges))
        full_set_edges = list(itertools.permutations(nodes,2))
        len_edges = len(full_set_edges)
        true_vector = [0]*len_edges
        ga_vector = [0]*len_edges  
        func(true, true_vector, full_set_edges)
        func(ga, ga_vector, full_set_edges)
        return f1_score(true_vector, ga_vector)

    optimized_structure = [(str(edge[0]), str(edge[1])) for edge in optimized_graph.get_edges()]
    score = custom_metric(optimized_graph, discretized_data)
    SHD = precision_recall(optimized_structure, true_net)['SHD']
    f1 = F1(true_net, optimized_structure)
    spent_time = elapsed_time
    # print([i.__name__ for i in optimiser.crossover.parameters.crossover_types])
    # textfile = open(file+"_p_"+str(percent)+"_LL_"+"sequential_"+str(sequential)+"_nich_"+str(nich)+"_"+str(crossover_fun[0].__name__)+".txt", "a")
    textfile = open(file + '_niching' + ".txt", "a")    
    textfile.write('Structure = ' + str(optimized_structure)+'\n')
    textfile.write('Score = ' + str(score)+'\n')
    textfile.write('SHD = ' + str(SHD)+'\n')
    textfile.write('F1 = ' + str(f1)+'\n')
    textfile.write('Spent_time = ' + str(spent_time)+'\n')
    textfile.close()     


if __name__ == '__main__':
    percent = 0.02
    # файл с исходными данными (должен лежать в 'examples/data/')
    # file = 'sachs'
    # размер популяции 
    pop_size = 20
    # количество поколений
    n_generation = 1000
    # вероятность кроссовера
    crossover_probability = 0.8
    # вероятность мутации
    mutation_probability = 0.9
    time_m = 10
    sequential_count = 3
    max_numb_nich = 100
    crossovers = [
            [custom_crossover_exchange_edges,
            custom_crossover_exchange_parents_both,
            custom_crossover_exchange_parents_one]
            ]
    # n - число запусков
    # files = ['asia', 'healthcare', 'cancer', 'earthquake', 'sachs']
    files = ['sachs']
    for file in files:
        for crossover in crossovers:
            n = 1
            number = 1
            while number <= n:
                run_example() 
                number += 1 
    # n = 5
    # number = 1
    # while number <= n:
    #     run_example() 
    #     number += 1 






