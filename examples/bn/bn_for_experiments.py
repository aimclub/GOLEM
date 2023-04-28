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
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.objective.objective_eval import ObjectiveEvaluate
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import PopulationalOptimizer
from pgmpy.estimators import K2Score, BicScore
from bn_model import  BNModel
from bn_node import BNNode
from pomegranate import BayesianNetwork
import seaborn as sns
import matplotlib.pyplot as plt
from bamt.networks.hybrid_bn import HybridBN
from bamt.networks.discrete_bn import DiscreteBN
from bamt.networks.continuous_bn import ContinuousBN
from golem.core.dag.linked_graph import LinkedGraph
from golem.core.dag.graph_utils import ordered_subnodes_hierarchy
from numpy import std, mean, log
from math import log10
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import norm
import itertools
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork as BayesianNetwork_pgmpy
from pgmpy.metrics import structure_score, log_likelihood_score

from sklearn.linear_model import (LinearRegression as SklearnLinReg, LogisticRegression as SklearnLogReg)
from bn_genetic_operators import (
    custom_mutation_add_structure, 
    custom_mutation_delete_structure, 
    custom_mutation_reverse_structure,
    
    custom_mutation_change_node)

from xgboost import XGBRegressor

# задаем метрику
# def custom_metric(graph: BNModel, data: pd.DataFrame):
#     score = 0
#     graph_nx, labels = graph_structure_as_nx_graph(graph)
#     struct = []
#     for pair in graph_nx.edges():
#         l1 = str(labels[pair[0]])
#         l2 = str(labels[pair[1]])
#         struct.append([l1, l2])
    
#     bn_model = BayesianNetwork_pgmpy(struct)
#     bn_model.add_nodes_from(data.columns)    
#     score = K2Score(data).score(bn_model)

#     return -score


# def custom_metric(graph: BNModel, data: pd.DataFrame):
#     score = 0
#     edges_count = len(graph.get_edges())
#     graph_nx, labels = graph_structure_as_nx_graph(graph)
#     data_train , data_test = train_test_split(data, train_size = 0.9, random_state=42, shuffle = False)
#     struct = []
#     for pair in graph_nx.edges():
#         l1 = str(labels[pair[0]])
#         l2 = str(labels[pair[1]])
#         struct.append([l1, l2])
    
#     bn_model = BayesianNetwork_pgmpy(struct)
#     bn_model.add_nodes_from(data.columns)    
#     bn_model.fit(
#     data=data_train,
#     estimator=MaximumLikelihoodEstimator
# )
#     score = log_likelihood_score(bn_model, data_test)
#     # score = K2Score(data).score(bn_model)
#     # score -= (edges_count + 2*len(graph.nodes))*log10(len(data))/2
#     score -= (edges_count*percent)*log10(len(data))*edges_count

#     return -score

# fitness-SHD
# def custom_metric(graph: BNModel, data: pd.DataFrame):
#     if graph.get_edges():
#         structure = [(str(edge[0]), str(edge[1])) for edge in graph.get_edges()]
#         shd = precision_recall(structure, true_net)['SHD']
#     else:
#         shd = len(true_net)
#     return shd


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
                # model = SklearnLinReg()
                model = XGBRegressor()
            else:
                model = SklearnLogReg()
            features = data_train[columns].to_numpy()
            
            if len(set(target)) == 1:
                # the model will predict with probability 1 (log(1) = 0) => continue
                continue

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
                    probas_i = predict_proba[i]
                    try:
                        proba = probas_i[target[i]]
                    except:
                        proba = 0.0000001
                    li.append(log(proba))
                score += sum(li)
    edges_count = len(graph.get_edges())
    # score -= (edges_count + 2*len(graph.nodes))*log10(len_data)/2

    score -= (edges_count*percent)*log10(len_data)*edges_count
    # score -= len(graph.nodes)*log10(len_data)*edges_count/2
    # score -= len(graph.nodes)*log10(len_data)*edges_count/4

    return -score

# def custom_metric(graph: BNModel, data: pd.DataFrame):
    # score = 0
    # nodes = data.columns.to_list()
    # graph_nx, labels = graph_structure_as_nx_graph(graph)
    # data_values=data.values
    # struct = []
    # for pair in graph_nx.edges():
    #     l1 = str(labels[pair[0]])
    #     l2 = str(labels[pair[1]])
    #     struct.append([l1, l2])

    # new_struct=[ [] for _ in range(len(vertices))]
    # for pair in struct:
    #     i=dir_of_vertices[pair[1]]
    #     j=dir_of_vertices[pair[0]]
    #     new_struct[i].append(j)
    
    # new_struct=tuple(map(lambda x: tuple(x), new_struct))   
    # bn_model = BayesianNetwork_pgmpy(struct)
    # bn_model.add_nodes_from(vertices)
    # bn_model.fit(data, estimator=MaximumLikelihoodEstimator)
    # LL = log_likelihood_score(bn_model, data)

    # Dim = 0
    # for i in nodes:
    #     unique = (unique_values[i])
    #     for j in new_struct[dir_of_vertices[i]]:
    #         unique = unique * unique_values[dir_of_vertices_rev[j]]
    #     Dim += unique
    # score = LL - (percent*Dim)*log10(len(data))*Dim    

    # return -score


# задаем правила на запрет дублирующих узлов
def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True

def _no_disc_node_cont_parents(graph):
    # graph, labels = graph_structure_as_nx_graph(graph)
    for parent, child in graph.get_edges():
        if (parent.content['type'] == 'cont') & (child.content['type'] == 'disc') :
            raise ValueError(f'Discrete node has cont parent')
    return True

# для SHD     
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
    

# def precision_recall(pred_net: list, true_net: list, decimal = 4):
#     pred_dict = child_dict(pred_net)
#     true_dict = child_dict(true_net)
#     corr_undir = 0
#     corr_dir = 0
#     for e0, e1 in pred_net:
#         flag = True
#         if e1 in true_dict:
#             if e0 in true_dict[e1]:
#                 corr_undir += 1
#                 corr_dir += 1
#                 flag = False
#         if (e0 in true_dict) and flag:
#             if e1 in true_dict[e0]:
#                 corr_undir += 1
#     pred_len = len(pred_net)
#     true_len = len(true_net)
#     shd = pred_len + true_len - corr_undir - corr_dir
#     return {'AP': round(corr_undir/pred_len, decimal),
#             'AR': round(corr_undir/true_len, decimal),
#             'F1_undir':round(2*(corr_undir/pred_len)*(corr_undir/true_len)/(corr_undir/pred_len+corr_undir/true_len), decimal),
#             'AHP': round(corr_dir/pred_len, decimal),
#             'AHR': round(corr_dir/true_len, decimal),
#            'F1_dir': round(2*(corr_dir/pred_len)*(corr_dir/true_len)/(corr_dir/pred_len+corr_dir/true_len), decimal),
#             'SHD': shd}

def table(structure, data):

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])    
    discretized_data, est = p.apply(data)
    final_bn = []
    if 'cont' in p.info['types'].values() and ('disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values()):
        final_bn = HybridBN(has_logit=False, use_mixture=False)
    elif 'disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values():
        final_bn = DiscreteBN()
    elif 'cont' in p.info['types'].values():
        final_bn = ContinuousBN(use_mixture=False)  

    final_bn.add_nodes(p.info)
    # structure = [(str(edge[0]), str(edge[1])) for edge in network.get_edges()]
    # structure = network.get_edges()
    final_bn.set_structure(edges=structure)
    final_bn.get_info()
    final_bn.fit_parameters(data)

    prediction = dict()

    for c in vertices:
        test = data.drop(columns=[c])
        pred = final_bn.predict(test, 5)
        prediction.update(pred)
    
    result = dict()
    for key, value in prediction.items():
        if p.nodes_types[key]=="disc":
            res=round(accuracy_score(data[key], value), 2)
        elif p.nodes_types[key]=="cont":
            res=round(mean_squared_error(data[key], value, squared=False), 2)
        result[key]=res
    
    return result

def run_example():
    with open('examples/data/txt/'+(file)+'.txt') as f:
        lines = f.readlines()
    global true_net
    true_net = []
    for l in lines:
        e0 = l.split()[0]
        e1 = l.split()[1].split('\n')[0]
        true_net.append((e0, e1))    


    data = pd.read_csv('examples/data/csv/'+file+'.csv') 
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data.reset_index(inplace=True, drop=True)
    # print(data['Theft'])
    # from pgmpy.utils import get_example_model
    # from pgmpy.sampling import BayesianModelSampling
    # model = get_example_model("alarm")
    # data = BayesianModelSampling(model).forward_sample(size=int(1e3))
    # edges = model.edges()
    # true_net = edges    

    

    global vertices
    vertices = list(data.columns)

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    # p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    p = pp.Preprocessor([('encoder', encoder)])
    
    # if file == 'vk_data' or file == 'hack_processed_with_rf':
    #     p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    # else:
    #     # p = pp.Preprocessor([])
    #     p = pp.Preprocessor([('encoder', encoder)])    
    
    discretized_data, est = p.apply(data)
    global dir_of_vertices
    dir_of_vertices={vertices[i]:i for i in range(len(vertices))}    
    global dir_of_vertices_rev
    dir_of_vertices_rev={i:vertices[i] for i in range(len(vertices))}        
    global unique_values
    unique_values = {vertices[i]:len(pd.unique(discretized_data[vertices[i]])) for i in range(len(vertices))}
    # global node_type
    node_type = p.info['types'] 
    # global types
    types=list(node_type.values())

    if 'cont' in types and ('disc' in types or 'disc_num' in types):
        bn = HybridBN(has_logit=False, use_mixture=False)
        # rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates, _has_disc_parents]
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates, _no_disc_node_cont_parents]
    elif 'disc' in types or 'disc_num' in types:
        bn = DiscreteBN()
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    elif 'cont' in types:
        bn = ContinuousBN(use_mixture=False)
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates, _no_disc_node_cont_parents]

    bn.add_nodes(p.info)

    ppp = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)]) 
    ddiscretized_data, est = ppp.apply(data)
    
    bn.add_edges(ddiscretized_data, scoring_function=('K2', K2Score))
    print(bn.edges)
    initial_bamt = [BNModel(nodes=[BNNode(nodes_from=[],
                                                      content={'name': v,
                                                               'type': p.nodes_types[v]}) for v in vertices])]

    init = deepcopy(initial_bamt[0])

    for node in initial_bamt[0].nodes: 
        parents = []
        for n in bn.nodes:
            if str(node) == str(n):
                parents = n.cont_parents + n.disc_parents
                break
        for n2 in initial_bamt[0].nodes:
            if str(n2) in parents:
                node.nodes_from.append(n2)
         


    BAMT_network = deepcopy(initial_bamt[0])
    structure_BAMT = [(str(edge[0]), str(edge[1])) for edge in BAMT_network.get_edges()]
    Score_BAMT = round(custom_metric(BAMT_network, data=discretized_data), 6)

    # custom_mutation_change_node(BAMT_network)
    # правила для байесовских сетей: нет петель, нет циклов, нет повторяющихся узлов
    # rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates
    #  ]
    # optimized_structure = [('yecO', 'cspG'), ('atpD', 'dnaK'), ('yfaD','eutG'), ('yheI', 'folK'), ('yceP', 'ibpB'), ('aceB', 'icdA'), ('lacZ', 'lacA'), ('lacA', 'lacY'), ('dnaK', 'mopB'), ('cspG', 'yaeM'), ('folK', 'ycgX'), ('hupB', 'yfiA'), ('b1191', 'ygcE')]
    # SHD_GA = precision_recall(optimized_structure, true_net)['SHD']
    # SHD_BAMT = precision_recall(structure_BAMT, true_net)['SHD']
    
    # задаем для оптимизатора fitness-функцию
    objective = Objective({'custom': custom_metric})
    objective_eval = ObjectiveEvaluate(objective, data = discretized_data)    

    initial = [BNModel(nodes=[BNNode(nodes_from=[],
                                                      content={'name': v, 
                                                               'type': p.nodes_types[v]}) for v in vertices])]
    
    print(custom_metric(initial[0], data=discretized_data))
    requirements = GraphRequirements(
        max_arity=100,
        max_depth=100, 
        num_of_generations=n_generation,
        timeout=timedelta(minutes=time_m),
        history_dir = None,
        early_stopping_iterations = early_stopping_iterations,
        n_jobs = -1
        # history_dir = 'C://Users//anaxa//Documents//Projects//GOLEM//examples//bn//history'
        )

    optimiser_parameters = GPAlgorithmParameters(
        pop_size=pop_size,
        max_pop_size = pop_size,
        crossover_prob=crossover_probability, 
        mutation_prob=mutation_probability,
        genetic_scheme_type = GeneticSchemeTypesEnum.steady_state,
        selection_types = [SelectionTypesEnum.tournament],
        mutation_types = [
        custom_mutation_add_structure, 
        custom_mutation_delete_structure, 
        custom_mutation_reverse_structure,
        # custom_mutation_change_node
        ],
        
        crossover_types = crossover
        # crossover_types = [
        #     custom_crossover_exchange_edges,
        #     custom_crossover_exchange_parents_both,
        #     custom_crossover_exchange_parents_one
        #     ]
    )

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=BNModel, base_node_class=BNNode),
        rules_for_constraint=rules)

    optimiser = EvoGraphOptimizer(
        graph_generation_params=graph_generation_params,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements,
        initial_graphs=initial,
        objective=objective)
    optimiser.number = number

    init = initial[0]
    def structure_to_opt_graph(vertices, structure):

        encoder = preprocessing.LabelEncoder()
        discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        # p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
        p = pp.Preprocessor([('encoder', encoder)])
        discretized_data, est = p.apply(data)
        fdt = BNModel(nodes=[BNNode(nodes_from=[],
                                                      content={'name': v,
                                                               'type': p.nodes_types[v]}) for v in vertices])

        bn = []
        if 'cont' in p.info['types'].values() and ('disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values()):
            bn = HybridBN(has_logit=False, use_mixture=False)
        elif 'disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values():
            bn = DiscreteBN()
        elif 'cont' in p.info['types'].values():
            bn = ContinuousBN(use_mixture=False)  

        bn.add_nodes(p.info)
        bn.set_structure(edges=structure)
        
        for node in fdt.nodes: 
            parents = []
            for n in bn.nodes:
                if str(node) == str(n):
                    parents = n.cont_parents + n.disc_parents
                    break
            for n2 in fdt.nodes:
                if str(n2) in parents:
                    node.nodes_from.append(n2)        
        
        return fdt    
    
    # Score_true = custom_metric(structure_to_opt_graph(vertices, true_net), discretized_data)
    
    # запуск оптимизатора
    optimized_graph = optimiser.optimise(objective_eval)[0]
    print('GA score =',custom_metric(optimized_graph, data=discretized_data))
    print('BAMT score =', Score_BAMT) 

    print(optimized_graph.operator.get_edges())
    
    # print(optimiser.history.archive_history)

    
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
    score_GA = custom_metric(optimized_graph, discretized_data)
    SHD_GA = precision_recall(optimized_structure, true_net)['SHD']
    SHD_BAMT = precision_recall(structure_BAMT, true_net)['SHD']
    f1_GA = F1(true_net, optimized_structure)
    f1_BAMT = F1(true_net, structure_BAMT)
    # f1_GA_undir = precision_recall(optimized_structure, true_net)['F1_undir']
    # f1_GA_dir = precision_recall(optimized_structure, true_net)['F1_dir']
    spent_time = optimiser.timer.minutes_from_start
    print("score GA: ", score_GA)
    print("score BAMT: ", Score_BAMT)   
    # print("score true: ", Score_true) 
    print("SHD GA: ", SHD_GA)
    print("SHD BAMT: ", SHD_BAMT)
    print("f1 GA: ", f1_GA)
    print("f1 BAMT: ", f1_BAMT)    
    print("время: ", spent_time)

    # print([i.__name__ for i in optimiser.crossover.parameters.crossover_types])
    # textfile = open(file+"_p_"+str(percent)+"_LL_"+"sequential_"+str(sequential)+"_nich_"+str(nich)+"_"+str(crossover_fun[0].__name__)+".txt", "a")

    textfile = open('C:\\Users\\anaxa\\Desktop\\bn_score_shd\\bn_score_only_mutations_' + file + "_"  + str(number) + ".txt", "a")   # + 'start_bamt' + "_" 
    textfile.write('Structure = ' + str(optimized_structure) + '\n')
    textfile.write('Score GA = ' + str(score_GA) + '\n')
    textfile.write('Score BAMT = ' + str(Score_BAMT) + '\n')
    # textfile.write('Score true = ' + str(Score_true) + '\n')
    textfile.write('SHD GA = ' + str(SHD_GA) + '\n')
    textfile.write('SHD BAMT = ' + str(SHD_BAMT) + '\n')
    textfile.write('F1 GA = ' + str(f1_GA) + '\n')
    textfile.write('F1 BAMT = ' + str(f1_BAMT) + '\n')    
    textfile.write('Spent_time = ' + str(spent_time) + '\n')
    textfile.write('Generation number = ' + str(optimiser.current_generation_num) + '\n')
    textfile.write('Population number = ' + str(optimiser.graph_optimizer_params.pop_size) + '\n')
    # textfile.write('F1 undir = ' + str(f1_GA_undir) + '\n')
    # textfile.write('F1 dir = ' + str(f1_GA_dir) + '\n')
    # textfile.write('optimized_structure = ' + str(table(optimized_structure, data)) + '\n')
    # textfile.write('BAMT_structure = ' + str(table(structure_BAMT, data)) + '\n')
    # textfile.write('true_structure = ' + str(table(true_net, data)) + '\n')
    textfile.close()     


    # print('optimized_structure: ', table(optimized_structure, data))
    # print('BAMT_structure: ', table(structure_BAMT, data))
    # print('true_structure: ', table(true_net, data))
    # print('final')
if __name__ == '__main__':
    percent = 0.02
    # файл с исходными данными (должен лежать в 'examples/data/')
    # file = 'sachs'
    # размер популяции 
    pop_size = 40 # 40
    # количество поколений
    n_generation = 10000 #100  
    # вероятность кроссовера
    crossover_probability = 0.8
    # вероятность мутации
    mutation_probability = 0.9
    # stopping_after_n_generation
    early_stopping_iterations = 20
    # timeout
    time_m = 1000
    crossovers = [
        # [CrossoverTypesEnum.exchange_edges],
        #         [CrossoverTypesEnum.exchange_parents_one],
        #         [CrossoverTypesEnum.exchange_parents_both],
                [CrossoverTypesEnum.exchange_edges,
                CrossoverTypesEnum.exchange_parents_one,
                CrossoverTypesEnum.exchange_parents_both]
            ]
    # n - число запусков
    files = ['asia', 'cancer', 'earthquake', 'survey', 'sachs', 'alarm', 'barley', 'child', 'water', 'ecoli70', 'magic-niab', 'healthcare', 'sangiovese', 'mehra-complete'] # 'asia', 'healthcare', 'cancer', 'earthquake', 'sachs', 'alarm', 'child', 'ecoli70','magic-niab', 'mehra-complete', 'sangiovese', 'insurance',
    # files = ['sachs'] # 'barley', 'mildew', 'insurance'
    for file in files:
        for crossover in crossovers:
            n = 1
            number = 1
            while number <= n:
                # try:
                
                run_example() 
                number += 1 
                # except:
                #     continue
    # n = 5
    # number = 1
    # while number <= n:
    #     run_example() 
    #     number += 1 






