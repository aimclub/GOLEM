
from audioop import cross
from datetime import timedelta
import sys
# from typing import Optional, Union, List
import os
# import time

parentdir = os.getcwd()

sys.path.insert(0, parentdir)
# from golem.core.dag.graph import Graph
from copy import deepcopy
import pandas as pd
from random import choice, sample
from sklearn import preprocessing
import bamt.preprocessors as pp
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.dag.convert import graph_structure_as_nx_graph
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.objective.objective_eval import ObjectiveEvaluate
from golem.core.optimisers.optimizer import GraphGenerationParams
from pgmpy.estimators import K2Score
from math import ceil
from composite_model import CompositeModel
from composite_node import CompositeNode
from bamt.networks.hybrid_bn import HybridBN
from bamt.networks.discrete_bn import DiscreteBN
from bamt.networks.continuous_bn import ContinuousBN
from scipy.stats import norm
from numpy import std, mean, log
from sklearn.metrics import mean_squared_error
from itertools import chain
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

from golem.core.dag.linked_graph import LinkedGraph

from golem.core.dag.graph_utils import ordered_subnodes_hierarchy

from ML import ML_models
from sklearn.linear_model import (LinearRegression as SklearnLinReg, LogisticRegression as SklearnLogReg)


def composite_metric(graph: CompositeModel, data: pd.DataFrame):
    try:
        data_all = data
        data_train , data_test = train_test_split(data_all, train_size = 0.8, random_state=42)
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
                model, columns, target, idx = node.content['parent_model'](), [n.content['name'] for n in node.nodes_from], data_of_node_train.to_numpy(), data_train.index.to_numpy()
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
    except Exception as ax:
        print(ax)
    return -score


# задаем кроссовер (обмен ребрами)
def custom_crossover_exchange_edges(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):

    def find_node(graph: CompositeModel, node):
        name = node.content['name']
        for graph_node in graph.nodes:
            if graph_node.content['name'] == name:
                return graph_node

    num_cros = 100
    try:
        for _ in range(num_cros):
            old_edges1 = []
            old_edges2 = []
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            edges_1 = new_graph_first.operator.get_edges()
            edges_2 = new_graph_second.operator.get_edges()
            count = ceil(min(len(edges_1), len(edges_2))/2)
            choice_edges_1 = sample(edges_1, count)
            choice_edges_2 = sample(edges_2, count)
            
            for pair in choice_edges_1:
                new_graph_first.operator.disconnect_nodes(pair[0], pair[1], False)
            for pair in choice_edges_2:
                new_graph_second.operator.disconnect_nodes(pair[0], pair[1], False)  
            
            old_edges1 = new_graph_first.operator.get_edges()
            old_edges2 = new_graph_second.operator.get_edges()

            new_edges_2 = [[find_node(new_graph_second, i[0]), find_node(new_graph_second, i[1])] for i in choice_edges_1]
            new_edges_1 = [[find_node(new_graph_first, i[0]), find_node(new_graph_first, i[1])] for i in choice_edges_2] 
            for pair in new_edges_1:
                if pair not in old_edges1:
                    new_graph_first.operator.connect_nodes(pair[0], pair[1])
            for pair in new_edges_2:
                if pair not in old_edges2:
                    new_graph_second.operator.connect_nodes(pair[0], pair[1])                                             
            
            if old_edges1 != new_graph_first.operator.get_edges() or old_edges2 != new_graph_second.operator.get_edges():
                break

        if old_edges1 == new_graph_first.operator.get_edges() and new_edges_1!=[] and new_edges_1!=None:
            new_graph_first = deepcopy(graph_first)
        if old_edges2 == new_graph_second.operator.get_edges() and new_edges_2!=[] and new_edges_2!=None:
            new_graph_second = deepcopy(graph_second)
    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second

def custom_crossover_exchange_parents_both(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):
    
    def find_node(graph: CompositeModel, node):
        name = node.content['name']
        for graph_node in graph.nodes:
            if graph_node.content['name'] == name:
                return graph_node
    
    num_cros = 100
    try:
        for _ in range(num_cros):
            old_edges1 = []
            old_edges2 = []
            parents1 = []
            parents2 = []
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            edges = new_graph_second.operator.get_edges()
            flatten_edges = list(chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            if nodes_with_parent_or_child!=[]:
                
                selected_node2=choice(nodes_with_parent_or_child)
                parents2=selected_node2.nodes_from

                selected_node1=find_node(new_graph_first, selected_node2)
                parents1=selected_node1.nodes_from
                
                if parents1:
                    for p in parents1:
                        new_graph_first.operator.disconnect_nodes(p, selected_node1, False)
                if parents2:
                    for p in parents2:
                        new_graph_second.operator.disconnect_nodes(p, selected_node2, False)

                old_edges1 = new_graph_first.operator.get_edges()
                old_edges2 = new_graph_second.operator.get_edges()

                if parents2!=[] and parents2!=None:
                    parents_in_first_graph=[find_node(new_graph_first, i) for i in parents2]
                    for parent in parents_in_first_graph:
                        if [parent, selected_node1] not in old_edges1:
                            new_graph_first.operator.connect_nodes(parent, selected_node1)

                if parents1!=[] and parents1!=None:
                    parents_in_second_graph=[find_node(new_graph_second, i) for i in parents1]
                    for parent in parents_in_second_graph:
                        if [parent, selected_node2] not in old_edges2:
                            new_graph_second.operator.connect_nodes(parent, selected_node2)            


            if old_edges1 != new_graph_first.operator.get_edges() or old_edges2 != new_graph_second.operator.get_edges():
                break    
        
        if old_edges1 == new_graph_first.operator.get_edges() and parents2!=[] and parents2!=None:
            new_graph_first = deepcopy(graph_first)                
        if old_edges2 == new_graph_second.operator.get_edges() and parents1!=[] and parents1!=None:
            new_graph_second = deepcopy(graph_second)       

    except Exception as ex:
        print(ex)    
    return new_graph_first, new_graph_second

def custom_crossover_all_model(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):

    def find_node(graph: CompositeModel, node):
        name = node.content['name']
        for graph_node in graph.nodes:
            if graph_node.content['name'] == name:
                return graph_node    
        
    num_cros = 100
    try:
        for _ in range(num_cros):
            selected_node1=choice(graph_first.nodes)
            if selected_node1.nodes_from == None or selected_node1.nodes_from == []:
                continue
            
            selected_node2=find_node(graph_second, selected_node1)           
            if selected_node2.nodes_from == None or selected_node2.nodes_from == []:
                continue            

            model1 = selected_node1.content['parent_model']
            model2 = selected_node2.content['parent_model']

            selected_node1.content['parent_model'] = model2
            selected_node2.content['parent_model'] = model1

            break

    except Exception as ex:
        print(ex)
    return graph_first, graph_second


# Структурные мутации
# задаем три варианта мутации: добавление узла, удаление узла, разворот узла
def custom_mutation_add_structure(graph: CompositeModel, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            nodes_not_cycling = (random_node.descriptive_id not in
                                 [n.descriptive_id for n in ordered_subnodes_hierarchy(other_random_node)] and
                                 other_random_node.descriptive_id not in
                                 [n.descriptive_id for n in ordered_subnodes_hierarchy(random_node)])
            if nodes_not_cycling:
                graph.operator.connect_nodes(random_node, other_random_node)
                break

    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph
 

def custom_mutation_delete_structure(graph: CompositeModel, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.disconnect_nodes(other_random_node, random_node, False)
                break
    except Exception as ex:
        print(ex) 
    return graph


def custom_mutation_reverse_structure(graph: CompositeModel, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.reverse_edge(other_random_node, random_node)   
                break         
    except Exception as ex:
        print(ex)  
    return graph


def connect_nodes(self, parent: CompositeNode, child: CompositeNode):
    if child.descriptive_id not in [p.descriptive_id for p in ordered_subnodes_hierarchy(parent)]:
        try:
            if child.nodes_from==None or child.nodes_from==[]:
                child.nodes_from = []
                child.nodes_from.append(parent)
                ml_models = ML_models()
                child.content['parent_model'] = ml_models.get_model_by_children_type(child)                
            else:                      
                child.nodes_from.append(parent)
        except Exception as ex:
            print(ex)

def disconnect_nodes(self, node_parent: CompositeNode, node_child: CompositeNode,
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

    if node_child.nodes_from == [] or node_child.nodes_from == None:
        node_child.content['parent_model'] = None



def reverse_edge(self, node_parent: CompositeNode, node_child: CompositeNode):
    self.disconnect_nodes(node_parent, node_child, False)
    self.connect_nodes(node_child, node_parent)

LinkedGraph.reverse_edge = reverse_edge
LinkedGraph.connect_nodes = connect_nodes
LinkedGraph.disconnect_nodes = disconnect_nodes



def custom_mutation_add_model(graph: CompositeModel, **kwargs):
    try:
        all_nodes = graph.nodes
        nodes_with_parents = [node for node in all_nodes if (node.nodes_from!=[] and node.nodes_from!=None)]
        if nodes_with_parents == []:
            return graph
        node = choice(nodes_with_parents)
        ml_models = ML_models()
        node.content['parent_model'] = ml_models.get_model_by_children_type(node)
        # node.content['parent_model'] = random_choice_model(node)
    except Exception as ex:
        print(ex)  
    return graph

def mutation_set1(graph: CompositeModel, **kwargs):
    return custom_mutation_add_model(custom_mutation_add_structure(graph, **kwargs))

def mutation_set2(graph: CompositeModel, **kwargs): 
    return custom_mutation_add_model(custom_mutation_delete_structure(graph, **kwargs))
        
def mutation_set3(graph: CompositeModel, **kwargs):
    return custom_mutation_add_model(custom_mutation_reverse_structure(graph, **kwargs))

def cross_set(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):
    g11, g12 = custom_crossover_exchange_parents_both(graph_first, graph_second, max_depth)
    g21, g22 = custom_crossover_exchange_edges(g11, g12, max_depth)
    g31, g32 = custom_crossover_all_model(g21, g22, max_depth)
    return g31, g32

# задаем правила на запрет дублирующих узлов
def _has_no_duplicates(graph: CompositeModel):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True
    

def run_example():

    data = pd.read_csv('examples/data/'+file+'.csv')   
    if file == 'hack_processed_with_rf':
        data.drop(['Oil density', 'Oil recovery factor','Field name', 'Reservoir unit', 'Country', 'Region', 'Basin name', 'Latitude', 'Longitude', 'Operator company', 'Onshore/offshore', 'Hydrocarbon type', 'Reservoir status',  'Condensate recovery factor', 'Gas recovery factor'], axis=1, inplace=True)
    else:
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
    if file != 'hack_processed_with_rf':
        data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    # x = 'Permeability'
    # y = 'Oil recovery factor'
    # plt.scatter(data[x], data[y])
    # plt.xlabel(x)
    # plt.ylabel(y)    
    # plt.show()
    vertices = list(data.columns)
    print(len(vertices))
    print(vertices)

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder)])
    p2 = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, _ = p.apply(data)
    discretized_data2, _ = p2.apply(data)


    # словарь: {имя_узла: уникальный_номер_узла}
    # global dir_of_nodes
    # dir_of_nodes={data.columns[i]:i for i in range(len(data.columns))}     

    # правила для байесовских сетей: нет петель, нет циклов, нет повторяющихся узлов
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]


    # инициализация начальной сети (пустая)
      
    initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                    content={'name': vertex,
                                                            'type': p.nodes_types[vertex],
                                                            'parent_model': None}) 
                                                    for vertex in vertices])] 
    init = initial[0]

    # print('Создание начальной популяции')
    # def create_population(pop_size):
    #     initial = []
    #     for i in range(0, pop_size):
    #         rand = randint(1, 2*len(vertices))
    #         fl1 = False
    #         while not fl1:
    #             try:
    #                 fl1 = False
    #                 g=deepcopy(init)
    #                 for _ in range(rand):
    #                     g=deepcopy(custom_mutation_add_structure(g))
    #                 mylist = []
    #                 for rule_func in rules:
    #                     mylist.append(rule_func(g))
    #                 fl1=all(mylist)
    #             except:
    #                 pass
    #         initial.append(g)
            
    #     return initial

    # initial = create_population(pop_size)
    # print('Конец создания начальной популяции')

    # добавим для начального графа три ребра
    # init = custom_mutation_add_structure(custom_mutation_add_structure(custom_mutation_add_structure(init)))

    def structure_to_opt_graph(fdt, structure):

        encoder = preprocessing.LabelEncoder()
        # discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        p = pp.Preprocessor([('encoder', encoder)])
        discretized_data, est = p.apply(data)

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



#########################
# семплирование
    # init = structure_to_opt_graph(init, [('H', 'A'), ('C', 'A'), ('H', 'C'), ('H', 'D'), ('O', 'D'), ('D', 'I'), ('C', 'I'), ('C', 'O'), ('A', 'O'), ('O', 'T'), ('I', 'T')])
    # h 2 2
    # a 1 5

    # di = {'A': 'mlp',
    # 'C': 'mlp',
    # 'D': 'lgbmreg',
    # 'I': 'treg',
    # 'O': 'treg',
    # 'T': 'linear'}

    # for node in init.nodes:
    #     if not (node.nodes_from == None or node.nodes_from == []):
    #         node.content['parent_model'] = SkLearnEvaluationStrategy(di[node.content['name']])
    
    # bn = Nets.HybridBN(use_mixture=True, has_logit=True)
    # info = p.info
    # bn.add_nodes(info)
    # structure = [(str(edge[0]), str(edge[1])) for edge in init.get_edges()]
    # bn.set_structure(edges=structure)
    # print(bn.get_info())
    # dict_reg = {}
    # for n in init.nodes:
    #     dict_reg[str(n)] = n.content['parent_model']
    # bn.set_regressor(regressors=dict_reg)
    # bn.fit_parameters(data)
    # sample = bn.sample(1000)    
    # print(sample)

    # data['C'].value_counts().plot(kind='bar', grid=True, color='#607c8e')
    # plt.ylabel('Count') 
    # plt.xlabel('Values') 
    # plt.show()
    # sample['I'].plot.hist(grid=True, bins=20, rwidth=0.9,
    #                 color='#607c8e')
    # plt.xlabel('Values')  
    # plt.show()
    # data['I'].plot.hist(grid=True, bins=20, rwidth=0.9,
    #                 color='#607c8e')       
    # plt.xlabel('Values')         
    # plt.show()

#########################

    # data.dropna(inplace=True)
    # data.reset_index(inplace=True, drop=True)
    # predict = bn.predict(data.iloc[:, :6])
    # print((predict))

# назначить parent_model узлам с родителями    
    for node in init.nodes:
        if not (node.nodes_from == None or node.nodes_from == []):
            ml_models = ML_models()
            node.content['parent_model'] = ml_models.get_model_by_children_type(node)            
            # node.content['parent_model'] = random_choice_model(node)   


    def bamt_sсore():
        initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                        content={'name': vertex,
                                                                'type': p.nodes_types[vertex],
                                                                'parent_model': None}) 
                                                        for vertex in vertices])] 
        init = initial[0]
        
        types=list(p.info['types'].values())
        if 'cont' in types and ('disc' in types or 'disc_num' in types):
            bn = HybridBN(has_logit=False, use_mixture=False)
            rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
        elif 'disc' in types or 'disc_num' in types:
            bn = DiscreteBN()
            rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
        elif 'cont' in types:
            bn = ContinuousBN(use_mixture=False)
            rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

        bn.add_nodes(p.info)
        bn.add_edges(discretized_data2, scoring_function=('K2', K2Score))        

        for node in init.nodes: 
            parents = []
            for n in bn.nodes:
                if str(node) == str(n):
                    parents = n.cont_parents + n.disc_parents
                    break
            for n2 in init.nodes:
                if str(n2) in parents:
                    node.nodes_from.append(n2)

        for node in init.nodes:
            if not (node.nodes_from == None or node.nodes_from == []):
                if node.content['type'] == 'cont':
                    node.content['parent_model'] = SklearnLinReg
                else:
                    node.content['parent_model'] = SklearnLogReg
    
        return print('score_bamt', composite_metric(init, discretized_data))
    # генерация случайных индивидов
        # def child_dict(net: list):
        #     res_dict = dict()
        #     for e0, e1 in net:
        #         if e1 in res_dict:
        #             res_dict[e1].append(e0)
        #         else:
        #             res_dict[e1] = [e0]
        #     return res_dict

        # def precision_recall(pred, true_net: list, decimal = 2):

        #     edges= pred.get_edges()
        #     struct = []
        #     for s in edges:
        #         struct.append((s[0].content['name'], s[1].content['name']))

        #     pred_net = deepcopy(struct)

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
        #     return {
        #     'SHD': shd}
        
        # true_net = [('Erk', 'Akt'), ('Mek', 'Erk'), ('PIP3', 'PIP2'), ('PKA', 'Akt'), ('PKA', 'Erk'), ('PKA', 'Jnk'), ('PKA', 'Mek'), ('PKA', 'P38'), ('PKA', 'Raf'), ('PKC', 'Jnk'), ('PKC', 'Mek'), ('PKC', 'P38'), ('PKC', 'PKA'), ('PKC', 'Raf'), ('Plcg', 'PIP2'), ('Plcg', 'PIP3'), ('Raf', 'Mek')]  
        # SHD = precision_recall(init, true_net)['SHD']       
        # print(SHD)

       
        
    
    
    def true_sсore():
        initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                        content={'name': vertex,
                                                                'type': p.nodes_types[vertex],
                                                                'parent_model': None}) 
                                                        for vertex in vertices])] 
        init = initial[0]
        
        dict_true_str = {'asia':
        [('asia', 'tub'), ('tub', 'either'), ('smoke', 'lung'), ('smoke', 'bronc'), ('lung', 'either'), ('bronc', 'dysp'), ('either', 'xray'), ('either', 'dysp')],

        'cancer':
        [('Pollution', 'Cancer'), ('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea')],

        'earthquake':
        [('Burglary', 'Alarm'), ('Earthquake', 'Alarm'), ('Alarm', 'JohnCalls'), ('Alarm', 'MaryCalls')],

        'sachs':
        [('Erk', 'Akt'), ('Mek', 'Erk'), ('PIP3', 'PIP2'), ('PKA', 'Akt'), ('PKA', 'Erk'), ('PKA', 'Jnk'), ('PKA', 'Mek'), ('PKA', 'P38'), ('PKA', 'Raf'), ('PKC', 'Jnk'), ('PKC', 'Mek'), ('PKC', 'P38'), ('PKC', 'PKA'), ('PKC', 'Raf'), ('Plcg', 'PIP2'), ('Plcg', 'PIP3'), ('Raf', 'Mek')],  

        'healthcare':
        [('A', 'C'), ('A', 'D'), ('A', 'H'), ('A', 'O'), ('C', 'I'), ('D', 'I'), ('H', 'D'), ('I', 'T'), ('O', 'T')],
        
        'child':
        [('BirthAsphyxia', 'Disease'), ('HypDistrib', 'LowerBodyO2'), ('HypoxiaInO2', 'LowerBodyO2'), ('HypoxiaInO2', 'RUQO2'), ('CO2', 'CO2Report'), ('ChestXray', 'XrayReport'), ('Grunting', 'GruntingReport'), ('Disease', 'Age'), ('Disease', 'LVH'), ('Disease', 'DuctFlow'), ('Disease', 'CardiacMixing'), ('Disease', 'LungParench'), ('Disease', 'LungFlow'), ('Disease', 'Sick'), ('LVH', 'LVHreport'), ('DuctFlow', 'HypDistrib'), ('CardiacMixing', 'HypDistrib'), ('CardiacMixing', 'HypoxiaInO2'), ('LungParench', 'HypoxiaInO2'), ('LungParench', 'CO2'), ('LungParench', 'ChestXray'), ('LungParench', 'Grunting'), ('LungFlow', 'ChestXray'), ('Sick', 'Grunting'), ('Sick', 'Age')],
        
        'magic-niab':
        [('YR.GLASS', 'YR.FIELD'), ('YR.GLASS', 'YLD'), ('HT', 'YLD'), ('HT', 'FUS'), ('MIL', 'YR.GLASS'), ('FT', 'YR.FIELD'), ('FT', 'YLD'), ('G418', 'YR.GLASS'), ('G418', 'YR.FIELD'), ('G418', 'G1294'), ('G418', 'G2835'), ('G311', 'YR.GLASS'), ('G311', 'G43'), ('G1217', 'YR.GLASS'), ('G1217', 'MIL'), ('G1217', 'G257'), ('G1217', 'G1800'), ('G800', 'YR.GLASS'), ('G800', 'G383'), ('G866', 'YR.GLASS'), ('G795', 'YR.GLASS'), ('G2570', 'YLD'), ('G260', 'YLD'), ('G2920', 'YLD'), ('G832', 'HT'), ('G832', 'YLD'), ('G832', 'FUS'), ('G1896', 'HT'), ('G1896', 'FUS'), ('G2953', 'HT'), ('G2953', 'G1896'), ('G2953', 'G1800'), ('G266', 'HT'), ('G266', 'FT'), ('G266', 'G1789'), ('G847', 'HT'), ('G942', 'HT'), ('G200', 'YR.FIELD'), ('G257', 'YR.FIELD'), ('G257', 'G2208'), ('G257', 'G1800'), ('G2208', 'YR.FIELD'), ('G2208', 'MIL'), ('G1373', 'YR.FIELD'), ('G599', 'YR.FIELD'), ('G599', 'G1276'), ('G261', 'YR.FIELD'), ('G383', 'FUS'), ('G1853', 'G311'), ('G1853', 'FUS'), ('G1033', 'FUS'), ('G1945', 'MIL'), ('G1338', 'MIL'), ('G1338', 'G266'), ('G1276', 'FT'), ('G1276', 'G266'), ('G1263', 'FT'), ('G2318', 'FT'), ('G1294', 'FT'), ('G1800', 'FT'), ('G1750', 'YR.GLASS'), ('G1750', 'G1373'), ('G524', 'MIL'), ('G775', 'FT'), ('G2835', 'HT'), ('G2835', 'G1800')],
        
        'hack_processed_with_rf':
        [
        ('Tectonic regime','Structural setting'), ('Structural setting', 'Depth'), ('Structural setting', 'Gross'),
        ('Structural setting', 'Period'), ('Gross', 'Netpay'), ('Period', 'Porosity'),  ('Period', 'Gross'), 
        ('Porosity', 'Depth'), ('Porosity', 'Permeability'), ('Lithology', 'Gross'), ('Lithology', 'Permeability')
        ]
        }     
        init = structure_to_opt_graph(init, dict_true_str[file])
        
        for node in init.nodes:
            if not (node.nodes_from == None or node.nodes_from == []):
                if node.content['type'] == 'cont':
                    node.content['parent_model'] = SklearnLinReg
                else:
                    node.content['parent_model'] = SklearnLogReg   
    
        # return print('score_true', composite_metric(init, discretized_data))    
        return init
    
 
    graph = true_sсore()


    # задаем для оптимизатора fitness-функцию
    # objective = Objective(composite_metric) 
    objective = Objective({'custom': composite_metric})
    objective_eval = ObjectiveEvaluate(objective, data = discretized_data)    

    requirements = GraphRequirements(
        max_arity=100,
        max_depth=100, 
        num_of_generations=n_generation,
        timeout=timedelta(minutes=time_m),
        history_dir = None
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
        custom_mutation_reverse_structure, 
        custom_mutation_add_model
        # mutation_set1,
        # mutation_set2,
        # mutation_set3
        ],

        crossover_types = [
            custom_crossover_exchange_edges,
            custom_crossover_exchange_parents_both,
            custom_crossover_all_model
            # cross_set
            ]
    )

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CompositeModel, base_node_class=CompositeNode),
        rules_for_constraint=rules,
        # node_factory=DefaultOptNodeFactory(available_node_types=nodes_types)
        )

    optimiser = EvoGraphOptimizer(
        graph_generation_params=graph_generation_params,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements,
        initial_graphs=[init],
        objective=objective)
    optimiser.number = number


    # запуск оптимизатора
    optimized_graph = optimiser.optimise(objective_eval)[0]


    # bn = Nets.HybridBN(use_mixture=True, has_logit=True)
    # info = p.info
    # bn.add_nodes(info)
    # structure = [(str(edge[0]), str(edge[1])) for edge in optimized_graph.get_edges()]
    # bn.set_structure(edges=structure)
    # print(bn.get_info())
    # dict_reg = {}
    # for n in optimized_graph.nodes:
    #     dict_reg[str(n)] = n.content['parent_model']
    # bn.set_regressor(regressors=dict_reg)
    # bn.fit_parameters(data)
    # # sample = bn.sample(2500)
    # # data.dropna(inplace=True)
    # # data.reset_index(inplace=True, drop=True)
    # predict = bn.predict(data.iloc[:, :3])

    graph = optimized_graph
    name_nodes = [str(n) for n in graph.nodes]
    graph_nodes = filter(lambda x: str(x) in name_nodes, graph.nodes)
    for node in graph_nodes:
        if node.content['parent_model'] != None:
            model_name = str(node.content['parent_model']).split('.')[-1][:-2]
            parents = [n.content['name'] for n in node.nodes_from]
            new_node = CompositeNode(nodes_from=None,
                                        content={'name': model_name,
                                                'type' : 'model',
                                                'parent_model': None})
            graph.add_node(new_node)

            for parent in parents:
                parent = [n for n in graph.nodes if n.content['name'] == parent][0]
                graph.operator.disconnect_nodes(parent, node, False)
                graph.operator.connect_nodes(parent, new_node)   

            graph.operator.connect_nodes(new_node, node)
            graph.operator.sort_nodes()

    size_dict = {}
    color_dict = {}
    for n in graph.nodes:
        if n.content['type'] == 'model':
            size_dict[n.content['name']] = 10
            color_dict[n.content['name']] = 'yellow'
        else:
            size_dict[n.content['name']] = 30
            color_dict[n.content['name']] = 'pink'
    

    graph.show(engine = 'pyvis', node_color = color_dict, node_size_scale = size_dict,
    save_path=(parentdir + '\\examples\\composite_bn\\' + 'graph' + '_' + str(k) + '_' + str(file) + '_' + str(number) + '.html'))





if __name__ == '__main__':

    # файл с исходными данными (должен лежать в 'examples/data/')
    file = 'asia'
    k = 1
    # размер популяции 
    pop_size = 20
    # количество поколений
    n_generation = 1000
    # вероятность кроссовера
    crossover_probability = 0.8
    # вероятность мутации
    mutation_probability = 0.9
    time_m = 1000
    n = 5
    number = 1
    while number <= n:
        run_example() 
        number += 1 






