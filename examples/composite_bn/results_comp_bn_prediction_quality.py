
from audioop import cross
from datetime import timedelta
import sys
import os
parentdir = os.getcwd()
sys.path.insert(0, parentdir)
import pandas as pd
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
from golem.core.optimisers.genetic.operators.crossover import exchange_parents_one_crossover, exchange_parents_both_crossover, exchange_edges_crossover
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.objective.objective_eval import ObjectiveEvaluate
from golem.core.optimisers.optimizer import GraphGenerationParams
from pgmpy.estimators import K2Score
from composite_model import CompositeModel
from composite_node import CompositeNode
from bamt.networks.hybrid_bn import HybridBN
from bamt.networks.discrete_bn import DiscreteBN
from bamt.networks.continuous_bn import ContinuousBN
from scipy.stats import norm
from numpy import std, mean, log
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from golem.core.dag.linked_graph import LinkedGraph
from golem.core.dag.graph_utils import ordered_subnodes_hierarchy
from ML import ML_models
# from sklearn.linear_model import (LinearRegression as SklearnLinReg, LogisticRegression as SklearnLogReg)
from composite_bn_genetic_operators import (
    custom_crossover_exchange_edges, 
    custom_crossover_exchange_parents_both, 
    custom_crossover_all_model, 
    custom_mutation_add_structure, 
    custom_mutation_delete_structure, 
    custom_mutation_reverse_structure,
    custom_mutation_add_model,
    cross_set, mutation_set1, mutation_set2, mutation_set3)
from math import log10
import linecache
import itertools
from sklearn.metrics import f1_score

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDRegressor
)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


def custom_crossover_exchange_parents_one(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):
    
    new_graph_first, new_graph_second = exchange_parents_one_crossover(graph_first, graph_second, 2)
    for node in new_graph_first.nodes:
        if node.nodes_from and not node.content['parent_model']:
            ml_models = ML_models()
            node.content['parent_model'] = ml_models.get_model_by_children_type(node)   
        elif not node.nodes_from and node.content['parent_model']: 
            node.content['parent_model'] = None
    for node in new_graph_second.nodes:
        if node.nodes_from and not node.content['parent_model']:
            ml_models = ML_models()
            node.content['parent_model'] = ml_models.get_model_by_children_type(node)            
        elif not node.nodes_from and node.content['parent_model']: 
            node.content['parent_model'] = None
    return new_graph_first, new_graph_second


def custom_crossover_exchange_parents_both(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):
    
    new_graph_first, new_graph_second = exchange_parents_both_crossover(graph_first, graph_second, 2)
    for node in new_graph_first.nodes:
        if node.nodes_from and not node.content['parent_model']:
            ml_models = ML_models()
            node.content['parent_model'] = ml_models.get_model_by_children_type(node)   
        elif not node.nodes_from and node.content['parent_model']: 
            node.content['parent_model'] = None
    for node in new_graph_second.nodes:
        if node.nodes_from and not node.content['parent_model']:
            ml_models = ML_models()
            node.content['parent_model'] = ml_models.get_model_by_children_type(node)            
        elif not node.nodes_from and node.content['parent_model']: 
            node.content['parent_model'] = None
    return new_graph_first, new_graph_second

def custom_crossover_exchange_edges(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):

    new_graph_first, new_graph_second = exchange_edges_crossover(graph_first, graph_second, 2)
    for node in new_graph_first.nodes:
        if node.nodes_from and not node.content['parent_model']:
            ml_models = ML_models()
            node.content['parent_model'] = ml_models.get_model_by_children_type(node)  
        elif not node.nodes_from and node.content['parent_model']: 
            node.content['parent_model'] = None
    for node in new_graph_second.nodes:
        if node.nodes_from and not node.content['parent_model']:
            ml_models = ML_models()
            node.content['parent_model'] = ml_models.get_model_by_children_type(node)            
        elif not node.nodes_from and node.content['parent_model']: 
            node.content['parent_model'] = None
    return new_graph_first, new_graph_second

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
                setattr(model, 'max_iter', 100000)
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
                        a = predict_proba[i]
                        try:
                            b = a[target[i]]
                        except:
                            b = 0.0000001
                        if b<0.0000001:
                            b = 0.0000001
                        li.append(log(b))
                    score += sum(li)
    
        edges_count = len(graph.get_edges())
        score -= (edges_count*percent)*log10(len_data)*edges_count    
    except Warning: # RuntimeWarning, ConvergenceWarning
        print(Warning)
        # PrintException()
    return -score

# задаем правила на запрет дублирующих узлов
def _has_no_duplicates(graph: CompositeModel):
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
    if file == 'hack_processed_with_rf':
        data.drop(['Oil density', 'Oil recovery factor','Field name', 'Reservoir unit', 'Country', 'Region', 'Basin name', 'Latitude', 'Longitude', 'Operator company', 'Onshore/offshore', 'Hydrocarbon type', 'Reservoir status',  'Condensate recovery factor', 'Gas recovery factor'], axis=1, inplace=True)
    else:
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
    if file != 'hack_processed_with_rf':
        data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    vertices = list(data.columns)

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder)])
    p2 = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, _ = p.apply(data)
    discretized_data2, _ = p2.apply(data)


    # правила для байесовских сетей: нет петель, нет циклов, нет повторяющихся узлов
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]


    # инициализация начальной сети (пустая)
      
    initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                    content={'name': vertex,
                                                            'type': p.nodes_types[vertex],
                                                            'parent_model': None}) 
                                                    for vertex in vertices])] 
    init = initial[0]


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


# назначить parent_model узлам с родителями    
    # for node in init.nodes:
    #     if not (node.nodes_from == None or node.nodes_from == []):
    #         ml_models = ML_models()
    #         node.content['parent_model'] = ml_models.get_model_by_children_type(node)            


    def bamt_golem():
        initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                        content={'name': vertex,
                                                                'type': p.nodes_types[vertex],
                                                                'parent_model': None}) 
                                                        for vertex in vertices])] 
        init = initial[0]
        
        types=list(p2.info['types'].values())
        if 'cont' in types and ('disc' in types or 'disc_num' in types):
            bn = HybridBN(has_logit=False, use_mixture=False)
            rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
        elif 'disc' in types or 'disc_num' in types:
            bn = DiscreteBN()
            rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
        elif 'cont' in types:
            bn = ContinuousBN(use_mixture=False)
            rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

        bn.add_nodes(p2.info)
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
                    node.content['parent_model'] = LinearRegression
                else:
                    node.content['parent_model'] = LogisticRegression
    
        # return print('score_bamt', composite_metric(init, discretized_data))
        return init
 
    init_bamt = bamt_golem()
    
    def true_golem():
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
                    node.content['parent_model'] = LinearRegression
                else:
                    node.content['parent_model'] = LogisticRegression   
    
        return init
    
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

    # задаем для оптимизатора fitness-функцию
    objective = Objective({'custom': composite_metric})
    objective_eval = ObjectiveEvaluate(objective, data = discretized_data)    

    requirements = GraphRequirements(
        max_arity=100,
        max_depth=100, 
        num_of_generations=n_generation,
        timeout=timedelta(minutes=time_m),
        history_dir = None,
        n_jobs=-1
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
        custom_mutation_add_model    
        ],

        crossover_types = [
            custom_crossover_exchange_parents_one,
            custom_crossover_exchange_parents_both,
            custom_crossover_exchange_edges,
            custom_crossover_all_model
            ]
    )

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CompositeModel, base_node_class=CompositeNode),
        rules_for_constraint=rules,
        )

    optimiser = EvoGraphOptimizer(
        graph_generation_params=graph_generation_params,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements,
        initial_graphs=[init_bamt],
        objective=objective)

    # запуск оптимизатора
    optimized_graph = optimiser.optimise(objective_eval)[0]

    optimized_structure = [(str(edge[0]), str(edge[1])) for edge in optimized_graph.get_edges()]
    BAMT_network = init_bamt
    structure_BAMT = [(str(edge[0]), str(edge[1])) for edge in BAMT_network.get_edges()]
    Score_BAMT = round(composite_metric(BAMT_network, data=discretized_data), 6)    
    score_GA = composite_metric(optimized_graph, discretized_data)
    SHD_GA = precision_recall(optimized_structure, true_net)['SHD']
    SHD_BAMT = precision_recall(structure_BAMT, true_net)['SHD']
    f1_GA = F1(true_net, optimized_structure)
    f1_BAMT = F1(true_net, structure_BAMT)
    spent_time = optimiser.timer.minutes_from_start

    

    if 'cont' in p.info['types'].values() and ('disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values()):
        bn = HybridBN(has_logit=True, use_mixture=False)
    elif 'disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values():
        bn = DiscreteBN()
    elif 'cont' in p.info['types'].values():
        bn = ContinuousBN(use_mixture=False)    
    # bn = Nets.HybridBN(use_mixture=False, has_logit=True)
    info = p.info
    bn.add_nodes(info)


    # initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
    #                                                 content={'name': vertex,
    #                                                         'type': p.nodes_types[vertex],
    #                                                         'parent_model': None}) 
    #                                                 for vertex in vertices])] 
    # init = initial[0]
    
    # types=list(p2.info['types'].values())
    # if 'cont' in types and ('disc' in types or 'disc_num' in types):
    #     bn = HybridBN(has_logit=False, use_mixture=False)
    #     rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    # elif 'disc' in types or 'disc_num' in types:
    #     bn = DiscreteBN()
    #     rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    # elif 'cont' in types:
    #     bn = ContinuousBN(use_mixture=False)
    #     rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

    # bn.add_nodes(p2.info)
    # bn.add_edges(discretized_data2, scoring_function=('K2', K2Score))        

    # for node in init.nodes: 
    #     parents = []
    #     for n in bn.nodes:
    #         if str(node) == str(n):
    #             parents = n.cont_parents + n.disc_parents
    #             break
    #     for n2 in init.nodes:
    #         if str(n2) in parents:
    #             node.nodes_from.append(n2)

    # for node in init.nodes:
    #     if not (node.nodes_from == None or node.nodes_from == []):
    #         if node.content['type'] == 'cont':
    #             node.content['parent_model'] = LinearRegression
    #         else:
    #             node.content['parent_model'] = LogisticRegression

    # # optimized_graph = init

    structure = [(str(edge[0]), str(edge[1])) for edge in optimized_graph.get_edges()]
    print('structure = ', structure)

    bn.set_structure(edges=structure)
    dict_reg = {}
    dict_class = {}
    for n in optimized_graph.nodes:
        if n.content['parent_model'] != None:
            if n.content['type'] == 'disc':
                dict_class[str(n)] = n.content['parent_model']()
            else:
                dict_reg[str(n)] = n.content['parent_model']()
    print('dict_reg = ', dict_reg)
    print('dict_class = ', dict_class)
    bn.set_regressor(regressors=dict_reg)
    # bn.set_classifiers(classifiers=dict_class)
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    p = pp.Preprocessor([])
    data, _ = p.apply(data)
    bn.fit_parameters(data)

    result = {'accuracy_score': {},
              'mean_squared_error': {}}

    # предсказание
    for node_bn in bn.nodes:
        node_name = node_bn.name
        other_node = [n for n in vertices if n != node_name]
        features = data[other_node]
        try:
            predict = pd.DataFrame(list(bn.predict(features).values())[0])   
            if bn.descriptor['types'][node_bn.name] == 'disc':
                acc_score = accuracy_score(data[node_name].values, predict)
                print(node_name, '_accuracy_score = ', acc_score)
                result['accuracy_score'][node_name] = acc_score
            else:
                mean_sq_error = mean_squared_error(data[node_name].values, predict, squared=False)
                print(node_name, '_mean_squared_error = ', mean_sq_error)
                result['mean_squared_error'][node_name] = mean_sq_error
        except:
            print('On node {} error.'.format(node_name))


    textfile = open('C:\\Users\\anaxa\\Desktop\\РЕЗУЛЬТАТЫ\\composite_' + file + "_"  + str(number) + ".txt", "a")   # + 'start_bamt' + "_" 
    textfile.write('Structure = ' + str(optimized_structure) + '\n')
    textfile.write('Score GA = ' + str(score_GA) + '\n')
    textfile.write('Score BAMT = ' + str(Score_BAMT) + '\n')
    textfile.write('SHD GA = ' + str(SHD_GA) + '\n')
    textfile.write('SHD BAMT = ' + str(SHD_BAMT) + '\n')
    textfile.write('F1 GA = ' + str(f1_GA) + '\n')
    textfile.write('F1 BAMT = ' + str(f1_BAMT) + '\n')    
    textfile.write('Spent_time = ' + str(spent_time) + '\n')
    textfile.write('Generation number = ' + str(optimiser.current_generation_num) + '\n')
    textfile.write('Population number = ' + str(optimiser.graph_optimizer_params.pop_size) + '\n')
    textfile.write('Parent model = ' + str(dict_reg|dict_class) + '\n')
    textfile.write('Prediction quality = ' + str(result) + '\n')
    textfile.close() 

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
    save_path=('C:\\Users\\anaxa\\Desktop\\composite\\' + 'bamt_composite' + '_' + str(file) + '_' + str(number) + '.html'))



if __name__ == '__main__':

    # файл с исходными данными (должен лежать в 'examples/data/')  'healthcare', 'sangiovese', 'magic-niab', 'mehra-complete','ecoli70' 
    files = ['healthcare', 'mehra-complete','ecoli70'] # ['asia', 'cancer', 'earthquake', 'survey', 'sachs', 'alarm', 'barley', 'child', 'water', 'ecoli70', 'magic-niab', 'healthcare', 'sangiovese', 'mehra-complete'] 
    k = 1
    # размер популяции 
    pop_size = 40 #40
    # количество поколений
    n_generation = 10000 #10000
    # вероятность кроссовера
    crossover_probability = 0.8
    # вероятность мутации
    mutation_probability = 0.9
    time_m = 1000
    percent = 0.02
    n = 1
    for file in files:
        number = 1
        while number <= n:
            run_example() 
            number += 1 






