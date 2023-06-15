
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
from composite_model import CompositeModel
from composite_node import CompositeNode
from scipy.stats import norm
from numpy import std, mean, log
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from ML import ML_models
from composite_bn_genetic_operators import (
    custom_crossover_all_model, 
    custom_mutation_add_structure, 
    custom_mutation_delete_structure, 
    custom_mutation_reverse_structure,
    custom_mutation_add_model,
)
from math import log10


def composite_metric(graph: CompositeModel, data: pd.DataFrame, percent = 0.02):

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
            model, columns, target, idx = ML_models().dict_models[node.content['parent_model']](), [n.content['name'] for n in node.nodes_from], data_of_node_train.to_numpy(), data_train.index.to_numpy()
            setattr(model, 'max_iter', 100000)
            features = data_train[columns].to_numpy()                
            if len(set(target)) == 1:
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

    return -score

# задаем правила на запрет дублирующих узлов
def _has_no_duplicates(graph: CompositeModel):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True


def run_example(file):

    data = pd.read_csv('examples/data/csv/'+file+'.csv')   

    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    vertices = list(data.columns)

    encoder = preprocessing.LabelEncoder()
    p = pp.Preprocessor([('encoder', encoder)])
    discretized_data, _ = p.apply(data)

    # правила для байесовских сетей: нет петель, нет циклов, нет повторяющихся узлов
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]


    # инициализация начальной сети (пустая)
    initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                    content={'name': vertex,
                                                            'type': p.nodes_types[vertex],
                                                            'parent_model': None}) 
                                                    for vertex in vertices])] 


    # задаем для оптимизатора fitness-функцию
    objective = Objective({'custom': composite_metric})
    objective_eval = ObjectiveEvaluate(objective, data = discretized_data)    

    requirements = GraphRequirements(
        max_arity=100,
        max_depth=100, 
        num_of_generations=n_generation,
        timeout=timedelta(minutes=time_m),
        history_dir = None,
        early_stopping_iterations = early_stopping_iterations,
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
            exchange_parents_one_crossover,
            exchange_parents_both_crossover,
            exchange_edges_crossover,
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
        initial_graphs=initial,
        objective=objective)

    # запуск оптимизатора
    optimized_graph = optimiser.optimise(objective_eval)[0]
    optimized_graph


if __name__ == '__main__':

    file = 'healthcare'
    # размер популяции 
    pop_size = 40
    # количество поколений
    n_generation = 1000
    # вероятность кроссовера
    crossover_probability = 0.8
    # вероятность мутации
    mutation_probability = 0.9
    # stopping_after_n_generation
    early_stopping_iterations = 20
    time_m = 1000
    

    run_example(file) 






