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
from pgmpy.estimators import K2Score, BicScore
from bn_model import  BNModel
from bn_node import BNNode
from pgmpy.models import BayesianNetwork
import seaborn as sns
import matplotlib.pyplot as plt
import bamt.networks as Nets
from golem.core.dag.linked_graph import LinkedGraph
from golem.core.dag.graph_utils import ordered_subnodes_hierarchy
from sklearn.linear_model import (LinearRegression as SklearnLinReg, LogisticRegression as SklearnLogReg)
from bn_genetic_operators import (
    custom_crossover_exchange_edges, 
    custom_crossover_exchange_parents_both, 
    custom_crossover_exchange_parents_one,
    custom_mutation_add_structure, 
    custom_mutation_delete_structure, 
    custom_mutation_reverse_structure)


# задаем метрику
def custom_metric(graph: BNModel, data: pd.DataFrame):
    score = 0
    graph_nx, labels = graph_structure_as_nx_graph(graph)
    struct = []
    for pair in graph_nx.edges():
        l1 = str(labels[pair[0]])
        l2 = str(labels[pair[1]])
        struct.append([l1, l2])
    
    bn_model = BayesianNetwork(struct)
    bn_model.add_nodes_from(data.columns)    
    
    score = K2Score(data).score(bn_model)

    return -score


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
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)
    
    # global unique_values
    # unique_values = {vertices[i]:len(pd.unique(discretized_data[vertices[i]])) for i in range(len(vertices))}
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
                                                      content={'name': v}) for v in vertices])]
    

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
        custom_mutation_reverse_structure
        ],

        crossover_types = [
            custom_crossover_exchange_edges,
            custom_crossover_exchange_parents_both,
            custom_crossover_exchange_parents_one
            ]
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

    
    # запуск оптимизатора
    optimized_graph = optimiser.optimise(objective_eval)[0]
    print(optimized_graph.operator.get_edges())




if __name__ == '__main__':
    # файл с исходными данными (должен лежать в 'examples/data/')
    file = 'asia'
    # размер популяции 
    pop_size = 20
    # количество поколений
    n_generation = 1000
    # вероятность кроссовера
    crossover_probability = 0.8
    # вероятность мутации
    mutation_probability = 0.9
    time_m = 1000
    # число запусков
    n = 1
    number = 1
    while number <= n:
        run_example() 
        number += 1 






