from datetime import timedelta
import sys
import os
parentdir = os.getcwd()
sys.path.insert(0, parentdir)
import pandas as pd
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
from bn_model import  BNModel
from bn_node import BNNode
from bn_genetic_operators import (
    custom_mutation_add_structure, 
    custom_mutation_delete_structure, 
    custom_mutation_reverse_structure)


def custom_metric(graph: BNModel, data: pd.DataFrame):
    if graph.get_edges():
        structure = [(str(edge[0]), str(edge[1])) for edge in graph.get_edges()]
        shd = precision_recall(structure, true_net)['SHD']
    else:
        shd = len(true_net)
    return shd

def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True
  
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
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data.reset_index(inplace=True, drop=True)

    global vertices
    vertices = list(data.columns)

    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

    objective = Objective({'custom': custom_metric})
    objective_eval = ObjectiveEvaluate(objective, data = data)    

    initial = [BNModel(nodes=[BNNode(nodes_from=[],
                                                      content={'name': v, 
                                                               }) for v in vertices])]
    
    requirements = GraphRequirements(
        max_arity=100,
        max_depth=100, 
        num_of_generations=n_generation,
        timeout=timedelta(minutes=time_m),
        history_dir = None,
        early_stopping_iterations = early_stopping_iterations,
        n_jobs = -1
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
        custom_mutation_reverse_structure
        ],
        crossover_types = crossovers
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
    
    optimiser.optimise(objective_eval)


if __name__ == '__main__':
    pop_size = 40
    n_generation = 10000
    crossover_probability = 0
    mutation_probability = 0.9
    early_stopping_iterations = 20
    time_m = 1000
    crossovers = [CrossoverTypesEnum.exchange_edges,
                CrossoverTypesEnum.exchange_parents_one,
                CrossoverTypesEnum.exchange_parents_both]
    files = ['asia', 'cancer', 'earthquake', 'survey', 'sachs', 'alarm', 'barley', 'child', 'water', 'ecoli70', 'magic-niab', 'healthcare', 'sangiovese', 'mehra-complete']
    for file in files:
        run_example()







