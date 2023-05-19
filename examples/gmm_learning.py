
import os
import sys

parentdir = os.getcwd()
sys.path.insert(0, parentdir)


from gmm_model import GMMModel
from gmm_node import GMMNode
from scipy import stats
import numpy as np
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

from random import choice, random, randint

def pdf_gmm(x, weights, means, covs):
    p = 0
    for i in range(len(weights)):
        p += weights[i] * stats.multivariate_normal.pdf(x, mean=means[i], cov=covs[i], allow_singular=True)
    return p




def olr (w, means, covs):
    n_comp = len(w)
    olr_values = []
    for i in range(n_comp):
        for j in range(i+1, n_comp, 1):
            delta = (np.array(means[j]) - np.array(means[i])) * 1/1000
            points = [np.array(means[i]) - 10*delta]
            current_point = np.array(means[i]) - 10*delta
            for k in range(1030):
                new_point = current_point + delta
                current_point = new_point
                points.append(new_point)
            w1 = w[i]
            w2 = w[j]
            w1_new = w1 / (w1 + w2)
            w2_new = 1 - w1_new
            w_new = [w1_new, w2_new]
            m_new = [means[i], means[j]]
            cov_new = [covs[i], covs[j]]
            peaks = []
            saddles = []
            for k in range(1, 1030, 1):
                pdf_k = pdf_gmm(points[k], w_new, m_new, cov_new)
                pdf_prev_k = pdf_gmm(points[k-1], w_new, m_new, cov_new)
                pdf_next_k = pdf_gmm(points[k+1], w_new, m_new, cov_new)
                if ((pdf_k - pdf_prev_k) > 0) & ((pdf_k - pdf_next_k) > 0):
                    peaks.append(pdf_k)
                if (((pdf_k - pdf_prev_k) < 0) & ((pdf_k - pdf_next_k) < 0)) | (((pdf_k - pdf_prev_k) == 0) & ((pdf_k - pdf_next_k) == 0)):
                #if ((pdf_k - pdf_prev_k) < 0) & ((pdf_k - pdf_next_k) < 0):
                    saddles.append(pdf_k)
            olr_current = 0
            if len(peaks) == 1:
                olr_current = 1
            else:
                olr_current = saddles[0] / np.min(peaks)
            olr_values.append(olr_current)
    return np.mean(olr_values)


def has_no_equal_mean(graph):
    mean= []
    for node in graph.nodes:
        mean.append(node.content['mean'])
    mean_new = list(dict.fromkeys(mean))
    if len(mean) != len(mean_new):
        raise ValueError('Custom graph has duplicates')
    return True



def metric_optimisation(gmm: GMMModel, target_olr: float):
    w = []
    mean= []
    covs = []
    for node in gmm.nodes:
        w.append(node.content['w'])
        mean.append([node.content['mean']])
        covs.append([[node.content['var']]])
    model_olr = 0
    try:
        model_olr = olr(w, mean, covs)
    except:
        print(w)
        print(mean)
        print(covs)
    metric = (model_olr - target_olr)*(model_olr - target_olr)

    return metric



def custom_crossover_exchange_mean(graph1: GMMModel,graph2: GMMModel, **kwargs):
 
    node1 = choice(graph1.nodes)
    node2 = choice(graph2.nodes)
    node1.content['mean'] = node2.content['mean']
        
    
    return graph1, graph2





def custom_mutation_change_mean(graph: GMMModel, **kwargs):
    try:
        node = choice(graph.nodes)
        node.content['mean'] = randint(0,20)
    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph



def run_example():

    number_of_components = 2
    target_olr = 0.2

    vertices = []
    for i in range(number_of_components):
        vertices.append('G'+str(i+1))


    initial = [GMMModel(nodes=[GMMNode(nodes_from=None,
                                                    content={'name': vertex,
                                                             'w':1/number_of_components,
                                                             'mean':randint(0,20),
                                                             'var':0.5
                                                            }) 
                                                    for vertex in vertices])]
    
    objective = Objective({'custom': metric_optimisation})
    objective_eval = ObjectiveEvaluate(objective, target_olr=target_olr)    

    

    requirements = GraphRequirements(
        max_arity=100,
        max_depth=100, 
        early_stopping_iterations=5,
        num_of_generations=n_generation,
        timeout=timedelta(minutes=time_m),
        history_dir = None
        )

    optimiser_parameters = GPAlgorithmParameters(
        pop_size=pop_size,
        crossover_prob=0.8, 
        mutation_prob=0.9,
        genetic_scheme_type = GeneticSchemeTypesEnum.steady_state,
        selection_types = [SelectionTypesEnum.tournament],
        mutation_types = [
        custom_mutation_change_mean
        ],
        crossover_types = [custom_crossover_exchange_mean]
    )
    rules = [has_no_equal_mean]
    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=GMMModel, base_node_class=GMMNode),
        rules_for_constraint=rules,
        # node_factory=DefaultOptNodeFactory(available_node_types=nodes_types)
        )

    optimiser = EvoGraphOptimizer(
        graph_generation_params=graph_generation_params,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements,
        initial_graphs=initial,
        objective=objective)



    # запуск оптимизатора
    optimized_graph = optimiser.optimise(objective_eval)[0]
    print('=============')
    for node in optimized_graph.nodes:
        print(node.content['name'])
        print(node.content['w'])
        print(node.content['mean'])
        print(node.content['var'])



if __name__ == '__main__':

    n_generation=200
    time_m=40
    pop_size = 30
    run_example()



