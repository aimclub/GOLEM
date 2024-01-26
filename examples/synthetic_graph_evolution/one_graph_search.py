import collections
from datetime import timedelta
from functools import partial
from wrapt_timeout_decorator import timeout
from datetime import datetime
from typing import Optional, Union, List

import pandas as pd
import time
# import timeout_decorator

from golem.core.dag.graph_delegate import GraphDelegate
from golem.core.dag.linked_graph_node import LinkedGraphNode

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
from random import choice, random, randint, sample, choices

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters

from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum

from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
import numpy as np
from golem.core.dag.verification_rules import has_no_self_cycled_nodes
import random

import pickle
from functools import partial


class GeneratorModel(GraphDelegate):
    def __init__(self, nodes: Optional[Union[LinkedGraphNode, List[LinkedGraphNode]]] = None):
        super().__init__(nodes)
        self.unique_pipeline_id = 1

    def number_of_nodes(self):
        return len(self.nodes)

    def degree(self):
        edges = self.get_edges()
        dic = {}
        for num, i in enumerate(self.nodes):
            dic[int(i.name)] = 0

        if len(edges) > 0:
            edges = np.array(edges)
            edges = np.transpose(edges)
            edge_1, edge_2 = edges.tolist()
            dic_1 = collections.Counter(edge_1)
            dic_2 = collections.Counter(edge_2)
            for node in dic_1:
                dic[int(node.name)] += dic_1[node]
            for node in dic_2:
                dic[int(node.name)] += dic_2[node]
        return dic


class GeneratorNode(LinkedGraphNode):
    def __str__(self):
        return self.content["name"]


def run_graph_search(adaptive_mutation_type,selection_type, elitism_type, regularization_type, genetic_scheme_type, dense, cycle, path, star, num_edges, des_degree, des_cluster, des_num_nodes,
                     timeout=10):
    print('datetime', datetime.now())

    # Generate target graph that will be sought by optimizer

    def avg_deg_count(des_d, G):
        d = np.mean(list(G.degree().values()))
        return (d - des_d) * (d - des_d)

    def avg_cluster_count(des_cl, G):
        G_new = nx.Graph()

        G_new.add_nodes_from(G.nodes)
        G_new.add_edges_from(G.get_edges())

        d = nx.average_clustering(G_new.to_undirected())
        return abs(d - des_cl)


    # Generate initial population with random graphs
    initial_graphs = []

    for i in range(20):
        Init2 = GeneratorModel(nodes=[GeneratorNode(nodes_from=[],
                                                    content={'name': vertex})
                                      for vertex in range(des_num_nodes)])

        init_edges = []

        i = 0
        while i < (int(des_degree * des_num_nodes / 2)):
            # print(i)
            node_1, node_2 = choices(Init2.nodes, k=2)

            if (node_1, node_2) not in init_edges and (node_2, node_1) not in init_edges and node_1 != node_2:
                init_edges.append((node_1, node_2))
                Init2.connect_nodes(node_1, node_2)
                i += 1

        initial_graphs.append(Init2)

        print('ended making graph')

    print('avg degree of random graph: {} vs des:{}'.format(np.mean(list(dict(Init2.degree()).values())), des_degree))

    nodes = []
    for num, i in enumerate(Init2.nodes):
        nodes.append(i)
    G_new = nx.Graph()
    G_new.add_nodes_from(nodes)
    for edge in Init2.get_edges():
        G_new.add_edge(edge[0], edge[1])

    print('clustering coefficient of random graph: {} vs des:{}'.format(nx.average_clustering(G_new.to_undirected()),
                                                                        des_cluster))

    objective = Objective({'avg degree': partial(avg_deg_count, des_degree),
                           'cluster coef': partial(avg_cluster_count, des_cluster)}, is_multi_objective=True)
    # ,

    # Setup optimization parameters
    max_graph_size = des_num_nodes
    requirements = GraphRequirements(
        max_arity=max_graph_size,
        max_depth=max_graph_size * 10000,
        num_of_generations=600,
        early_stopping_iterations=100,
        timeout=timedelta(minutes=timeout),
        n_jobs=-1,
        num_edges=num_edges
    )

    mutation_types = [MutationTypesEnum.single_edge,
                      MutationTypesEnum.batch_edge_5,
                      ]

    if dense:
        mutation_types.append(MutationTypesEnum.dense_edge_5)
    if star:
        mutation_types.append(MutationTypesEnum.star_edge_5)
    if cycle:
        mutation_types.append(MutationTypesEnum.cycle_edge_5)
    if path:
        mutation_types.append(MutationTypesEnum.path_edge_5)

    gp_params = GPAlgorithmParameters(
        max_pop_size=10,
        crossover_prob=0.8,
        mutation_prob=1,
        selection_types = selection_type,
        elitism_type = elitism_type,
        regularization_type=regularization_type,
        genetic_scheme_type=genetic_scheme_type,
       # genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free,
        multi_objective = objective.is_multi_objective,
        mutation_types = mutation_types,
        adaptive_mutation_type = adaptive_mutation_type, #MutationAgentTypeEnum.default,
        context_agent_type=ContextAgentTypeEnum.nodes_num,
        crossover_types=[CrossoverTypesEnum.none]
    )

    graph_gen_params = GraphGenerationParams(adapter=BaseNetworkxAdapter())
    all_parameters = (requirements, graph_gen_params, gp_params)

    # Build and run the optimizer
    time_before = datetime.now()
    optimiser = EvoGraphOptimizer(objective, initial_graphs, *all_parameters)
    found_graphs = optimiser.optimise(objective)
    history = optimiser.history
    #history.show.fitness_line()
    #    print('i have showed the history')
    time_after = datetime.now()
   # Restore the NetworkX graph back from internal Graph representation
    found_graph = graph_gen_params.adapter.restore(found_graphs[0])
    act_ad = np.mean(list(dict(found_graph.degree()).values()))
    print('avg degree of found graph real: {} vs des:{}'.format(act_ad,
                                                                des_degree))

    nodes = []
    colors = []
    for num, i in enumerate(found_graph.nodes):
        nodes.append(i)

    G_new = nx.Graph()
    G_new.add_nodes_from(nodes)
    for edge in found_graph.get_edges():
        G_new.add_edge(edge[0], edge[1])

    G_new.add_edges_from(found_graph.get_edges())
    act_cl = nx.average_clustering(G_new.to_undirected())
    print('clustering coefficient real: {} vs des:{}'.format(act_cl,
                                                             des_cluster))
    return time_after-time_before, act_cl, act_ad, history
