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


def run_graph_search(dense, cycle, path, star, size, num_edges, des_degree, des_cluster, des_num_nodes,
                     des_label_assort, des_asp, timeout=15, visualize=True):
    print('datetime', datetime.now())

    # Generate target graph that will be sought by optimizer

    def overall_mape(des_values, fact_values):
        mape = 0
        for i, value in enumerate(des_values):
            mape += np.abs(value - fact_values[i]) / value
        return mape / len(des_values)

    def shortes_paths(G):

        avg_shortes_path = 0
        connected_components = 0
        for nodes in nx.connected_components(G):
            connected_components += 1
            g = G.subgraph(list(nodes))
            g_ig = ig.Graph.from_networkx(g)
            num_nodes = g.number_of_nodes()
            avg = 0
            for shortes_paths in g_ig.shortest_paths():
                for sp in shortes_paths:
                    avg += sp
            if num_nodes != 1:
                avg_shortes_path += avg / (num_nodes * num_nodes - num_nodes)
            else:
                avg_shortes_path = avg

        avg_s_p = avg_shortes_path / connected_components
        return avg_s_p

    def label_assortativity(G):
        label_assort = 0
        edges = G.get_edges()
        dic = {}
        for num, i in enumerate(G.nodes):
            dic[int(i.name)] = []

        edges = np.array(edges)
        edges = np.transpose(edges)

        if len(edges) > 0:
            edge_1, edge_2 = edges.tolist()
            for i in G.nodes:
                s_l = 0
                t = 0
                ind_1 = []
                ind_2 = []
                for l, node in enumerate(edge_1):
                    if node == i:
                        ind_1.append(l)
                    if edge_2[l] == i:
                        ind_2.append(l)

                result_1 = np.array(edge_2)[ind_1]
                result_2 = np.array(edge_1)[ind_2]

                for neigbour in result_1:
                    t += 1
                    if (neigbour.content['label'] == i.content["label"]):
                        s_l += 1
                for neigbour in result_2:
                    t += 1
                    if (neigbour.content['label'] == i.content["label"]):
                        s_l += 1
                if t > 0:
                    label_assort += s_l / t

            label_assort = label_assort / len(G.nodes)
        return label_assort

    def lab_assort_count(des_label_assort, G):
        fact_label = label_assortativity(G)
        return (des_label_assort - fact_label) * (des_label_assort - fact_label)

    def asp_count(des_shortest_paths, G):

        G_new = nx.Graph()
        G_new.add_nodes_from(G.nodes)
        G_new.add_edges_from(G.get_edges())

        fact_asp = shortes_paths(G_new)
        return (des_shortest_paths - fact_asp) * (des_shortest_paths - fact_asp)

    def avg_deg_count(des_d, G):
        d = np.mean(list(G.degree().values()))
        return (d - des_d) * (d - des_d)

    def avg_cluster_count(des_cl, G):
        G_new = nx.Graph()

        G_new.add_nodes_from(G.nodes)
        G_new.add_edges_from(G.get_edges())

        d = nx.average_clustering(G_new.to_undirected())
        return (d - des_cl) * (d - des_cl)

    def normalize(weights):
        total = sum(weights)
        return [w / total for w in weights]

    distributions = [
        lambda: [1 / des_num_nodes] * des_num_nodes,

    ]

    # Generate initial population with random graphs
    initial_graphs = []

    print(distributions)
    for i in range(20):
        Init2 = GeneratorModel(nodes=[GeneratorNode(nodes_from=[],
                                                    content={'name': vertex,
                                                             'label': random.choices([0, 1], weights=[
                                                                 0.5 + 0.5 * des_label_assort,
                                                                 0.5 - 0.5 * des_label_assort], k=1)})
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
    colors = []
    for num, i in enumerate(Init2.nodes):
        nodes.append(i)
        if i.content['label'] == 0:
            colors.append('red')
        else:
            colors.append('blue')
    G_new = nx.Graph()
    G_new.add_nodes_from(nodes)
    for edge in Init2.get_edges():
        G_new.add_edge(edge[0], edge[1])

    print('clustering coefficient of random graph: {} vs des:{}'.format(nx.average_clustering(G_new.to_undirected()),
                                                                        des_cluster))

    objective = Objective({'avg degree': partial(avg_deg_count, des_degree),
                           'cluster coef': partial(avg_cluster_count, des_cluster),
                           'label assort': partial(lab_assort_count, des_label_assort),
                           'shortest paths': partial(asp_count, des_asp)}, is_multi_objective=True)

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

                      MutationTypesEnum.star_edge_5,

                      MutationTypesEnum.path_edge_5,


                      MutationTypesEnum.cycle_edge_5,

                      MutationTypesEnum.dense_edge_5,
                      ]

    if dense:
        mutation_types.append(MutationTypesEnum.dense_edge)
    if star:
        mutation_types.append(MutationTypesEnum.star_edge)
    if cycle:
        mutation_types.append(MutationTypesEnum.cycle_edge)
    if path:
        mutation_types.append(MutationTypesEnum.path_edge)

    gp_params = GPAlgorithmParameters(
        max_pop_size=10,
        crossover_prob=0.8,
        mutation_prob=1,
        genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
        multi_objective=objective.is_multi_objective,
        mutation_types=mutation_types,
        adaptive_mutation_type=MutationAgentTypeEnum.default,
        context_agent_type=ContextAgentTypeEnum.adjacency_matrix,

        crossover_types=[CrossoverTypesEnum.none]
    )

    graph_gen_params = GraphGenerationParams(adapter=BaseNetworkxAdapter())
    all_parameters = (requirements, graph_gen_params, gp_params)

    # Build and run the optimizer
    optimiser = EvoGraphOptimizer(objective, initial_graphs, *all_parameters)
    found_graphs = optimiser.optimise(objective)

    if visualize:
        # Restore the NetworkX graph back from internal Graph representation
        found_graph = graph_gen_params.adapter.restore(found_graphs[0])
        print('avg degree of found graph real: {} vs des:{}'.format(np.mean(list(dict(found_graph.degree()).values())),
                                                                    des_degree))

        nodes = []
        colors = []
        for num, i in enumerate(found_graph.nodes):
            nodes.append(i)
            if i.content['label'] == 0:
                colors.append('red')
            else:
                colors.append('blue')
        G_new = nx.Graph()
        G_new.add_nodes_from(nodes)
        for edge in found_graph.get_edges():
            G_new.add_edge(edge[0], edge[1])

        G_new.add_edges_from(found_graph.get_edges())

        print('clustering coefficient real: {} vs des:{}'.format(nx.average_clustering(G_new.to_undirected()),
                                                                 des_cluster))
        print('label assortativity real: {} vs des: {} '.format(label_assortativity(found_graphs[0]), des_label_assort))
        print('shortest paths real: {} vs des: {}'.format(shortes_paths(G_new.to_undirected()), des_asp))

        return G_new