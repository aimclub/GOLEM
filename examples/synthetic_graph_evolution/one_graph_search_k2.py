import collections
import math
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
    def __init__(self, nodes: Optional[Union[LinkedGraphNode, List[LinkedGraphNode]]] = None,codes_in=None,nlevels=3):
        super().__init__(nodes)
        self.nodes = nodes
        self.unique_pipeline_id = 1
        self.nlevels = nlevels
        self.Nverts = pow(2, nlevels)
        c = 1
        self.sizes = []
        for _ in range(nlevels):
            self.sizes.append(c)
            c *= 2
        self.patterns = [[], [(0, 0)], [(0, 0), (1, 1)], [(0, 1), (1, 0)], [(0, 0), (1, 0), (1, 1)],
                         [(0, 0), (0, 1), (1, 0), (1, 1)]]
        self.weights = [pow(k, 0.3) for k in list(range(1, 7))]

        if codes_in is None:
            codes = []
            for i in range(nlevels):
                for _ in range(100):
                    cd = self.gen_pattern(self.sizes[i] * self.sizes[i])
                    if any(cd):
                        break
                codes.append(cd)
            self.codes = codes
        else:
            self.codes = codes_in

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

    def gen_pattern(self, k):
        code_symbols = list(range(len(self.patterns)))
        return random.choices(code_symbols, self.weights, k=k)
    def code_to_graph(self, p_prev, ilevel, coords):
        for e in self.patterns[p_prev]:
            if not e:
                continue
            i, j = e
            coords_new = (coords[0] + i * self.sizes[-ilevel], coords[1] + j * self.sizes[-ilevel])
            if ilevel == self.nlevels:
                self.edges.append(coords_new)
            else:
                ic = i * self.sizes[ilevel] + j
                p_new = self.codes[ilevel][ic]
                self.code_to_graph(p_new, ilevel + 1, coords_new)

    def gen_edges(self):
        self.edges = []
        self.code_to_graph(self.codes[0][0], 1, (0, 0))
        if not self.edges:
            return np.array([])
        edge_list = np.array(self.edges).T
        c, r = edge_list
        edge_list = edge_list[:, c > r]  # take only low triangular matrix
        edge_list = np.concatenate((edge_list, edge_list[::-1, :]), axis=1)

        for edge in edge_list.transpose():
            node_1, node_2 = edge
            for node in self.nodes:

                if node.name == str(node_1):
                    V1=node
                elif node.name==str(node_2):
                    V2=node
            self.connect_nodes(V1,V2)

        return edge_list

    def mutate(self):
        # choose code level
        lev = random.randint(0, self.nlevels - 1)
        len_lev = len(self.codes[lev])

        # choose element
        n_to_change = (lev - 1) * 2
        # n_to_change = 1 # predefined!!!!!!!!
        if lev < 2:
            n_to_change = 1

        new_codes = [c.copy() for c in self.codes]
        for _ in range(100):
            new_els = self.gen_pattern(n_to_change)
            new_code = new_codes[lev].copy()
            for i, v in enumerate(random.sample(range(len_lev), k=n_to_change)):
                new_code[v] = new_els[i]
            if any(new_code):
                new_codes[lev] = new_code
                break

        return GeneratorModel(nodes = self.nodes ,codes_in=new_codes,nlevels=self.nlevels)

def gen_with_deg(nodes, deg_exp, delta = 0.3,nlvl=3):
    deg_curr = 0
    for _ in range(1000):
        gen = GeneratorModel(nodes = nodes, nlevels=nlvl)
        ee = gen.gen_edges()
        if ee.size == 0:
            continue

        deg_curr = ee.shape[1]/gen.Nverts
        if abs(deg_curr - deg_exp)<delta:
            break
    return gen



class GeneratorNode(LinkedGraphNode):
    def __str__(self):
        return self.content["name"]


def run_graph_search(dense, cycle, path, star, size, num_edges, des_degree, des_cluster, des_num_nodes,
                     des_label_assort, des_asp, timeout=8, visualize=True):
    print('datetime', datetime.now())

    # Generate target graph that will be sought by optimizer

    def shortest_paths(G):
        d = datetime.now()
        avg_shortes_path = 0
        connected_components = 0
        for nodes in nx.connected_components(G):
            connected_components += 1
            g = G.subgraph(list(nodes))
            g_ig = ig.Graph.from_networkx(g)
            num_nodes = g.number_of_nodes()
            avg = 0
            for shortest_paths in g_ig.shortest_paths():
                for sp in shortest_paths:
                    avg += sp
            if num_nodes != 1:
                avg_shortes_path += avg / (num_nodes * num_nodes - num_nodes)
            else:
                avg_shortes_path = avg

        avg_s_p = avg_shortes_path / connected_components
        return avg_s_p

    def shortest_paths_2(G):
        d = datetime.now()
        avg_shortes_path = 0
        connected_components = 0
        for nodes in nx.connected_components(G):
            connected_components += 1
            g = G.subgraph(list(nodes))
            #g_ig = ig.Graph.from_networkx(g)
            avg_shortes_path += nx.average_shortest_path_length(g)



        avg_s_p = avg_shortes_path / connected_components
        print('time for asp', datetime.now() - d )
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

        fact_asp = shortest_paths(G_new)
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

    # Generate initial population with random graphs
    initial_graphs = []


    for i in range(20):
        print(i)
        Init2=gen_with_deg(nodes=[GeneratorNode(nodes_from=[],
                                                    content={'name': vertex,
                                                             'label': random.choices([0, 1], weights=[
                                                                 0.5 + 0.5 * des_label_assort,
                                                                 0.5 - 0.5 * des_label_assort], k=1)})
                                      for vertex in range(des_num_nodes)], deg_exp=des_degree, nlvl=int(math.log(des_num_nodes,2)))

        initial_graphs.append(Init2)


    print('avg degree of random graph: {} vs des:{}'.format(np.mean(list(dict(Init2.degree()).values())), des_degree))

    nodes = []
    for num, i in enumerate(Init2.nodes):
        nodes.append(num)


    G_new = nx.Graph()
    G_new.add_nodes_from(nodes)
    for edge in Init2.get_edges():
        G_new.add_edge(int(edge[0].name), int(edge[1].name))

    print('connected_components',nx.number_connected_components(G_new))
    print('clustering coefficient of random graph: {} vs des:{}'.format(nx.average_clustering(G_new.to_undirected()),
                                                                        des_cluster))
    print('shortest paths real: {} vs des: {}'.format(shortest_paths(G_new.to_undirected()), des_asp))
    objective = Objective({'avg degree': partial(avg_deg_count, des_degree),
                           'cluster coef': partial(avg_cluster_count, des_cluster),
                           'shortest paths': partial(asp_count, des_asp)}, is_multi_objective=True)



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

                      #MutationTypesEnum.batch_edge_3,
                      ]



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
    time_before = datetime.now()
    optimiser = EvoGraphOptimizer(objective, initial_graphs, *all_parameters)
    found_graphs = optimiser.optimise(objective)
    time_after = datetime.now()

    if visualize:
        # Restore the NetworkX graph back from internal Graph representation
        found_graph = graph_gen_params.adapter.restore(found_graphs[0])
        act_ad = np.mean(list(dict(found_graph.degree()).values()))
        print('avg degree of found graph real: {} vs des:{}'.format(act_ad,
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

        act_cl = nx.average_clustering(G_new.to_undirected())
        act_asp = shortest_paths(G_new.to_undirected())




        print('clustering coefficient real: {} vs des:{}'.format(act_cl, des_cluster))
       # print('label assortativity real: {} vs des: {} '.format(label_assortativity(found_graphs[0]), des_label_assort))
        print('shortest paths real: {} vs des: {}'.format(act_asp, des_asp))
        print('number of nodes', G_new.number_of_nodes())
        return G_new, time_after, time_before, act_cl, act_asp,act_ad
